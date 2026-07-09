import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
from stonks.config.settings import settings
from stonks.logging.logger import logger
from stonks.features.technical import (
    add_returns,
    add_moving_averages,
    add_emas,
    add_volatility,
    add_rsi,
    add_macd,
    add_bollinger_bands
)
from stonks.features.statistical import add_rolling_skew, add_rolling_kurt

class FeaturePipeline:
    """Orchestrates features generation, targets creation, and caching feature datasets to a local Feature Store."""
    
    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or settings.FEATURE_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_store_path(self, symbol: str) -> Path:
        return self.store_dir / f"{symbol.strip().upper()}.csv"
        
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all technical and statistical computations in pipeline sequence."""
        df = df.copy()
        
        # Apply technical indicator transformations
        df = add_returns(df)
        df = add_moving_averages(df)
        df = add_emas(df)
        df = add_volatility(df)
        df = add_rsi(df)
        df = add_macd(df)
        df = add_bollinger_bands(df)
        
        # Apply statistical transformations
        df = add_rolling_skew(df)
        df = add_rolling_kurt(df)
        
        return df

    def get_features(
        self, 
        symbol: str, 
        raw_df: pd.DataFrame, 
        is_training: bool = True,
        use_store: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Coordinates feature generation and target labeling.
        Supports caching and retrieval from the Feature Store.
        
        Args:
            symbol: Ticker symbol
            raw_df: Raw historical dataframe from MarketDataService
            is_training: If True, computes target and drops the last row (which lacks target).
                         If False, computes features up to the last row for prediction.
            use_store: If True, uses feature_store/ to cache and load precalculated features.
            
        Returns:
            Tuple[X, y]: X is a DataFrame of features, y is a Series of target labels (or None for prediction).
        """
        symbol = symbol.strip().upper()
        store_path = self._get_store_path(symbol)
        
        # Check Feature Store cache
        df = None
        if use_store and store_path.exists():
            try:
                cached_features = pd.read_csv(store_path, parse_dates=["Date"])
                cached_features.set_index("Date", inplace=True)
                cached_features.index = pd.to_datetime(cached_features.index)
                
                # Check if raw data has the same length and end date as cached features
                # and verify that the cached file has Phase 3 columns (e.g. 'spy_return_1d')
                if (len(raw_df) == len(cached_features) and 
                    raw_df.index[-1] == cached_features.index[-1] and
                    "spy_return_1d" in cached_features.columns):
                    logger.info(f"Feature Store hit: Loading precomputed features for {symbol} from {store_path}")
                    df = cached_features
                else:
                    logger.info(f"Feature Store outdated or legacy for {symbol}: Recomputing features.")
            except Exception as e:
                logger.warning(f"Error loading from Feature Store for {symbol}, recomputing: {e}")
                
        if df is None:
            logger.info(f"Computing features for {symbol} (no active cache found).")
            df = self.compute_all_features(raw_df)
            
            # Step 1: SPY Ingestion and Alignment
            try:
                from stonks.data.market_data import market_data_service
                logger.info(f"Phase 3 Features: Ingesting SPY benchmark context for {symbol}...")
                
                # Fetch SPY data matching ticker range
                spy_df = market_data_service.fetch_data("SPY", period=settings.YFINANCE_PERIOD, interval=settings.YFINANCE_INTERVAL)
                spy_df = spy_df.reindex(raw_df.index, method="ffill")
                
                # Calculate SPY base returns, rsi, macd, volatility
                from stonks.features.technical import add_returns, add_rsi, add_macd, add_volatility
                spy_df = add_returns(spy_df)
                spy_df = add_rsi(spy_df)
                spy_df = add_macd(spy_df)
                spy_df = add_volatility(spy_df)
                
                spy_ma50 = spy_df["Close"].rolling(50).mean()
                spy_ma100 = spy_df["Close"].rolling(100).mean()
                
                # Append SPY features
                df["spy_return_1d"] = spy_df["daily_return"]
                df["spy_return_5d"] = spy_df["return_5d"]
                df["spy_return_20d"] = spy_df["return_20d"]
                df["spy_rsi"] = spy_df["rsi"]
                df["spy_macd"] = spy_df["macd"]
                df["spy_volatility_20d"] = spy_df["volatility_20d"]
                df["spy_trend_strength"] = (spy_df["Close"] - spy_ma50) / (spy_ma50 + 1e-9)
                
                # Step 2: Relative Strength Features
                df["relative_strength_5d"] = df["return_5d"] - df["spy_return_5d"]
                df["relative_strength_20d"] = df["return_20d"] - df["spy_return_20d"]
                
                stock_return_50d = df["Close"].pct_change(50)
                spy_return_50d = spy_df["Close"].pct_change(50)
                df["relative_strength_50d"] = stock_return_50d - spy_return_50d
                df["relative_momentum_score"] = (df["relative_strength_5d"] + df["relative_strength_20d"] + df["relative_strength_50d"]) / 3.0
                
                # Step 3: Volume Intelligence
                vol_sma_20 = df["Volume"].rolling(20).mean()
                df["volume_sma_20"] = vol_sma_20
                df["volume_ratio"] = df["Volume"] / (vol_sma_20 + 1e-9)
                df["volume_momentum"] = df["Volume"].pct_change(5)
                df["volume_trend"] = (df["Volume"] - vol_sma_20) / (vol_sma_20 + 1e-9)
                df["abnormal_volume_flag"] = (df["volume_ratio"] > 2.0).astype(float)
                
                # Step 4: Market Regime Detection
                regime = np.zeros(len(df))
                bull_mask = (spy_df["Close"] > spy_ma50) & (spy_ma50 > spy_ma100)
                bear_mask = (spy_df["Close"] < spy_ma50) & (spy_ma50 < spy_ma100)
                
                regime[bull_mask] = 1.0
                regime[bear_mask] = -1.0
                df["market_regime"] = regime
                
                # Ensure no NaNs in added cols to prevent row drop
                added_cols = [
                    "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_rsi", "spy_macd",
                    "spy_volatility_20d", "spy_trend_strength", "relative_strength_5d",
                    "relative_strength_20d", "relative_strength_50d", "relative_momentum_score",
                    "volume_sma_20", "volume_ratio", "volume_momentum", "volume_trend",
                    "abnormal_volume_flag", "market_regime"
                ]
                df[added_cols] = df[added_cols].fillna(0.0)
                logger.info(f"Phase 3 Features computed successfully for {symbol}.")
                
            except Exception as e:
                logger.error(f"Failed to calculate Phase 3 features for {symbol}: {e}")
                raise e
                
            if use_store:
                try:
                    df.to_csv(store_path)
                    logger.info(f"Saved computed Phase 3 features for {symbol} to Feature Store at {store_path}")
                except Exception as e:
                    logger.error(f"Failed to write features to store: {e}")
                
        # Generate Target: 1 if tomorrow's Close > today's Close, else 0 (preserves NaN for the last row)
        tomorrow_close = df["Close"].shift(-1)
        df["Target"] = np.where(tomorrow_close.isna(), np.nan, (tomorrow_close > df["Close"]).astype(float))
        
        # Define base technical and statistical feature column list
        base_feature_cols = [
            "daily_return", "return_5d", "return_20d",
            "ma10", "ma20", "ma50", "ema20", "ema50",
            "volatility_20d", "rsi", "macd", "macd_signal",
            "bb_middle", "bb_upper", "bb_lower",
            "skew_20d", "kurt_20d"
        ]
        
        sentiment_cols = [
            "sentiment_score", "positive_news_ratio", "negative_news_ratio",
            "neutral_news_ratio", "article_count", "average_sentiment",
            "weighted_sentiment", "recency_weighted_sentiment"
        ]
        
        market_context_cols = [
            "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_rsi", "spy_macd",
            "spy_volatility_20d", "spy_trend_strength", "relative_strength_5d",
            "relative_strength_20d", "relative_strength_50d", "relative_momentum_score",
            "volume_sma_20", "volume_ratio", "volume_momentum", "volume_trend",
            "abnormal_volume_flag", "market_regime"
        ]
        
        # Initialize sentiment columns with neutral/zero values
        for col in sentiment_cols:
            if col not in df.columns:
                df[col] = 0.0
                
        # If live prediction (not training) and we have news, populate the latest row with real sentiment
        if not is_training:
            try:
                from stonks.data.news_data import news_collector
                from stonks.sentiment.sentiment_analyzer import sentiment_analyzer
                from stonks.sentiment.sentiment_features import aggregate_sentiment_features
                
                logger.info(f"Live Prediction: Collecting and analyzing news for {symbol}")
                raw_articles = news_collector.get_news(symbol, force_refresh=False)
                if raw_articles:
                    analyzed_articles = sentiment_analyzer.analyze_batch(raw_articles)
                    sentiment_features = aggregate_sentiment_features(analyzed_articles)
                    
                    # Update only the last row (current day) with the computed sentiment features
                    last_idx = df.index[-1]
                    for col, val in sentiment_features.items():
                        df.at[last_idx, col] = val
                    logger.info(f"Live Prediction: Successfully merged news sentiment features for {symbol}: {sentiment_features}")
                else:
                    logger.warning(f"Live Prediction: No news articles fetched for {symbol}. Keeping neutral sentiment features.")
            except Exception as e:
                logger.error(f"Failed to integrate live news sentiment features for {symbol}: {e}")
                
        feature_cols = base_feature_cols + sentiment_cols + market_context_cols
        
        if is_training:
            # For training, drop any row with NaNs (including start rows without indicator histories
            # and the very last row which lacks a tomorrow target).
            clean_df = df.dropna(subset=feature_cols + ["Target"])
            X = clean_df[feature_cols]
            y = clean_df["Target"]
            logger.info(f"Prepared training dataset for {symbol}: Shape {X.shape}")
            return X, y
        else:
            # For prediction, we need the last row (today). We must NOT drop it because of target NaN!
            # But we must drop initial rows which have NaN features (MA50 etc) to avoid feeding NaNs to the model.
            clean_df = df.dropna(subset=feature_cols)
            X = clean_df[feature_cols]
            y = clean_df["Target"]
            logger.info(f"Prepared prediction dataset for {symbol}: Shape {X.shape}")
            return X, y

# Global feature pipeline instance
feature_pipeline = FeaturePipeline()
