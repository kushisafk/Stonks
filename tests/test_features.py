import pandas as pd
import numpy as np
from unittest.mock import patch
from src.features.technical import add_rsi, add_bollinger_bands, add_macd
from src.features.statistical import add_rolling_skew, add_rolling_kurt
from src.features.feature_pipeline import FeaturePipeline

def test_indicators_math():
    """Verify that indicators calculate correct boundaries and relative levels."""
    # Create 60 days of oscillating dummy stock data
    dates = pd.date_range(start="2026-01-01", periods=60, freq="D")
    prices = np.sin(np.linspace(0, 10, 60)) * 50 + 150
    df = pd.DataFrame({
        "Open": prices, 
        "High": prices + 2.0, 
        "Low": prices - 2.0, 
        "Close": prices, 
        "Volume": 1000.0
    }, index=dates)
    df.index.name = "Date"
    
    # Test RSI boundaries
    df_rsi = add_rsi(df, period=14)
    assert "rsi" in df_rsi.columns
    valid_rsi = df_rsi["rsi"].dropna()
    assert len(valid_rsi) > 0
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all()
    
    # Test Bollinger Bands ordering: Upper > Middle > Lower
    df_bb = add_bollinger_bands(df)
    assert "bb_upper" in df_bb.columns
    assert "bb_middle" in df_bb.columns
    assert "bb_lower" in df_bb.columns
    bb_nonnull = df_bb.dropna(subset=["bb_upper", "bb_middle", "bb_lower"])
    assert len(bb_nonnull) > 0
    assert (bb_nonnull["bb_upper"] >= bb_nonnull["bb_middle"]).all()
    assert (bb_nonnull["bb_middle"] >= bb_nonnull["bb_lower"]).all()

def test_feature_pipeline_store(tmp_path):
    """Verify that FeaturePipeline writes to store, labels target, and handles prediction offsets."""
    # Create 60 days of strictly increasing prices
    dates = pd.date_range(start="2026-01-01", periods=60, freq="D")
    prices = np.linspace(100.0, 160.0, 60)
    raw_df = pd.DataFrame({
        "Open": prices, 
        "High": prices + 1.0, 
        "Low": prices - 1.0, 
        "Close": prices, 
        "Volume": 500.0
    }, index=dates)
    raw_df.index.name = "Date"
    
    pipeline = FeaturePipeline(store_dir=tmp_path)
    
    # 1. Test training mode (must drop the last row as its target is NaN)
    X_train, y_train = pipeline.get_features("MSFT", raw_df, is_training=True, use_store=True)
    
    # Verify that file was saved to store
    store_file = tmp_path / "MSFT.csv"
    assert store_file.exists()
    
    # Length check: 60 total rows - 49 rows (MA50 warm up) - 1 row (target look-ahead NaN) = 10 rows
    assert len(X_train) == 10
    assert len(y_train) == 10
    # Since prices strictly increase, target should be 1 (price increase)
    assert (y_train == 1).all()
    
    # 2. Test prediction mode (must preserve the very last row!)
    X_pred, y_pred = pipeline.get_features("MSFT", raw_df, is_training=False, use_store=True)
    
    # Length check: 60 total rows - 49 rows (MA50 warm up) = 11 rows (1 more than training!)
    assert len(X_pred) == 11
    # The last row must represent our last raw index date
    assert X_pred.index[-1] == pd.to_datetime("2026-03-01")
    # Verify there are absolutely no NaNs in features
    assert not X_pred.isnull().any().any()


@patch("src.data.market_data.market_data_service.fetch_data")
def test_phase3_features_calculation(mock_fetch, tmp_path):
    """Verify that all 17 Phase 3 features are correctly calculated and index-aligned without NaNs."""
    # Create 120 days of dummy price and volume data for the stock
    dates = pd.date_range(start="2026-01-01", periods=120, freq="D")
    prices = np.linspace(100.0, 150.0, 120)
    
    # 2.0x volume spike on the 100th day to trigger abnormal volume flag
    volumes = np.ones(120) * 1000.0
    volumes[100] = 3000.0
    
    stock_df = pd.DataFrame({
        "Open": prices, 
        "High": prices + 1.0, 
        "Low": prices - 1.0, 
        "Close": prices, 
        "Volume": volumes
    }, index=dates)
    stock_df.index.name = "Date"
    
    # Create mock SPY data: strictly increasing (Bull market regime: Close > MA50 > MA100)
    spy_prices = np.linspace(400.0, 550.0, 120)
    spy_df = pd.DataFrame({
        "Open": spy_prices,
        "High": spy_prices + 1.0,
        "Low": spy_prices - 1.0,
        "Close": spy_prices,
        "Volume": 2000.0
    }, index=dates)
    spy_df.index.name = "Date"
    
    mock_fetch.return_value = spy_df
    
    pipeline = FeaturePipeline(store_dir=tmp_path)
    X, y = pipeline.get_features("AAPL", stock_df, is_training=True, use_store=False)
    
    # 17 new Phase 3 features to assert
    added_cols = [
        "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_rsi", "spy_macd",
        "spy_volatility_20d", "spy_trend_strength", "relative_strength_5d",
        "relative_strength_20d", "relative_strength_50d", "relative_momentum_score",
        "volume_sma_20", "volume_ratio", "volume_momentum", "volume_trend",
        "abnormal_volume_flag", "market_regime"
    ]
    
    # Check that all features exist in X
    for col in added_cols:
        assert col in X.columns, f"{col} missing from computed feature columns"
        
    # Check that there are no NaNs in features
    assert not X[added_cols].isnull().any().any(), "NaNs detected in Phase 3 feature columns"
    
    # Validate volume spike abnormal flag triggers
    # Shift index because warmup/target drop might affect row location
    # 100th index is "2026-04-11". Let's check if the date is in X and what its flag is
    spike_date = dates[100]
    if spike_date in X.index:
        assert X.loc[spike_date, "abnormal_volume_flag"] == 1.0, "Abnormal volume flag failed to trigger on volume spike"
        
    # Validate market regime (strictly rising SPY should be a bull market = 1.0)
    # Check later dates where MA50 and MA100 have warmed up
    warm_date = dates[110]
    if warm_date in X.index:
        assert X.loc[warm_date, "market_regime"] == 1.0, "Market regime classification failed to detect bull market"

