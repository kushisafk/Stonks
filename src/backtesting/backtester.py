import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.config.settings import settings
from src.logging.logger import logger
from src.features.feature_pipeline import feature_pipeline
from src.models.random_forest import RandomForestModel
from src.decision.decision_engine import decision_engine

class Backtester:
    """Simulates trading strategies using chronological walk-forward validation and realistic transaction frictions."""
    
    def __init__(
        self, 
        starting_capital: float = 10000.0,
        commission: Optional[float] = None,
        slippage: Optional[float] = None,
        risk_free_rate: float = 0.02
    ):
        self.starting_capital = starting_capital
        self.commission = commission if commission is not None else settings.COMMISSION
        self.slippage = slippage if slippage is not None else settings.SLIPPAGE
        self.risk_free_rate = risk_free_rate
        
    def run_walk_forward(
        self, 
        symbol: str, 
        raw_df: pd.DataFrame, 
        train_window: int = 250, 
        test_window: int = 50,
        model_name: str = "rf"
    ) -> Dict[str, Any]:
        """
        Executes a rolling walk-forward backtest on historical stock data.
        
        Args:
            symbol: Ticker symbol
            raw_df: Time-series dataframe of stock data
            train_window: Size of sliding training window (default 250 days)
            test_window: Size of testing window (default 50 days)
            model_name: "rf" for Random Forest only, or "ensemble" for RF + FinBERT ensemble
        """
        logger.info(f"Initiating walk-forward backtest ({model_name}) for {symbol}...")
        
        # Ingest and generate features
        X, y = feature_pipeline.get_features(symbol, raw_df, is_training=True, use_store=False)
        
        total_len = len(X)
        if total_len < train_window + test_window:
            raise ValueError(
                f"Dataset size {total_len} is too small for train={train_window} and test={test_window} configurations. "
                f"Needs at least {train_window + test_window} rows."
            )
            
        all_preds = []
        all_targets = []
        all_probs = []
        test_indices = []
        
        # Sliding chronological window split
        for i in range(train_window, total_len, test_window):
            if i + test_window > total_len:
                # If trailing fold is smaller than test_window, evaluate up to the end
                fold_test_end = total_len
            else:
                fold_test_end = i + test_window
                
            X_train = X.iloc[i - train_window:i]
            y_train = y.iloc[i - train_window:i]
            
            X_test = X.iloc[i:fold_test_end]
            y_test = y.iloc[i:fold_test_end]
            
            if len(X_test) == 0:
                break
                
            # Train a Random Forest model on this training window
            rf_model = RandomForestModel(n_estimators=100, max_depth=8, random_state=42)
            rf_model.train(X_train, y_train)
            
            # Predict using chosen model configuration
            model_name_lower = model_name.strip().lower()
            if model_name_lower == "rf":
                probs = rf_model.predict_proba(X_test)
                preds = rf_model.predict(X_test)
            elif model_name_lower in ("ensemble", "rf+finbert", "rf_finbert"):
                from src.ensemble.weighted_voting import WeightedEnsemble
                from src.models.finbert import FinBERTModel
                
                ensemble = WeightedEnsemble()
                ensemble.register_model("rf", rf_model)
                
                finbert_model = FinBERTModel()
                ensemble.register_model("finbert", finbert_model)
                
                # Configure weights
                ensemble.set_weight("rf", settings.RF_WEIGHT)
                ensemble.set_weight("finbert", settings.FINBERT_WEIGHT)
                
                probs = ensemble.predict_proba(X_test)
                preds = ensemble.predict(X_test)
            else:
                raise ValueError(f"Unsupported model_name for backtester: {model_name}")
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(y_test)
            test_indices.extend(X_test.index)
            
            if fold_test_end == total_len:
                break
                
        # Consolidate out-of-sample predictions
        eval_df = pd.DataFrame({
            "probability": all_probs,
            "prediction": all_preds,
            "target": all_targets
        }, index=test_indices)
        
        # Merge prices to match predictions
        eval_df = eval_df.join(raw_df["Close"], how="left")
        
        # Run strategy simulation
        sim_metrics = self.simulate_strategy(eval_df)
        
        # Calculate ML classification metrics
        ml_metrics = self.calculate_ml_metrics(np.array(all_preds), np.array(all_targets))
        
        return {
            "symbol": symbol,
            "ml_metrics": ml_metrics,
            "trading_metrics": sim_metrics
        }
        
    def calculate_ml_metrics(self, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculates standard classification metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        # zero_division handles edge cases of no positive predictions cleanly
        return {
            "accuracy": float(accuracy_score(targets, preds)),
            "precision": float(precision_score(targets, preds, zero_division=0)),
            "recall": float(recall_score(targets, preds, zero_division=0)),
            "f1": float(f1_score(targets, preds, zero_division=0))
        }
        
    def simulate_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs the asset simulation incorporating signals, slippage, and commissions."""
        capital = self.starting_capital
        position = 0.0  # shares held
        equity_curve = []
        trades = []  # list of trade returns to calculate win rate
        
        # Track buy and sell price histories for win-rate and execution
        last_buy_price = None
        
        # Retrieve thresholds from decision engine to match live agent signals
        engine = decision_engine
        
        for idx, row in df.iterrows():
            prob = row["probability"]
            close = row["Close"]
            
            # Get signal
            decision = engine.make_decision(prob)
            signal = decision["signal"]
            
            # Execute signals
            if signal == "BUY" and position == 0.0:
                # Buy asset: pay slippage and commissions
                buy_price = close * (1.0 + self.slippage)
                net_capital = capital * (1.0 - self.commission)
                position = net_capital / buy_price
                capital = 0.0
                last_buy_price = buy_price
                logger.debug(f"Backtester: BUY {position:.4f} shares at {buy_price:.2f}")
                
            elif signal == "SELL" and position > 0.0:
                # Sell asset: pay slippage and commissions
                sell_price = close * (1.0 - self.slippage)
                gross_proceeds = position * sell_price
                net_proceeds = gross_proceeds * (1.0 - self.commission)
                capital = net_proceeds
                position = 0.0
                
                # Record trade metrics
                trade_return = (sell_price / last_buy_price) - 1.0 - (2.0 * self.commission)
                trades.append(trade_return)
                last_buy_price = None
                logger.debug(f"Backtester: SELL shares at {sell_price:.2f}, Capital: {capital:.2f}")
                
            # Record daily equity value (cash + stock value)
            current_equity = capital if position == 0.0 else (position * close)
            equity_curve.append(current_equity)
            
        # End of simulation exit to cash if still holding position
        if position > 0.0:
            close = df.iloc[-1]["Close"]
            sell_price = close * (1.0 - self.slippage)
            gross_proceeds = position * sell_price
            capital = gross_proceeds * (1.0 - self.commission)
            if last_buy_price is not None:
                trade_return = (sell_price / last_buy_price) - 1.0 - (2.0 * self.commission)
                trades.append(trade_return)
            equity_curve[-1] = capital
            
        equity_series = pd.Series(equity_curve, index=df.index)
        
        # Calculate returns
        strategy_return = (capital / self.starting_capital) - 1.0
        
        # Buy & Hold Return
        initial_close = df.iloc[0]["Close"]
        final_close = df.iloc[-1]["Close"]
        # Include B&H entering and exiting friction once for realism
        bh_entry = initial_close * (1.0 + self.slippage)
        bh_exit = final_close * (1.0 - self.slippage)
        bh_return = (bh_exit / bh_entry) - 1.0 - (2.0 * self.commission)
        
        # Win Rate
        win_rate = 0.0
        if trades:
            wins = sum(1 for t in trades if t > 0)
            win_rate = wins / len(trades)
            
        # Annualized volatility and Sharpe Ratio
        daily_returns = equity_series.pct_change().dropna()
        ann_vol = daily_returns.std() * np.sqrt(252.0) if len(daily_returns) > 1 else 0.0
        
        # Annualized strategy return
        trading_days = len(df)
        years = trading_days / 252.0
        
        # CAGR calculation
        if years > 0 and capital > 0:
            ann_return = (capital / self.starting_capital) ** (1.0 / years) - 1.0
        else:
            ann_return = strategy_return
            
        sharpe = 0.0
        if ann_vol > 0:
            sharpe = (ann_return - self.risk_free_rate) / ann_vol
            
        # Maximum Drawdown
        peak = equity_series.cummax()
        drawdown = (equity_series - peak) / peak
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        
        return {
            "starting_capital": self.starting_capital,
            "ending_capital": float(capital),
            "strategy_return": float(strategy_return),
            "buy_and_hold_return": float(bh_return),
            "win_rate": float(win_rate),
            "total_trades": len(trades),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd)
        }
