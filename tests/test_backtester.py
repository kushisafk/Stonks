import pytest
import pandas as pd
import numpy as np
from src.backtesting.backtester import Backtester

def test_backtester_simulation():
    """Verify order execution logic, win-rate calculation, and returns under zero friction."""
    # Setup a deterministic series of prices and signals to verify orders execution
    dates = pd.date_range(start="2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "probability": [0.30, 0.70, 0.50, 0.35, 0.50],  # signals: SELL, BUY, HOLD, SELL, HOLD
        "Close": [100.0, 100.0, 110.0, 120.0, 120.0],
    }, index=dates)
    
    # 0.0 commission and slippage for simple verification first
    backtester = Backtester(starting_capital=10000.0, commission=0.0, slippage=0.0)
    metrics = backtester.simulate_strategy(df)
    
    # Day 0: prob 0.30 -> SELL. We have no position, do nothing. Capital = 10000
    # Day 1: prob 0.70 -> BUY. We buy at 100.0 Close. Position = 100 shares. Cash = 0
    # Day 2: prob 0.50 -> HOLD. Keep shares. Equity = 100 * 110 = 11000
    # Day 3: prob 0.35 -> SELL. We sell at 120.0 Close. Cash = 100 * 120 = 12000. Shares = 0
    # Day 4: prob 0.50 -> HOLD. Keep cash. Capital = 12000
    
    assert metrics["ending_capital"] == 12000.0
    assert metrics["strategy_return"] == pytest.approx(0.20)
    assert metrics["total_trades"] == 1
    assert metrics["win_rate"] == 1.0

def test_backtester_friction():
    """Verify order execution and compounding costs under high slippage and commission settings."""
    dates = pd.date_range(start="2026-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "probability": [0.30, 0.70, 0.50, 0.35, 0.50],  # signals: SELL, BUY, HOLD, SELL, HOLD
        "Close": [100.0, 100.0, 110.0, 120.0, 120.0],
    }, index=dates)

    # Test with commission=0.01 (1%) and slippage=0.01 (1%) to verify friction calculations
    # Day 1: BUY close=100.0. Slippage makes buy_price = 100 * 1.01 = 101.0
    #        Commission makes net_capital = 10000 * 0.99 = 9900
    #        Shares bought = 9900 / 101.0 = 98.0198 shares
    # Day 3: SELL close=120.0. Slippage makes sell_price = 120 * 0.99 = 118.8
    #        Gross proceeds = 98.0198 * 118.8 = 11644.752
    #        Net proceeds = 11644.752 * 0.99 = 11528.30
    
    backtester_fric = Backtester(starting_capital=10000.0, commission=0.01, slippage=0.01)
    metrics_fric = backtester_fric.simulate_strategy(df)
    assert metrics_fric["ending_capital"] == pytest.approx(11528.30, abs=0.1)

def test_run_walk_forward():
    """Verify that walk-forward validation executes fully on a rolling dataset."""
    # Make a dummy dataset of 360 trading days to allow a 250 train / 50 test split after NaNs
    dates = pd.date_range(start="2025-01-01", periods=360, freq="D")
    prices = np.linspace(100.0, 160.0, 360) + np.random.randn(360) * 2
    raw_df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": 1000.0
    }, index=dates)
    raw_df.index.name = "Date"
    
    backtester = Backtester(starting_capital=10000.0, commission=0.001, slippage=0.0005)
    results = backtester.run_walk_forward("AAPL", raw_df, train_window=250, test_window=50)
    
    assert results["symbol"] == "AAPL"
    assert "ml_metrics" in results
    assert "trading_metrics" in results
    assert "accuracy" in results["ml_metrics"]
    assert "sharpe_ratio" in results["trading_metrics"]
    assert "max_drawdown" in results["trading_metrics"]
