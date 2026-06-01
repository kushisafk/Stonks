import pandas as pd
import numpy as np
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
