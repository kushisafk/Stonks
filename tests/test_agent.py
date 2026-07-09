import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from stonks.agent.pipeline import TradingAgent

@patch("stonks.data.market_data.market_data_service.fetch_data")
def test_trading_agent_pipeline(mock_fetch, tmp_path):
    """Verify that TradingAgent coordinates data, training, inference, and logs successfully."""
    # Prepare dummy market data dataframe (120 days for indicator warmup and safety limits)
    dates = pd.date_range(start="2026-01-01", periods=120, freq="D")
    prices = np.linspace(100.0, 150.0, 120) + np.random.randn(120) * 1.5
    dummy_df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": 1000.0
    }, index=dates)
    dummy_df.index.name = "Date"
    
    mock_fetch.return_value = dummy_df
    
    # Configure custom directories on agent to avoid dirtying workspace
    agent = TradingAgent(model_dir=tmp_path / "models")
    
    # Configure global decisions CSV inside the temp directory
    test_csv = tmp_path / "decisions.csv"
    from stonks.logging.logger import decision_logger
    decision_logger.csv_path = test_csv
    decision_logger._initialize_csv()
    
    # Run pipeline with forced training
    result = agent.run_pipeline("AAPL", force_train=True)
    
    # 1. Assert return payload structure and parameters
    assert result["symbol"] == "AAPL"
    assert result["signal"] in ["BUY", "SELL", "HOLD"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["close_price"] == pytest.approx(prices[-1])
    assert "explanation" in result
    assert "rf" in result["probabilities"]
    assert "timestamp" in result
    
    # 2. Verify model weights were serialized to model_dir
    expected_model_file = tmp_path / "models" / "AAPL_RF.joblib"
    assert expected_model_file.exists()
    
    # 3. Verify CSV decision logger appended the decision row
    assert test_csv.exists()
    df_csv = pd.read_csv(test_csv)
    assert len(df_csv) == 1
    assert df_csv.iloc[0]["Ticker"] == "AAPL"
    assert df_csv.iloc[0]["Signal"] == result["signal"]
    assert df_csv.iloc[0]["Confidence"] == pytest.approx(result["confidence"], abs=1e-3)
    assert df_csv.iloc[0]["ClosePrice"] == pytest.approx(prices[-1])
