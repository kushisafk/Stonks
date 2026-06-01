import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.agent.pipeline import TradingAgent
from src.ensemble.weighted_voting import WeightedEnsemble
from src.models.finbert import FinBERTModel
from src.models.random_forest import RandomForestModel

@patch("src.agent.pipeline.market_data_service.fetch_data")
@patch("src.data.news_data.news_collector.get_news")
@patch("src.sentiment.sentiment_analyzer.sentiment_analyzer.analyze_batch")
def test_trading_agent_pipeline_with_sentiment(mock_analyze, mock_get_news, mock_fetch):
    """Verify that TradingAgent executes prediction and explanation incorporating news sentiment."""
    # 1. Mock market data: 120 days of mock prices
    dates = pd.date_range(start="2026-01-01", periods=120, freq="D")
    prices = np.linspace(100.0, 160.0, 120)
    mock_df = pd.DataFrame({
        "Open": prices,
        "High": prices + 1.0,
        "Low": prices - 1.0,
        "Close": prices,
        "Volume": 500.0
    }, index=dates)
    mock_df.index.name = "Date"
    mock_fetch.return_value = mock_df
    
    # 2. Mock news articles
    mock_get_news.return_value = [
        {"headline": "AAPL surges", "summary": "AAPL beating expectations.", "published_at": "2026-06-01T12:00:00", "source": "Bloomberg"}
    ]
    
    # 3. Mock analyzed articles
    mock_analyze.return_value = [
        {
            "headline": "AAPL surges", 
            "summary": "AAPL beating expectations.", 
            "published_at": "2026-06-01T12:00:00", 
            "source": "Bloomberg",
            "sentiment": {"positive": 0.85, "neutral": 0.1, "negative": 0.05}
        }
    ]
    
    # Instantiate agent using temp directory for model files
    agent = TradingAgent()
    
    # Run the pipeline (force_train=True to fit RF and save to directory)
    result = agent.run_pipeline("AAPL", force_train=True)
    
    # Assert return structure
    assert result["symbol"] == "AAPL"
    assert "signal" in result
    assert "confidence" in result
    assert result["close_price"] == 160.0
    assert "explanation" in result
    assert "probabilities" in result
    assert "rf" in result["probabilities"]
    assert "finbert" in result["probabilities"]
    assert "timestamp" in result
    
    # Check that explanation contains sentiment score and article counts
    explanation = result["explanation"].lower()
    assert "sentiment" in explanation
    assert "articles" in explanation

def test_ensemble_voting_rf_and_finbert():
    """Verify that WeightedEnsemble correctly aggregates RF price probabilities and FinBERT sentiment signals."""
    rf_model = MagicMock(spec=RandomForestModel)
    rf_model.predict_proba.return_value = np.array([0.60])
    
    finbert_model = MagicMock(spec=FinBERTModel)
    finbert_model.predict_proba.return_value = np.array([0.80])
    
    ensemble = WeightedEnsemble()
    ensemble.register_model("rf", rf_model)
    ensemble.register_model("finbert", finbert_model)
    
    # Set weights to 0.70 and 0.30
    ensemble.set_weight("rf", 0.70)
    ensemble.set_weight("finbert", 0.30)
    
    X = pd.DataFrame({"dummy": [1.0]})
    
    # Expected: 0.70 * 0.60 + 0.30 * 0.80 = 0.42 + 0.24 = 0.66
    prob = ensemble.predict_proba(X)
    assert prob[0] == pytest.approx(0.66)
    
    # Check individual probabilities breakdown
    breakdown = ensemble.get_individual_probabilities(X)
    assert breakdown["rf"][0] == 0.60
    assert breakdown["finbert"][0] == 0.80
