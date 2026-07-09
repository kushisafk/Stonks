import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from stonks.api.app import app

client = TestClient(app)

def test_health_endpoint():
    """Verify that the health-check route returns status 200 and correct environment metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data

def test_models_endpoint():
    """Verify that the models diagnostic route successfully returns weights and active stub counts."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_models" in data
    assert "total_active_weight" in data
    assert data["total_active_weight"] == pytest.approx(1.0)
    
    # Assert model info parameters exist
    rf_info = data["active_models"]["rf"]
    assert rf_info["is_active"] is True
    assert rf_info["status"] == "READY"
    assert rf_info["weight"] == 0.7
    
    lstm_info = data["active_models"]["lstm"]
    assert lstm_info["is_active"] is True
    assert lstm_info["status"] == "STUB"
    assert lstm_info["weight"] == 0.0
    
    finbert_info = data["active_models"]["finbert"]
    assert finbert_info["is_active"] is True
    assert finbert_info["status"] == "STUB"
    assert finbert_info["weight"] == 0.3

@patch("stonks.api.app.trading_agent.run_pipeline")
def test_predict_endpoint(mock_pipeline):
    """Verify that the predict ticker route successfully queries the agent and formats as DecisionResponse."""
    # Configure mock prediction payload
    mock_payload = {
        "symbol": "TSLA",
        "signal": "BUY",
        "confidence": 0.7645,
        "close_price": 200.50,
        "explanation": "BUY signal generated due to bullish MACD crossover and oversold RSI.",
        "probabilities": {"rf": 0.7645},
        "timestamp": "2026-06-01T12:00:00"
    }
    mock_pipeline.return_value = mock_payload
    
    response = client.get("/predict/TSLA?force_train=True")
    
    # Assert call arguments
    mock_pipeline.assert_called_once_with("TSLA", force_train=True)
    
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "TSLA"
    assert data["signal"] == "BUY"
    assert data["confidence"] == 0.7645
    assert data["close_price"] == 200.50
    assert "explanation" in data
    
@patch("stonks.api.app.market_data_service.fetch_data")
@patch("stonks.api.app.Backtester.run_walk_forward")
def test_backtest_endpoint(mock_run, mock_fetch):
    """Verify that the backtest RESEARCH route handles walk-forward requests and triggers logic."""
    # Configure mock market data and mock backtest results
    mock_fetch.return_value = MagicMock()
    mock_results = {
        "symbol": "MSFT",
        "ml_metrics": {
            "accuracy": 0.62,
            "precision": 0.65,
            "recall": 0.60,
            "f1": 0.62
        },
        "trading_metrics": {
            "starting_capital": 10000.0,
            "ending_capital": 12500.0,
            "strategy_return": 0.25,
            "buy_and_hold_return": 0.15,
            "win_rate": 0.58,
            "total_trades": 8,
            "annualized_volatility": 0.18,
            "sharpe_ratio": 1.28,
            "max_drawdown": -0.12
        }
    }
    mock_run.return_value = mock_results
    
    response = client.get("/backtest/MSFT?train_window=200&test_window=40")
    
    # Assert backtester called with customized query params
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert kwargs["train_window"] == 200
    assert kwargs["test_window"] == 40
    
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "MSFT"
    assert data["ml_metrics"]["accuracy"] == 0.62
    assert data["trading_metrics"]["sharpe_ratio"] == 1.28
    assert data["trading_metrics"]["max_drawdown"] == -0.12

@patch("stonks.data.news_data.news_collector.get_news")
@patch("stonks.sentiment.sentiment_analyzer.sentiment_analyzer.analyze_batch")
def test_sentiment_endpoint(mock_analyze, mock_get_news):
    """Verify that the sentiment API route successfully retrieves, analyzes, and aggregates news sentiment."""
    # Configure mock news articles
    mock_get_news.return_value = [
        {"headline": "AAPL surges on strong earnings", "summary": "AAPL beating expectations.", "published_at": "2026-06-01T12:00:00", "source": "Bloomberg"}
    ]
    
    # Configure mock analyzed articles
    mock_analyze.return_value = [
        {
            "headline": "AAPL surges on strong earnings", 
            "summary": "AAPL beating expectations.", 
            "published_at": "2026-06-01T12:00:00", 
            "source": "Bloomberg",
            "sentiment": {"positive": 0.9, "neutral": 0.05, "negative": 0.05}
        }
    ]
    
    response = client.get("/sentiment/AAPL")
    
    # Assert mock calls
    mock_get_news.assert_called_once_with("AAPL", force_refresh=False)
    mock_analyze.assert_called_once()
    
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "AAPL"
    # positive - negative = 0.9 - 0.05 = 0.85
    assert data["sentiment_score"] == pytest.approx(0.85)
    assert data["articles_analyzed"] == 1
    assert data["positive_ratio"] == 1.0
    assert data["negative_ratio"] == 0.0
