import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from stonks.sentiment.sentiment_analyzer import SentimentAnalyzer
from stonks.sentiment.sentiment_features import aggregate_sentiment_features, calculate_article_metrics
from stonks.models.finbert import FinBERTModel
import pandas as pd
import numpy as np

def test_calculate_article_metrics():
    """Verify metrics calculation for positive, neutral, negative and dominant class identification."""
    art = {
        "headline": "Bullish news",
        "summary": "Everything is great",
        "sentiment": {"positive": 0.8, "neutral": 0.15, "negative": 0.05}
    }
    metrics = calculate_article_metrics(art)
    assert metrics["score"] == pytest.approx(0.75)
    assert metrics["confidence"] == 0.8
    assert metrics["label"] == "positive"
    
    art_neg = {
        "headline": "Bearish news",
        "summary": "Everything is bad",
        "sentiment": {"positive": 0.1, "neutral": 0.1, "negative": 0.8}
    }
    metrics_neg = calculate_article_metrics(art_neg)
    assert metrics_neg["score"] == pytest.approx(-0.7)
    assert metrics_neg["confidence"] == 0.8
    assert metrics_neg["label"] == "negative"

def test_aggregate_sentiment_features():
    """Verify aggregated statistics, ratios, average sentiment, and recency decay weighting."""
    ref_time = datetime(2026, 6, 1, 12, 0, 0)
    
    articles = [
        {
            "headline": "First news",
            "published_at": (ref_time - timedelta(hours=2)).isoformat(),
            "sentiment": {"positive": 0.8, "neutral": 0.1, "negative": 0.1}
        },
        {
            "headline": "Second news",
            "published_at": (ref_time - timedelta(days=2)).isoformat(),
            "sentiment": {"positive": 0.2, "neutral": 0.2, "negative": 0.6}
        }
    ]
    
    # Lambda = 0.5 decay
    # Article 1: age = 2 hours = 1/12 day. Weight = exp(-0.5 * 1/12) = 0.959
    # Article 2: age = 2 days. Weight = exp(-0.5 * 2) = 0.368
    
    features = aggregate_sentiment_features(articles, lambda_decay=0.5, reference_time=ref_time)
    
    assert features["article_count"] == 2.0
    assert features["positive_news_ratio"] == 0.5
    assert features["negative_news_ratio"] == 0.5
    assert features["neutral_news_ratio"] == 0.0
    
    # Score 1: 0.8 - 0.1 = 0.7
    # Score 2: 0.2 - 0.6 = -0.4
    # Average: (0.7 - 0.4) / 2 = 0.15
    assert features["sentiment_score"] == pytest.approx(0.15)
    assert features["average_sentiment"] == pytest.approx(0.15)
    
    # Recency weighted should be closer to Article 1 (0.7) than Article 2 (-0.4)
    assert features["recency_weighted_sentiment"] > 0.15

def test_finbert_model_wrapper():
    """Verify FinBERTModel wrapper behavior and formula mapping sentiment to [0,1] probability."""
    model = FinBERTModel()
    assert model.is_trained
    
    # If no features, return neutral 0.50
    X_empty = pd.DataFrame({"dummy": [1.0, 2.0]})
    probs_empty = model.predict_proba(X_empty)
    np.testing.assert_array_almost_equal(probs_empty, np.array([0.5, 0.5]))
    
    # If features exist, test formula mapping
    X_valid = pd.DataFrame({"sentiment_score": [1.0, -1.0, 0.0, 0.5]})
    # Prob = (score + 1) / 2
    # 1.0 -> 1.0
    # -1.0 -> 0.0
    # 0.0 -> 0.5
    # 0.5 -> 0.75
    expected_probs = np.array([1.0, 0.0, 0.5, 0.75])
    probs_valid = model.predict_proba(X_valid)
    np.testing.assert_array_almost_equal(probs_valid, expected_probs)
    
    # Test predict binary classification boundary
    predictions = model.predict(X_valid)
    np.testing.assert_array_equal(predictions, np.array([1, 0, 0, 1]))  # score 0.0 is prob 0.5 which is not > 0.5, so 0.
