import math
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from src.logging.logger import logger

def calculate_article_metrics(article: Dict[str, Any]) -> Dict[str, float]:
    """
    Extracts class probabilities and computes individual article sentiment score and confidence.
    """
    sentiment = article.get("sentiment", {"positive": 0.0, "neutral": 1.0, "negative": 0.0})
    pos = float(sentiment.get("positive", 0.0))
    neu = float(sentiment.get("neutral", 0.0))
    neg = float(sentiment.get("negative", 0.0))
    
    score = pos - neg
    confidence = max(pos, neu, neg)
    
    # Identify dominant label
    if pos >= neu and pos >= neg:
        label = "positive"
    elif neg >= pos and neg >= neu:
        label = "negative"
    else:
        label = "neutral"
        
    return {
        "score": score,
        "confidence": confidence,
        "positive": pos,
        "neutral": neu,
        "negative": neg,
        "label": label
    }

def aggregate_sentiment_features(
    articles: List[Dict[str, Any]], 
    lambda_decay: float = 0.5,
    reference_time: Optional[datetime] = None
) -> Dict[str, float]:
    """
    Aggregates individual article sentiments into a standardized feature dictionary.
    
    Args:
        articles: List of articles containing 'sentiment' dictionary
        lambda_decay: Decay parameter for recency weighting (default 0.5 per day)
        reference_time: The current evaluation datetime (defaults to datetime.now())
        
    Returns:
        Dict[str, float]: Aggregated sentiment features
    """
    if not articles:
        return {
            "sentiment_score": 0.0,
            "positive_news_ratio": 0.0,
            "negative_news_ratio": 0.0,
            "neutral_news_ratio": 0.0,
            "article_count": 0.0,
            "average_sentiment": 0.0,
            "weighted_sentiment": 0.0,
            "recency_weighted_sentiment": 0.0
        }
        
    ref_time = reference_time or datetime.now()
    
    scores = []
    confidences = []
    recency_weights = []
    
    pos_count = 0
    neg_count = 0
    neu_count = 0
    
    for art in articles:
        metrics = calculate_article_metrics(art)
        scores.append(metrics["score"])
        confidences.append(metrics["confidence"])
        
        # Calculate label counts
        label = metrics["label"]
        if label == "positive":
            pos_count += 1
        elif label == "negative":
            neg_count += 1
        else:
            neu_count += 1
            
        # Time decay weight calculation
        try:
            pub_time_str = art.get("published_at")
            if pub_time_str:
                pub_time = pd.to_datetime(pub_time_str)
                # Convert both to timezone-naive if one is naive, or match timezones
                if pub_time.tzinfo is not None:
                    pub_time = pub_time.tz_localize(None)
                ref_naive = ref_time.replace(tzinfo=None)
                
                age_days = (ref_naive - pub_time).total_seconds() / 86400.0
                age_days = max(0.0, age_days)  # Clip negative age (future articles)
            else:
                age_days = 0.0
        except Exception as e:
            logger.warning(f"Error parsing published_at '{art.get('published_at')}': {e}")
            age_days = 0.0
            
        weight = math.exp(-lambda_decay * age_days)
        recency_weights.append(weight)
        
    n = len(articles)
    avg_score = sum(scores) / n
    
    # Weighted sentiment by confidence
    sum_conf = sum(confidences)
    weighted_score = sum(s * c for s, c in zip(scores, confidences)) / sum_conf if sum_conf > 0 else avg_score
    
    # Recency weighted sentiment
    sum_weight = sum(recency_weights)
    recency_weighted_score = sum(s * w for s, w in zip(scores, recency_weights)) / sum_weight if sum_weight > 0 else avg_score
    
    return {
        "sentiment_score": avg_score,
        "positive_news_ratio": pos_count / n,
        "negative_news_ratio": neg_count / n,
        "neutral_news_ratio": neu_count / n,
        "article_count": float(n),
        "average_sentiment": avg_score,
        "weighted_sentiment": weighted_score,
        "recency_weighted_sentiment": recency_weighted_score
    }
