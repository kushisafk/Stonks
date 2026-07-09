"""
STONKS Decision Engine Demo

Demonstrates how the Trading Intelligence Manager combines model predictions, 
calibrated probabilities, and dynamic news sentiment indicators to route trade decisions.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stonks.intelligence.recommendation_engine import RecommendationEngine

def main():
    print("\n" + "=" * 60)
    print("STONKS Decision Intelligence Engine Consensus Logic")
    print("=" * 60)
    
    engine = RecommendationEngine()
    
    # Scenario A: High Calibrated Prediction, Neutral Sentiment
    prob_a = 0.74
    sent_a = 0.0  # Neutral
    features_a = {
        "sentiment_score": sent_a,
        "market_regime": 1.0,
        "volume_ratio": 1.5,
        "abnormal_volume_flag": 1.0,
        "relative_strength_20d": 0.05
    }
    
    res_a = engine.generate_recommendation("AAPL", prob_a, "BUY", features_a)
    intel_a = res_a["json_report"]
    tier_a = intel_a["confidence_tier"]
    rec_a = intel_a["recommendation"]
    reason_a = intel_a["reasoning"]
    
    print("\nScenario A: High probability prediction, Neutral News Sentiment")
    print(f"  Calibrated Prob: {prob_a:.1%}")
    print(f"  Sentiment Index: {sent_a}")
    print(f"  Confidence Tier: {tier_a}")
    print(f"  Recommendation : {rec_a}")
    print(f"  Explainer      : {reason_a}")
    
    # Scenario B: Marginally Bullish Prediction, Highly Bearish Sentiment (Inhibits Buy!)
    prob_b = 0.62  # Above default buy threshold if we use 60% alternative
    sent_b = -0.75  # Heavy negative sentiment
    features_b = {
        "sentiment_score": sent_b,
        "market_regime": -1.0,
        "volume_ratio": 0.5,
        "relative_strength_20d": -0.04
    }
    
    res_b = engine.generate_recommendation("AAPL", prob_b, "BUY", features_b)
    intel_b = res_b["json_report"]
    tier_b = intel_b["confidence_tier"]
    rec_b = intel_b["recommendation"]
    reason_b = intel_b["reasoning"]
    
    print("\nScenario B: Bullish prediction, but Highly Bearish News Sentiment")
    print(f"  Calibrated Prob: {prob_b:.1%}")
    print(f"  Sentiment Index: {sent_b}")
    print(f"  Confidence Tier: {tier_b}")
    print(f"  Recommendation : {rec_b}")
    print(f"  Explainer      : {reason_b}")
    
    # Scenario C: Indecisive Prediction, Bullish Sentiment
    prob_c = 0.52
    sent_c = 0.80  # positive news
    features_c = {
        "sentiment_score": sent_c,
        "market_regime": 1.0,
        "volume_ratio": 1.0,
        "relative_strength_20d": 0.01
    }
    
    res_c = engine.generate_recommendation("AAPL", prob_c, "HOLD", features_c)
    intel_c = res_c["json_report"]
    tier_c = intel_c["confidence_tier"]
    rec_c = intel_c["recommendation"]
    reason_c = intel_c["reasoning"]
    
    print("\nScenario C: Neutral prediction, but Highly Bullish News Sentiment")
    print(f"  Calibrated Prob: {prob_c:.1%}")
    print(f"  Sentiment Index: {sent_c}")
    print(f"  Confidence Tier: {tier_c}")
    print(f"  Recommendation : {rec_c}")
    print(f"  Explainer      : {reason_c}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
