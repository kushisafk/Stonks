import sys
from pathlib import Path

# Force append project source path to sys.path
project_path = Path(__file__).resolve().parent.parent
sys.path.append(str(project_path))

from src.models.random_forest import RandomForestModel

# Resolve AAPL model file path
model_path = project_path / "models_data" / "AAPL_rf.joblib"
if not model_path.exists():
    print(f"ERROR: Model file {model_path} does not exist. Please train AAPL first.")
    sys.exit(1)

try:
    # Load serialized model state
    model = RandomForestModel()
    model.load(model_path)

    # Extract feature importances dictionary
    importances = model.feature_importances

    # Sort descending
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    # Output formatted Markdown table
    print("\n" + "="*80)
    print("MARKDOWN TABLE OUTPUT:")
    print("="*80)
    print("| Rank | Feature | Importance Score | Percentage | Category |")
    print("| :--- | :--- | :---: | :---: | :--- |")
    
    # Categorize features
    def get_category(name):
        market_context_cols = [
            "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_rsi", "spy_macd",
            "spy_volatility_20d", "spy_trend_strength"
        ]
        relative_strength_cols = [
            "relative_strength_5d", "relative_strength_20d", "relative_strength_50d",
            "relative_momentum_score"
        ]
        volume_cols = [
            "volume_sma_20", "volume_ratio", "volume_momentum", "volume_trend",
            "abnormal_volume_flag"
        ]
        sentiment_cols = [
            "sentiment_score", "positive_news_ratio", "negative_news_ratio",
            "neutral_news_ratio", "article_count", "average_sentiment",
            "weighted_sentiment", "recency_weighted_sentiment"
        ]
        if name in market_context_cols:
            return "Market Context (SPY)"
        elif name in relative_strength_cols:
            return "Relative Strength"
        elif name in volume_cols:
            return "Volume Intelligence"
        elif name in sentiment_cols:
            return "Sentiment (FinBERT)"
        elif name == "market_regime":
            return "Market Regime Classifier"
        else:
            return "Technical (Legacy)"

    for idx, (name, score) in enumerate(sorted_importances, 1):
        pct_str = f"{score:.2%}"
        cat = get_category(name)
        print(f"| {idx} | **{name}** | {score:.4f} | {pct_str} | {cat} |")
    print("="*80)
except Exception as e:
    print(f"ERROR: Failed to extract importances: {e}")
    sys.exit(1)
