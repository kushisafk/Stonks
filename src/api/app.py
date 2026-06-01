from fastapi import FastAPI, HTTPException, Query
from src.config.settings import settings
from src.agent.pipeline import trading_agent
from src.data.market_data import market_data_service
from src.backtesting.backtester import Backtester
from src.models.model_registry import list_registered_models
from src.schemas.decision import DecisionResponse
from src.schemas.backtest import BacktestResponse
from src.schemas.models import ModelsResponse, ModelInfo
from src.schemas.sentiment import SentimentResponse

app = FastAPI(
    title="STONKS AI Stock Trading Decision Platform",
    description="Production-grade modular AI trading pipeline exposing predictive signals and backtest diagnostics.",
    version="1.0.0"
)

@app.get("/health", tags=["System"])
def health_check():
    """Returns application status and environment metadata."""
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
        "log_level": settings.LOG_LEVEL
    }

@app.get("/predict/{symbol}", response_model=DecisionResponse, tags=["Predictions"])
def get_prediction(
    symbol: str, 
    force_train: bool = Query(default=False, description="Force retraining of the Random Forest model on stock history")
):
    """
    Triggers end-to-end trading decision pipeline for a stock symbol.
    Fetches market data, computes indicators, loads/trains models, runs ensemble prediction, 
    evaluates decisions, logs results, and explains the signal.
    """
    try:
        payload = trading_agent.run_pipeline(symbol, force_train=force_train)
        return payload
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.get("/backtest/{symbol}", response_model=BacktestResponse, tags=["Research & Backtesting"])
def run_backtest(
    symbol: str,
    train_window: int = Query(default=250, ge=100, description="Size of sliding training window (trading days)"),
    test_window: int = Query(default=50, ge=10, description="Size of testing window (trading days)")
):
    """
    Executes a rolling out-of-sample walk-forward backtest for a stock symbol.
    Simulates trading orders incorporating commission and slippage frictions compared to Buy & Hold.
    """
    try:
        # 1. Ingest raw market data
        raw_df = market_data_service.fetch_data(symbol)
        
        # 2. Instantiate and run backtester
        backtester = Backtester()
        results = backtester.run_walk_forward(
            symbol=symbol,
            raw_df=raw_df,
            train_window=train_window,
            test_window=test_window
        )
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtesting execution failed: {str(e)}")

@app.get("/models", response_model=ModelsResponse, tags=["Diagnostics"])
def list_models_status():
    """
    Audits the central model registry and returning currently registered weights and operational readiness.
    """
    try:
        registered = list_registered_models()
        weights = settings.ensemble_weights
        
        active_models = {}
        total_weight = 0.0
        
        for name in ["rf", "lstm", "transformer", "finbert"]:
            cls = registered.get(name)
            weight = weights.get(name, 0.0)
            
            is_active = cls is not None
            status = "READY" if name == "rf" and is_active else ("STUB" if is_active else "DISABLED")
            
            active_models[name] = ModelInfo(
                weight=weight,
                is_active=is_active,
                status=status
            )
            total_weight += weight
            
        return ModelsResponse(
            active_models=active_models,
            total_active_weight=total_weight
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostics check failed: {str(e)}")

@app.get("/sentiment/{symbol}", response_model=SentimentResponse, tags=["Sentiment"])
def get_sentiment(symbol: str):
    """
    Fetches real-time stock news, analyzes individual article sentiments with FinBERT,
    and returns aggregated sentiment scores and class ratios.
    """
    try:
        from src.data.news_data import news_collector
        from src.sentiment.sentiment_analyzer import sentiment_analyzer
        from src.sentiment.sentiment_features import aggregate_sentiment_features
        
        symbol_upper = symbol.strip().upper()
        raw_articles = news_collector.get_news(symbol_upper, force_refresh=False)
        if not raw_articles:
            return {
                "symbol": symbol_upper,
                "sentiment_score": 0.0,
                "articles_analyzed": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0
            }
            
        analyzed_articles = sentiment_analyzer.analyze_batch(raw_articles)
        features = aggregate_sentiment_features(analyzed_articles)
        
        return {
            "symbol": symbol_upper,
            "sentiment_score": features["sentiment_score"],
            "articles_analyzed": int(features["article_count"]),
            "positive_ratio": features["positive_news_ratio"],
            "negative_ratio": features["negative_news_ratio"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")
