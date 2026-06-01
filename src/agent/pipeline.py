import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
from src.config.settings import settings
from src.logging.logger import logger, decision_logger
from src.data.market_data import market_data_service
from src.features.feature_pipeline import feature_pipeline
from src.models.random_forest import RandomForestModel
from src.ensemble.weighted_voting import WeightedEnsemble
from src.decision.decision_engine import decision_engine
from src.ai_layer.explainer import RuleBasedExplainer

class TradingAgent:
    """Coordinates the end-to-end trading process: data fetch, features, training/inference, signal generation, and logging."""
    
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or settings.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.explainer = RuleBasedExplainer()
        
    def _get_model_path(self, symbol: str) -> Path:
        """Constructs a persistent model weight file path."""
        return self.model_dir / f"{symbol.strip().upper()}_rf.joblib"
        
    def run_pipeline(self, symbol: str, force_train: bool = False) -> Dict[str, Any]:
        """
        Runs the end-to-end stock prediction pipeline.
        
        Args:
            symbol: Ticker symbol (e.g., AAPL)
            force_train: If True, retrains the Random Forest model on historical data.
            
        Returns:
            Dict[str, Any]: Consolidated prediction payload containing signal, confidence, explanation, etc.
        """
        symbol = symbol.strip().upper()
        logger.info(f"TradingAgent: Running pipeline for ticker {symbol}...")
        
        # 1. Fetch market data (uses cache if fresh)
        raw_df = market_data_service.fetch_data(symbol)
        
        # 2. Extract features for today's prediction (keeps last row!)
        X_pred, _ = feature_pipeline.get_features(symbol, raw_df, is_training=False, use_store=True)
        
        if X_pred.empty:
            raise ValueError(f"TradingAgent: Not enough historical data to calculate indicators for {symbol}.")
            
        # Get today's Close price and today's feature vector
        today_close = float(raw_df.iloc[-1]["Close"])
        today_features = X_pred.iloc[-1].to_dict()
        X_today = X_pred.iloc[[-1]]  # Keep as dataframe of 1 row for models
        
        model_path = self._get_model_path(symbol)
        rf_model = RandomForestModel()
        
        # 3. Model Training or Loading
        if force_train or not model_path.exists():
            logger.info(f"Model file not found or force_train=True. Initiating training for {symbol}...")
            # Fetch training dataset (nan targets and last row trimmed)
            X_train, y_train = feature_pipeline.get_features(symbol, raw_df, is_training=True, use_store=False)
            
            if len(X_train) < 50:
                raise ValueError(
                    f"TradingAgent: Insufficient training samples ({len(X_train)}) after NaNs to train model for {symbol}."
                )
                
            rf_model.train(X_train, y_train)
            rf_model.save(model_path)
        else:
            logger.info(f"Loading trained RandomForest model for {symbol} from {model_path}...")
            rf_model.load(model_path)
            
        # 4. Assemble Weighted Ensemble
        ensemble = WeightedEnsemble()
        ensemble.register_model("rf", rf_model)
        
        # Load and register pre-trained FinBERTModel wrapper
        from src.models.finbert import FinBERTModel
        finbert_model = FinBERTModel()
        ensemble.register_model("finbert", finbert_model)
        
        # Set weights dynamically from settings configurations
        ensemble.set_weight("rf", settings.RF_WEIGHT)
        ensemble.set_weight("finbert", settings.FINBERT_WEIGHT)
        
        # 5. Predict class probability via ensemble
        pred_prob = float(ensemble.predict_proba(X_today)[0])
        individual_probs = {
            name: float(probs[0]) 
            for name, probs in ensemble.get_individual_probabilities(X_today).items()
        }
        
        # 6. Translate probability to signal inside Decision Engine
        decision = decision_engine.make_decision(pred_prob)
        signal = decision["signal"]
        confidence = decision["confidence"]
        
        # 7. Generate natural language explanation using Explainer
        explanation = self.explainer.explain(signal, confidence, today_features)
        
        # 8. Persistent structured logging to Decisions CSV
        decision_logger.log_decision(
            ticker=symbol,
            signal=signal,
            confidence=confidence,
            close_price=today_close,
            probabilities=individual_probs
        )
        
        logger.info(f"TradingAgent: Pipeline finished for {symbol} -> Signal: {signal} ({confidence:.2f})")
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "close_price": today_close,
            "explanation": explanation,
            "probabilities": individual_probs,
            "timestamp": datetime.now().isoformat()
        }

# Global trading agent instance
trading_agent = TradingAgent()
