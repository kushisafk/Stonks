import numpy as np
import pandas as pd
from pathlib import Path
from stonks.models.base_model import BaseModel
from stonks.logging.logger import logger

class FinBERTModel(BaseModel):
    """
    BaseModel-compliant wrapper for the FinBERT Sentiment Model.
    In the predictive pipeline, it translates calculated sentiment features into directional probability signals.
    """
    
    def __init__(self):
        self.is_trained = True  # FinBERT is pre-trained on financial corpus, ready on init
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """FinBERT is pre-trained on financial text. Bypasses local training while maintaining BaseModel compat."""
        logger.info("FinBERTModel.train: Using pre-trained ProsusAI/finbert weights. Bypassing local fitting.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary price direction based on sentiment score boundary."""
        probs = self.predict_proba(X)
        return (probs > 0.50).astype(int)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Translates computed sentiment_score [-1.0, +1.0] into a price increase probability [0.0, 1.0].
        Formula: Prob = (Sentiment + 1) / 2
        """
        if "sentiment_score" not in X.columns:
            logger.warning(
                "FinBERTModel.predict_proba: 'sentiment_score' column not found in features. "
                "Returning neutral 0.50 probabilities."
            )
            return np.full(len(X), 0.50, dtype=float)
            
        scores = X["sentiment_score"].values
        # Map [-1.0, 1.0] to [0.0, 1.0]
        probs = (scores + 1.0) / 2.0
        # Safeguard probability boundaries
        probs = np.clip(probs, 0.0, 1.0)
        return probs
        
    def save(self, path: Path) -> None:
        """Saves model state metadata (pre-trained, so minimal state needed)."""
        logger.info(f"FinBERTModel.save: Saving model metadata state stub to {path}")
        
    def load(self, path: Path) -> None:
        """Loads model state metadata."""
        logger.info(f"FinBERTModel.load: Loading model metadata state stub from {path}")
