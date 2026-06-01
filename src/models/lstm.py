import numpy as np
import pandas as pd
from pathlib import Path
from src.models.base_model import BaseModel
from src.logging.logger import logger

class LSTMModel(BaseModel):
    """Stub implementation of PyTorch LSTM Model for future quantitative sequence learning."""
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.warning("LSTMModel.train: Sequence learning is not active in Phase 1. Ignoring training.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.warning("LSTMModel.predict: Sequence learning is not active. Stub returns neutral 0s.")
        return np.zeros(len(X), dtype=int)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logger.warning("LSTMModel.predict_proba: Sequence learning is not active. Stub returns neutral 0.50.")
        return np.full(len(X), 0.50, dtype=float)
        
    def save(self, path: Path) -> None:
        logger.warning(f"LSTMModel.save: Bypassing model serialization to {path}.")
        
    def load(self, path: Path) -> None:
        logger.warning(f"LSTMModel.load: Bypassing model loading from {path}.")
