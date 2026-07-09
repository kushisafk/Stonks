import numpy as np
import pandas as pd
from pathlib import Path
from stonks.models.base_model import BaseModel
from stonks.logging.logger import logger

class TransformerModel(BaseModel):
    """Stub implementation of PyTorch Transformer Model for future quantitative sequence learning."""
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.warning("TransformerModel.train: Attention network is not active in Phase 1. Ignoring training.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        logger.warning("TransformerModel.predict: Attention network is not active. Stub returns neutral 0s.")
        return np.zeros(len(X), dtype=int)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logger.warning("TransformerModel.predict_proba: Attention network is not active. Stub returns neutral 0.50.")
        return np.full(len(X), 0.50, dtype=float)
        
    def save(self, path: Path) -> None:
        logger.warning(f"TransformerModel.save: Bypassing model serialization to {path}.")
        
    def load(self, path: Path) -> None:
        logger.warning(f"TransformerModel.load: Bypassing model loading from {path}.")
