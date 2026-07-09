import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from catboost import CatBoostClassifier
from stonks.models.base_model import BaseModel
from stonks.logging.logger import logger

class CatBoostModel(BaseModel):
    """Concrete implementation of BaseModel using catboost CatBoostClassifier."""
    
    def __init__(
        self, 
        iterations: int = 100,
        depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            logging_level="Silent",
            thread_count=-1
        )
        self.is_trained = False
        self._feature_names = []
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the CatBoost classifier on features X and labels y."""
        logger.info(f"Training CatBoostModel: Samples={len(X)}, Features={X.shape[1]}")
        self._feature_names = list(X.columns)
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("CatBoostModel training completed successfully.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary price direction target."""
        if not self.is_trained:
            raise ValueError(
                "CatBoostModel is not trained yet. Call train() or load() before predicting."
            )
        if list(X.columns) != self._feature_names:
            logger.warning("Feature column sequence mismatch! Realigning X to training layout.")
            X = X[self._feature_names]
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts prediction probability of the price increasing (class 1)."""
        if not self.is_trained:
            raise ValueError(
                "CatBoostModel is not trained yet. Call train() or load() before predicting."
            )
        if list(X.columns) != self._feature_names:
            X = X[self._feature_names]
            
        probs = self.model.predict_proba(X)
        
        if 1 not in self.model.classes_:
            logger.warning("Target class 1 was not observed during training. Returning 0.0 probabilities.")
            return np.zeros(len(X), dtype=float)
            
        class_1_idx = list(self.model.classes_).index(1)
        return probs[:, class_1_idx]
        
    def save(self, path: Path) -> None:
        """Serializes and saves the trained model state to disk using joblib."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model": self.model,
            "is_trained": self.is_trained,
            "_feature_names": self._feature_names
        }
        joblib.dump(state, path)
        logger.info(f"CatBoostModel serialized successfully to {path}")
        
    def load(self, path: Path) -> None:
        """Loads the serialized model state from disk using joblib."""
        if not path.exists():
            raise FileNotFoundError(f"No serialized model file found at path: {path}")
        state = joblib.load(path)
        self.model = state["model"]
        self.is_trained = state["is_trained"]
        self._feature_names = state.get("_feature_names", [])
        logger.info(f"CatBoostModel deserialized successfully from {path}")
        
    @property
    def feature_importances(self) -> Dict[str, float]:
        """
        Extracts and maps relative feature importances to their respective feature names.
        
        Returns:
            Dict[str, float]: Mappings of feature names to their importance values (0.0 to 1.0)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet. Feature importances are unavailable.")
        importances = self.model.feature_importances_
        # Normalize sum of importances to match RF / ExtraTrees behavior
        total = sum(importances)
        if total > 0:
            importances = [imp / total for imp in importances]
        return {name: float(imp) for name, imp in zip(self._feature_names, importances)}
