import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
from src.logging.logger import logger

class LogisticRegressionModel(BaseModel):
    """Concrete implementation of BaseModel using scikit-learn LogisticRegression."""
    
    def __init__(
        self, 
        max_iter: int = 1000,
        C: float = 1.0,
        random_state: int = 42
    ):
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self._feature_names = []
        
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Trains the Logistic Regression classifier on features X and labels y."""
        logger.info(f"Training LogisticRegressionModel: Samples={len(X)}, Features={X.shape[1]}")
        self._feature_names = list(X.columns)
        
        # Scale features internally to ensure stable convergence
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        logger.info("LogisticRegressionModel training completed successfully.")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts binary price direction target."""
        if not self.is_trained:
            raise ValueError(
                "LogisticRegressionModel is not trained yet. Call train() or load() before predicting."
            )
        if list(X.columns) != self._feature_names:
            logger.warning("Feature column sequence mismatch! Realigning X to training layout.")
            X = X[self._feature_names]
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts prediction probability of the price increasing (class 1)."""
        if not self.is_trained:
            raise ValueError(
                "LogisticRegressionModel is not trained yet. Call train() or load() before predicting."
            )
        if list(X.columns) != self._feature_names:
            X = X[self._feature_names]
            
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        
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
            "scaler": self.scaler,
            "is_trained": self.is_trained,
            "_feature_names": self._feature_names
        }
        joblib.dump(state, path)
        logger.info(f"LogisticRegressionModel serialized successfully to {path}")
        
    def load(self, path: Path) -> None:
        """Loads the serialized model state from disk using joblib."""
        if not path.exists():
            raise FileNotFoundError(f"No serialized model file found at path: {path}")
        state = joblib.load(path)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.is_trained = state["is_trained"]
        self._feature_names = state.get("_feature_names", [])
        logger.info(f"LogisticRegressionModel deserialized successfully from {path}")
