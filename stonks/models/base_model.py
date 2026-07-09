from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path

class BaseModel(ABC):
    """Abstract Base Model defining standard interfaces for all predictive models in STONKS."""
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the model on features X and target y.
        
        Args:
            X: Pandas DataFrame of shape (n_samples, n_features)
            y: Pandas Series of target labels of shape (n_samples,)
        """
        pass
        
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary target labels (1 for price increase, 0 for price decrease).
        
        Args:
            X: Pandas DataFrame of shape (n_samples, n_features)
            
        Returns:
            np.ndarray of shape (n_samples,) containing binary target values.
        """
        pass
        
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts prediction probabilities of the price increasing (class 1).
        
        Args:
            X: Pandas DataFrame of shape (n_samples, n_features)
            
        Returns:
            np.ndarray of shape (n_samples,) containing floating probabilities in [0.0, 1.0].
        """
        pass
        
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Serializes and saves the trained model state to a file.
        
        Args:
            path: Path to the target serialization file.
        """
        pass
        
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Loads the serialized model state from a file.
        
        Args:
            path: Path to the serialized model file.
        """
        pass
