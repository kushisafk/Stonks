import pytest
import pandas as pd
import numpy as np
from src.models.base_model import BaseModel
from src.ensemble.weighted_voting import WeightedEnsemble

# Create a simple mock model class for testing the ensemble math
class DummyModel(BaseModel):
    def __init__(self, constant_probs: np.ndarray):
        self.constant_probs = constant_probs
        
    def train(self, X, y): pass
    def predict(self, X): return (self.constant_probs >= 0.5).astype(int)
    def predict_proba(self, X): return self.constant_probs
    def save(self, path): pass
    def load(self, path): pass

def test_ensemble_registration_and_weights():
    """Verify that models can be added, removed, and weights normalize dynamically."""
    model_a = DummyModel(np.array([0.5]))
    model_b = DummyModel(np.array([0.5]))
    
    ensemble = WeightedEnsemble(initial_weights={"a": 1.0, "b": 1.0})
    
    # Assert errors when predicting on empty ensemble
    with pytest.raises(ValueError):
        ensemble.predict_proba(pd.DataFrame({"f": [1]}))
        
    # Register model A
    ensemble.register_model("a", model_a)
    norm_w1 = ensemble.get_normalized_weights()
    assert norm_w1["a"] == 1.0
    
    # Register model B
    ensemble.register_model("b", model_b)
    norm_w2 = ensemble.get_normalized_weights()
    assert norm_w2["a"] == 0.5
    assert norm_w2["b"] == 0.5
    
    # Set custom weights
    ensemble.set_weight("a", 3.0)
    ensemble.set_weight("b", 1.0)
    norm_w3 = ensemble.get_normalized_weights()
    assert norm_w3["a"] == 0.75  # 3.0 / 4.0
    assert norm_w3["b"] == 0.25  # 1.0 / 4.0
    
    # Remove model A
    ensemble.remove_model("a")
    norm_w4 = ensemble.get_normalized_weights()
    assert "a" not in norm_w4
    assert norm_w4["b"] == 1.0

def test_ensemble_math():
    """Verify the exact mathematical weighted average outputs from multiple registered models."""
    # Model A returns probabilities: [0.60, 0.80]
    # Model B returns probabilities: [0.40, 0.20]
    model_a = DummyModel(np.array([0.60, 0.80]))
    model_b = DummyModel(np.array([0.40, 0.20]))
    
    X = pd.DataFrame({"feat": [1, 2]})
    
    ensemble = WeightedEnsemble()
    ensemble.register_model("model_a", model_a)
    ensemble.register_model("model_b", model_b)
    
    # Test equal weighting (0.5 each)
    ensemble.set_weight("model_a", 1.0)
    ensemble.set_weight("model_b", 1.0)
    
    probs = ensemble.predict_proba(X)
    # Expected: [(0.6 + 0.4)/2, (0.8 + 0.2)/2] = [0.50, 0.50]
    assert len(probs) == 2
    assert probs[0] == pytest.approx(0.50)
    assert probs[1] == pytest.approx(0.50)
    
    # Test custom unequal weighting (Model A: 0.75 weight, Model B: 0.25 weight)
    ensemble.set_weight("model_a", 0.75)
    ensemble.set_weight("model_b", 0.25)
    
    probs_custom = ensemble.predict_proba(X)
    # Expected: 
    # Row 1: 0.75 * 0.60 + 0.25 * 0.40 = 0.45 + 0.10 = 0.55
    # Row 2: 0.75 * 0.80 + 0.25 * 0.20 = 0.60 + 0.05 = 0.65
    assert probs_custom[0] == pytest.approx(0.55)
    assert probs_custom[1] == pytest.approx(0.65)
    
    # Test binary classification prediction at default 0.50 threshold
    # Row 1 (0.55) -> 1, Row 2 (0.65) -> 1
    preds = ensemble.predict(X, threshold=0.50)
    assert (preds == 1).all()
    
    # Test binary classification at 0.60 threshold
    # Row 1 (0.55) -> 0, Row 2 (0.65) -> 1
    preds_strict = ensemble.predict(X, threshold=0.60)
    assert preds_strict[0] == 0
    assert preds_strict[1] == 1
