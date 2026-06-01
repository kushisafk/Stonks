import pytest
import pandas as pd
import numpy as np
from src.models.base_model import BaseModel
from src.models.lstm import LSTMModel
from src.models.transformer import TransformerModel
from src.models.finbert import FinBERTModel
from src.models.model_registry import get_model_class, register_model_class, list_registered_models

def test_stubs_behavior():
    """Verify that deep learning stubs produce baseline neutral prediction outputs."""
    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0]})
    
    for model_class in [LSTMModel, TransformerModel, FinBERTModel]:
        model = model_class()
        
        # Test train does not crash
        model.train(X, pd.Series([1, 0, 1]))
        
        # Predict should return binary 0s
        preds = model.predict(X)
        assert len(preds) == 3
        assert (preds == 0).all()
        
        # Predict proba should return 0.50 probabilities
        probs = model.predict_proba(X)
        assert len(probs) == 3
        assert (probs == 0.50).all()

def test_model_registry_resolution():
    """Verify that get_model_class dynamically loads stubs and handles RF error state."""
    # Resolve LSTM stub
    lstm_cls = get_model_class("lstm")
    assert lstm_cls == LSTMModel
    
    # Resolve Transformer stub (case-insensitive check)
    trans_cls = get_model_class("TRANSFORMER")
    assert trans_cls == TransformerModel
    
    # Resolve FinBERT stub
    finbert_cls = get_model_class("finbert")
    assert finbert_cls == FinBERTModel
    
    # Check that 'rf' resolves successfully to RandomForestModel since it is implemented
    from src.models.random_forest import RandomForestModel
    rf_cls = get_model_class("rf")
    assert rf_cls == RandomForestModel

def test_custom_model_registration():
    """Verify that new classes can be registered and listed in the central registry."""
    # Define dummy test model
    class CustomMockModel(BaseModel):
        def train(self, X, y): pass
        def predict(self, X): return np.ones(len(X))
        def predict_proba(self, X): return np.ones(len(X))
        def save(self, path): pass
        def load(self, path): pass
        
    # Register model class
    register_model_class("custom_mock", CustomMockModel)
    
    # Retrieve and assert
    resolved_cls = get_model_class("custom_mock")
    assert resolved_cls == CustomMockModel
    
    # Verify it is listed
    models_dict = list_registered_models()
    assert "custom_mock" in models_dict
    assert models_dict["custom_mock"] == CustomMockModel

def test_random_forest_model(tmp_path):
    """Verify RandomForestModel training, prediction, serialization, and importances."""
    from src.models.random_forest import RandomForestModel
    
    X = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100)
    })
    # Target is 1 if feat1 is positive, else 0
    y = pd.Series((X["feat1"] > 0.0).astype(int))
    
    rf = RandomForestModel(n_estimators=10, max_depth=3)
    assert not rf.is_trained
    
    # Train
    rf.train(X, y)
    assert rf.is_trained
    
    # Predict
    preds = rf.predict(X)
    probs = rf.predict_proba(X)
    
    assert len(preds) == 100
    assert len(probs) == 100
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    
    # Feature importances
    importances = rf.feature_importances
    assert "feat1" in importances
    assert "feat2" in importances
    # Sum of importances should be roughly 1.0
    assert sum(importances.values()) == pytest.approx(1.0)
    
    # Save & Load
    model_file = tmp_path / "rf.joblib"
    rf.save(model_file)
    assert model_file.exists()
    
    rf_loaded = RandomForestModel()
    rf_loaded.load(model_file)
    assert rf_loaded.is_trained
    assert rf_loaded._feature_names == ["feat1", "feat2"]
    
    # Verify predictions are identical after reload
    loaded_preds = rf_loaded.predict(X)
    assert (loaded_preds == preds).all()
    
    # Test registry resolves RF correctly now that it's implemented
    rf_class_resolved = get_model_class("rf")
    assert rf_class_resolved == RandomForestModel
