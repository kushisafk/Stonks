import json
import pytest
from pathlib import Path
from stonks.config.settings import settings
from stonks.models.model_registry import get_model_class, get_best_model
from stonks.models.xgboost import XGBoostModel
from stonks.models.catboost import CatBoostModel

def test_registry_get_best_model_fallback():
    """Verify that get_best_model falls back to configured default model if leaderboard doesn't exist."""
    # Temporarily rename leaderboard if it exists
    leaderboard_path = settings.MODEL_DIR / "leaderboard.json"
    temp_path = settings.MODEL_DIR / "leaderboard_temp_backup.json"
    
    if leaderboard_path.exists():
        leaderboard_path.rename(temp_path)
        
    try:
        # Default config MODEL is random_forest
        settings.MODEL = "random_forest"
        best_model_cls = get_best_model()
        from stonks.models.random_forest import RandomForestModel
        assert best_model_cls == RandomForestModel
        
        # Test switching to xgboost
        settings.MODEL = "xgboost"
        best_model_cls = get_best_model()
        assert best_model_cls == XGBoostModel
    finally:
        # Restore leaderboard
        if temp_path.exists():
            if leaderboard_path.exists():
                leaderboard_path.unlink()
            temp_path.rename(leaderboard_path)

def test_registry_get_best_model_from_leaderboard():
    """Verify that get_best_model reads rank 1 from leaderboard.json."""
    leaderboard_path = settings.MODEL_DIR / "leaderboard.json"
    temp_path = settings.MODEL_DIR / "leaderboard_temp_backup.json"
    
    if leaderboard_path.exists():
        leaderboard_path.rename(temp_path)
        
    try:
        # Create a mock leaderboard with xgboost at rank 1
        mock_data = [
            {"model": "xgboost", "overall_score": 0.88},
            {"model": "catboost", "overall_score": 0.85}
        ]
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(leaderboard_path, "w") as f:
            json.dump(mock_data, f)
            
        settings.MODEL = "catboost"  # config is catboost, but leaderboard has xgboost rank 1
        best_model_cls = get_best_model()
        assert best_model_cls == XGBoostModel
    finally:
        # Cleanup mock and restore
        if leaderboard_path.exists():
            leaderboard_path.unlink()
        if temp_path.exists():
            temp_path.rename(leaderboard_path)

def test_dynamic_ensemble_weights():
    """Verify settings.ensemble_weights resolves weights dynamically based on settings.MODEL."""
    settings.MODEL = "xgboost"
    settings.RF_WEIGHT = 0.75
    settings.FINBERT_WEIGHT = 0.25
    settings.LSTM_WEIGHT = 0.0
    settings.TRANSFORMER_WEIGHT = 0.0
    
    weights = settings.ensemble_weights
    assert weights["xgboost"] == 0.75
    assert weights["finbert"] == 0.25
    assert "rf" not in weights
    
    # Switch back to random_forest (or rf alias)
    settings.MODEL = "random_forest"
    weights = settings.ensemble_weights
    assert weights["rf"] == 0.75
    assert weights["finbert"] == 0.25
    assert "xgboost" not in weights
