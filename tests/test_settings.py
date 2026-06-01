import os
from pathlib import Path
from src.config.settings import Settings, settings

def test_settings_load_defaults():
    """Verify that settings are instantiated with robust default values and folders exist."""
    assert settings.APP_ENV in ["dev", "test", "production"]
    assert settings.BUY_THRESHOLD == 0.65
    assert settings.SELL_THRESHOLD == 0.40
    assert "AAPL" in settings.DEFAULT_TICKERS
    
    # Check that directories were automatically created
    assert settings.LOG_DIR.exists()
    assert settings.MODEL_DIR.exists()
    assert settings.CACHE_DIR.exists()
    assert settings.FEATURE_STORE_DIR.exists()
    
    # Assert directories are absolute paths
    assert settings.LOG_DIR.is_absolute()

def test_settings_custom_parameters(tmp_path):
    """Verify that setting values can be overridden and lists are parsed correctly."""
    custom_settings = Settings(
        APP_ENV="testing",
        BUY_THRESHOLD=0.70,
        SELL_THRESHOLD=0.30,
        DEFAULT_TICKERS="MSFT,GOOGL,AMZN",
        LOG_DIR=tmp_path / "logs",
        MODEL_DIR=tmp_path / "models",
        CACHE_DIR=tmp_path / "cache",
        FEATURE_STORE_DIR=tmp_path / "features",
        RF_WEIGHT=0.8,
        LSTM_WEIGHT=0.2
    )
    
    assert custom_settings.APP_ENV == "testing"
    assert custom_settings.BUY_THRESHOLD == 0.70
    assert custom_settings.SELL_THRESHOLD == 0.30
    assert custom_settings.DEFAULT_TICKERS == ["MSFT", "GOOGL", "AMZN"]
    
    # Check dynamic ensemble weights calculation and normalization
    weights = custom_settings.ensemble_weights
    assert weights["rf"] == 0.8 / 1.0  # RF / (RF + LSTM)
    assert weights["lstm"] == 0.2 / 1.0
    assert weights["transformer"] == 0.0
    assert weights["finbert"] == 0.0
