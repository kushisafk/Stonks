from pathlib import Path
from typing import List, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # Project Root Directory
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    
    # App Settings
    APP_ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")
    MODEL: str = Field(default="random_forest")
    
    # Storage Directories (resolved relative to root if relative)
    LOG_DIR: Path = Field(default=Path("logs"))
    MODEL_DIR: Path = Field(default=Path("models_data"))
    CACHE_DIR: Path = Field(default=Path("cache"))
    FEATURE_STORE_DIR: Path = Field(default=Path("feature_store"))
    NEWS_CACHE_DIR: Path = Field(default=Path("cache/news"))
    SENTIMENT_CACHE_DIR: Path = Field(default=Path("cache/sentiment"))
    
    # Default parameters
    DEFAULT_TICKERS: Union[str, List[str]] = Field(default=["AAPL", "MSFT", "GOOGL", "TSLA"])
    YFINANCE_PERIOD: str = Field(default="2y")
    YFINANCE_INTERVAL: str = Field(default="1d")
    
    # News & Sentiment parameters
    NEWS_PROVIDER: str = Field(default="yfinance")
    FINNHUB_API_KEY: str = Field(default="")
    NEWSAPI_API_KEY: str = Field(default="")
    NEWS_LOOKBACK_DAYS: int = Field(default=7)
    NEWS_MAX_ARTICLES: int = Field(default=15)
    
    # Decision thresholds
    BUY_THRESHOLD: float = Field(default=0.65)
    SELL_THRESHOLD: float = Field(default=0.40)
    
    # Model Weights (Including FinBERT weight in Phase 2)
    RF_WEIGHT: float = Field(default=0.70)
    FINBERT_WEIGHT: float = Field(default=0.30)
    LSTM_WEIGHT: float = Field(default=0.0)
    TRANSFORMER_WEIGHT: float = Field(default=0.0)
    
    # Backtesting costs
    COMMISSION: float = Field(default=0.0010)
    SLIPPAGE: float = Field(default=0.0005)
    
    # Scheduler settings
    SCHEDULER_INTERVAL_HOURS: int = Field(default=24)
    
    def __init__(self, **values):
        # Pre-process comma-separated tickers if loaded from environmental string
        if "DEFAULT_TICKERS" in values and isinstance(values["DEFAULT_TICKERS"], str):
            values["DEFAULT_TICKERS"] = [
                ticker.strip().upper() 
                for ticker in values["DEFAULT_TICKERS"].split(",") 
                if ticker.strip()
            ]
        super().__init__(**values)
        
        # Ensure default tickers is always a list
        if isinstance(self.DEFAULT_TICKERS, str):
            self.DEFAULT_TICKERS = [
                ticker.strip().upper() 
                for ticker in self.DEFAULT_TICKERS.split(",") 
                if ticker.strip()
            ]
        
        # Ensure target storage directories exist and are absolute
        target_dirs = ["LOG_DIR", "MODEL_DIR", "CACHE_DIR", "FEATURE_STORE_DIR", "NEWS_CACHE_DIR", "SENTIMENT_CACHE_DIR"]
        for attr in target_dirs:
            path = getattr(self, attr)
            if not path.is_absolute():
                resolved_path = (self.PROJECT_ROOT / path).resolve()
                setattr(self, attr, resolved_path)
            
            # Autocreate directory
            getattr(self, attr).mkdir(parents=True, exist_ok=True)
            
    @property
    def ensemble_weights(self) -> dict:
        """Returns normalized model weights for the ensemble, mapping configured model dynamically."""
        active_model = self.MODEL.strip().lower()
        # Support "random_forest" alias as "rf" for weight dictionary compatibility
        primary_key = "rf" if active_model in ("random_forest", "rf") else active_model
        
        raw_weights = {
            primary_key: self.RF_WEIGHT,
            "lstm": self.LSTM_WEIGHT,
            "transformer": self.TRANSFORMER_WEIGHT,
            "finbert": self.FINBERT_WEIGHT
        }
        
        # If the active model itself is one of the standard stubs (e.g. lstm), handle gracefully
        if primary_key in raw_weights and primary_key != "rf":
            pass # Keep primary_key weight
            
        total = sum(raw_weights.values())
        if total == 0:
            # Fallback if all weights are zero to avoid division by zero
            return {k: 1.0 / len(raw_weights) for k in raw_weights}
        return {k: v / total for k, v in raw_weights.items()}

# Instantiate global settings object
settings = Settings()
