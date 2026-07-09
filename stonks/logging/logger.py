import os
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from stonks.config.settings import settings

def setup_logging():
    """Configures application-wide logging to console and app.log file."""
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "app.log"
    
    # Configure root logging handlers
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8")
        ]
    )
    
    # Mute noisy third-party loggers on console StreamHandler
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    # Suppress HuggingFace hub unauthenticated requests warnings specifically
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Run logging setup automatically on module import
setup_logging()
logger = logging.getLogger("STONKS")

class DecisionLogger:
    """Manages appending trading signals and metadata to a persistent local CSV file."""
    
    def __init__(self, csv_path: Optional[Path] = None):
        self.csv_path = csv_path or (settings.LOG_DIR / "decisions.csv")
        self._initialize_csv()
        
    def _initialize_csv(self):
        """Creates the CSV file with column headers if it does not already exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            try:
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp", 
                        "Ticker", 
                        "Signal", 
                        "Confidence", 
                        "ClosePrice", 
                        "ModelProbabilities"
                    ])
                logger.info(f"Initialized CSV decision log at {self.csv_path}")
            except Exception as e:
                # Use standard print if logging is not ready, though logging is initialized above
                print(f"ERROR: Could not initialize decisions CSV at {self.csv_path}: {e}")
                
    def log_decision(
        self, 
        ticker: str, 
        signal: str, 
        confidence: float, 
        close_price: Optional[float] = None, 
        probabilities: Optional[Dict[str, float]] = None
    ):
        """
        Appends a trading decision to the CSV log.
        
        Args:
            ticker: The stock ticker symbol (e.g. AAPL)
            signal: BUY, SELL, or HOLD
            confidence: Float probability backing the signal
            close_price: Latest close price of the stock
            probabilities: Dictionary of individual model probabilities
        """
        timestamp = datetime.now().isoformat()
        close_str = f"{close_price:.4f}" if close_price is not None else "N/A"
        prob_str = str(probabilities) if probabilities is not None else "{}"
        
        row = [
            timestamp,
            ticker.upper(),
            signal.upper(),
            f"{confidence:.4f}",
            close_str,
            prob_str
        ]
        
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            logger.info(
                f"CSV log entry created: {ticker.upper()} | {signal.upper()} | Conf: {confidence:.2f} | Close: {close_str}"
            )
        except Exception as e:
            logger.error(f"Failed to write decision to CSV file: {e}")

# Instantiate global decision logger
decision_logger = DecisionLogger()
