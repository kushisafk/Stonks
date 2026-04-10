"""
Configuration settings for the Stonks trading model.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUTS_PLOTS_DIR = BASE_DIR / "outputs" / "plots"
OUTPUTS_RESULTS_DIR = BASE_DIR / "outputs" / "results"
MODELS_DIR = BASE_DIR / "outputs" / "models"

# Create directories if they don't exist
for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, OUTPUTS_PLOTS_DIR, OUTPUTS_RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Tickers
TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]

# Dates
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"

# Model Parameters
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

# Fibonacci & Golden Ratio
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]
GOLDEN_RATIO = 1.618

# Backtesting
INITIAL_CAPITAL = 100000.0  # INR
