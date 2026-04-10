"""
Golden Ratio module.
Computes Fibonacci levels and generates trading signals.
"""
import pandas as pd
from model.config.settings import FIBONACCI_LEVELS, GOLDEN_RATIO

def compute_fibonacci_levels(high: float, low: float) -> dict:
    """
    Uses GOLDEN_RATIO and FIBONACCI_LEVELS to compute support/resistance levels.
    """
    diff = high - low
    levels = {}
    for level in FIBONACCI_LEVELS:
        levels[f'Fib_{level}'] = high - diff * level
        
    return levels

def generate_signals(predicted_prices: pd.Series, fib_levels: dict) -> pd.Series:
    """
    Buy if predicted price bounces off support (0.618 or 0.786 level)
    Sell if predicted price hits resistance (0.236 or 0.382 level)
    Hold otherwise
    """
    signals = pd.Series('Hold', index=predicted_prices.index)
    
    threshold = 0.015  # 1.5% tolerance
    
    # Typically from High to Low in retracement:
    # 0.236 & 0.382 are upper boundaries (Resistance)
    # 0.618 & 0.786 are lower boundaries (Support)
    support_levels = [fib_levels['Fib_0.618'], fib_levels['Fib_0.786']]
    resistance_levels = [fib_levels['Fib_0.236'], fib_levels['Fib_0.382']]
    
    for date, pred in predicted_prices.items():
        # Check support (Buy)
        if any(abs(pred - sup) / sup <= threshold for sup in support_levels):
            signals[date] = 'Buy'
        # Check resistance (Sell)
        elif any(abs(pred - res) / res <= threshold for res in resistance_levels):
            signals[date] = 'Sell'
            
    return signals

def run_golden_ratio(ticker: str, predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns df with signals column.
    """
    df = predictions_df.copy()
    
    global_high = df['Actual'].max()
    global_low = df['Actual'].min()
    fib_levels = compute_fibonacci_levels(global_high, global_low)
    
    signal_series = generate_signals(df['Predicted'], fib_levels)
    df.loc[:, 'Signal'] = signal_series
    
    for k, v in fib_levels.items():
        df.loc[:, k] = v
        
    return df
