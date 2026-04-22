import datetime
import pandas as pd
import yfinance as yf
from model.src.data_pipeline import preprocess, load_scaler
from model.src.forester_model import load_model
from model.src.golden_ratio import compute_fibonacci_levels, generate_signals

def run_live(ticker: str) -> dict:
    """
    Runs the live trading pipeline for a given ticker.
    """
    start_date = "2018-01-01"
    end_date = datetime.date.today().isoformat()
    
    # Fetch data from yfinance
    df_raw = yf.download(ticker, start=start_date, end=end_date)
    if df_raw.empty:
        raise ValueError(f"No data fetched for {ticker}")
        
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.get_level_values(0)
        
    # Preprocess
    df_preprocessed = preprocess(df_raw)
    
    # Load scaler and transform only the required features
    scaler = load_scaler(ticker)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                'RSI', 'MACD', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'Daily_Return']
    features = [f for f in features if f in df_preprocessed.columns]
    
    # Scale features
    df_scaled = df_preprocessed.copy()
    df_scaled.loc[:, features] = scaler.transform(df_scaled[features])
    
    # Input feature vector is the last row of scaled data
    X_live = df_scaled[features].iloc[[-1]]
    
    # Load model and predict
    model = load_model(ticker)
    pred_val = model.predict(X_live)[0]
    
    # Compute Fibonacci levels from unscaled close prices
    global_high = float(df_preprocessed['Close'].max())
    global_low = float(df_preprocessed['Close'].min())
    fib_levels = compute_fibonacci_levels(global_high, global_low)
    
    # Generate signal
    last_date = df_preprocessed.index[-1]
    last_date_str = last_date.date().isoformat() if isinstance(last_date, pd.Timestamp) else str(last_date)
    predicted_series = pd.Series([pred_val], index=[last_date_str])
    signal_series = generate_signals(predicted_series, fib_levels)
    signal = str(signal_series.iloc[0])
    
    return {
        "ticker": ticker,
        "last_close": float(df_preprocessed['Close'].iloc[-1]),
        "predicted_next_close": float(pred_val),
        "signal": signal,
        "fib_levels": fib_levels,
        "as_of_date": end_date
    }
