"""
Data Pipeline for the Stonks trading model.
Fetches, preprocesses, and prepares feature data.
"""
import joblib
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from model.config.settings import DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR

def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads OHLCV data from yfinance and saves raw CSV.
    """
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        raise ValueError(f"No data fetched for {ticker}")
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    csv_path = DATA_RAW_DIR / f"{ticker}_raw.csv"
    df.to_csv(csv_path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and computes technical indicators.
    Indicators: SMA 20/50, RSI, MACD, Bollinger Bands, daily returns.
    """
    df = df.copy()
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    df.loc[:, 'SMA_20'] = df['Close'].rolling(window=20).mean()
    df.loc[:, 'SMA_50'] = df['Close'].rolling(window=50).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df.loc[:, 'RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df.loc[:, 'MACD'] = exp1 - exp2
    
    df.loc[:, 'BB_Mid'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df.loc[:, 'BB_Upper'] = df['BB_Mid'] + (std * 2)
    df.loc[:, 'BB_Lower'] = df['BB_Mid'] - (std * 2)
    
    df.loc[:, 'Daily_Return'] = df['Close'].pct_change()
    
    df.dropna(inplace=True)
    return df

def normalize(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Applies MinMaxScaler on features.
    """
    df_norm = df.copy()
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                'RSI', 'MACD', 'BB_Mid', 'BB_Upper', 'BB_Lower', 'Daily_Return']
    
    features = [f for f in features if f in df_norm.columns]
    scaler = MinMaxScaler()
    df_norm.loc[:, features] = scaler.fit_transform(df_norm[features])
    
    scaler_path = MODELS_DIR / f"{ticker}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    
    return df_norm

def load_scaler(ticker: str) -> MinMaxScaler:
    """
    Loads completed MinMaxScaler for a ticker.
    """
    scaler_path = MODELS_DIR / f"{ticker}_scaler.pkl"
    return joblib.load(scaler_path)

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates X (features) and y (next day closing price).
    Returns (X, y).
    Uses unnormalized target.
    """
    df_feat = df.copy()
    df_feat.loc[:, 'Target'] = df_feat['Close'].shift(-1)
    df_feat.dropna(inplace=True)
    
    # Target is actual next day closing price
    y = df_feat['Target']
    X = df_feat.drop(columns=['Target'])
    return X, y

def save_processed(df: pd.DataFrame, ticker: str) -> None:
    """
    Saves processed dataframe to data/processed/.
    """
    csv_path = DATA_PROCESSED_DIR / f"{ticker}_processed.csv"
    df.to_csv(csv_path)

def run_data_pipeline(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Executes the full data pipeline.
    """
    raw_df = fetch_data(ticker, start, end)
    df = preprocess(raw_df)
    
    save_processed(df, ticker)
    
    norm_df = normalize(df, ticker)
    X, y = prepare_features(norm_df)
    
    return df, X, y
