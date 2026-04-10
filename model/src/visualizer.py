"""
Visualizer module.
Generates and saves plots.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from model.config.settings import OUTPUTS_PLOTS_DIR
import numpy as np

# FIX for Windows Matplotlib 3.8.3 path deepcopy Stack Overflow
plt.rcParams['path.simplify'] = False

def plot_predictions(ticker: str, actual: pd.Series, predicted: pd.Series) -> None:
    """
    Actual vs predicted price chart.
    """
    plt.figure(figsize=(14, 7))
    
    dates = actual.index.to_numpy()
    y_actual = actual.to_numpy(dtype=float)
    y_pred = predicted.to_numpy(dtype=float)
    
    plt.plot(dates, y_actual, label='Actual Price (₹)', color='blue', alpha=0.6)
    plt.plot(dates, y_pred, label='Predicted Price (₹)', color='orange', alpha=0.8)
    
    plt.title(f"{ticker} - Actual vs Predicted Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (INR ₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(OUTPUTS_PLOTS_DIR / f"{ticker}_predictions.png"))
    plt.close()

def plot_signals(ticker: str, df: pd.DataFrame) -> None:
    """
    Price chart with Buy (green ▲) / Sell (red ▼) markers.
    """
    plt.figure(figsize=(14, 7))
    
    # Pure numpy conversion to bypass any pandas self-reference copies
    dates = df.index.to_numpy()
    y_actual = df['Actual'].to_numpy(dtype=float)
    
    plt.plot(dates, y_actual, label='Actual Price (₹)', color='black', alpha=0.5)
    
    buys = df[df['Signal'] == 'Buy']
    sells = df[df['Signal'] == 'Sell']
    
    if len(buys) > 0:
        b_dates = buys.index.to_numpy()
        b_actual = buys['Actual'].to_numpy(dtype=float)
        plt.scatter(b_dates, b_actual, marker='^', color='green', label='Buy Signal', s=100)
    if len(sells) > 0:
        s_dates = sells.index.to_numpy()
        s_actual = sells['Actual'].to_numpy(dtype=float)
        plt.scatter(s_dates, s_actual, marker='v', color='red', label='Sell Signal', s=100)
    
    plt.title(f"{ticker} - Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Price (INR ₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(OUTPUTS_PLOTS_DIR / f"{ticker}_signals.png"))
    plt.close()

def plot_portfolio(ticker: str, portfolio_df: pd.DataFrame) -> None:
    """
    Portfolio value over time vs buy-and-hold.
    """
    plt.figure(figsize=(14, 7))
    
    initial_capital = portfolio_df['Capital'].iloc[0] if portfolio_df['Shares'].iloc[0] == 0 else portfolio_df['Portfolio_Value'].iloc[0]
    initial_price = portfolio_df['Price'].iloc[0]
    shares_bnh = initial_capital / initial_price
    buy_hold_value = portfolio_df['Price'] * shares_bnh
    
    dates = portfolio_df.index.to_numpy()
    y_port = portfolio_df['Portfolio_Value'].to_numpy(dtype=float)
    y_bh = buy_hold_value.to_numpy(dtype=float)
    
    plt.plot(dates, y_port, label='Strategy Portfolio (₹)', color='purple', linewidth=2)
    plt.plot(dates, y_bh, label='Buy & Hold Portfolio (₹)', color='grey', linestyle='dashed')
    
    plt.title(f"{ticker} - Portfolio Performance vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (INR ₹)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(OUTPUTS_PLOTS_DIR / f"{ticker}_portfolio.png"))
    plt.close()

def plot_fibonacci(ticker: str, df: pd.DataFrame, fib_levels: dict) -> None:
    """
    Price chart with horizontal Fibonacci lines.
    """
    plt.figure(figsize=(14, 7))
    
    dates = df.index.to_numpy()
    y_actual = df['Actual'].to_numpy(dtype=float)
    plt.plot(dates, y_actual, label='Actual Price (₹)', color='blue')
    
    colors = ['r', 'g', 'm', 'c', 'orange']
    for i, (level_name, price_val) in enumerate(fib_levels.items()):
        plt.axhline(y=float(price_val), color=colors[i % len(colors)], linestyle='--', label=f'{level_name} (₹{float(price_val):.2f})', alpha=0.7)
        
    plt.title(f"{ticker} - Fibonacci Support & Resistance Levels")
    plt.xlabel("Date")
    plt.ylabel("Price (INR ₹)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(OUTPUTS_PLOTS_DIR / f"{ticker}_fibonacci.png"))
    plt.close()

def plot_all(ticker: str, df: pd.DataFrame, portfolio_df: pd.DataFrame) -> None:
    """
    Call all above and save.
    """
    print(f"[{ticker}] plotting predictions...")
    plot_predictions(ticker, df['Actual'], df['Predicted'])
    
    print(f"[{ticker}] plotting signals...")
    plot_signals(ticker, df)
    
    print(f"[{ticker}] plotting portfolio...")
    plot_portfolio(ticker, portfolio_df)
    
    print(f"[{ticker}] plotting fibonacci...")
    fib_cols = [c for c in df.columns if c.startswith('Fib_')]
    fib_levels = {c: df[c].iloc[0] for c in fib_cols}
    plot_fibonacci(ticker, df, fib_levels)
