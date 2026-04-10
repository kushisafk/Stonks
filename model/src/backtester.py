"""
Backtester module.
Simulates trading based on signals and computes performance metrics.
"""
import json
import pandas as pd
import numpy as np
from model.config.settings import OUTPUTS_RESULTS_DIR

def simulate_trades(df: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Start with INITIAL_CAPITAL in INR
    Buy/sell based on signals column
    Track portfolio value over time
    No fractional shares
    """
    portfolio = []
    capital = initial_capital
    shares = 0
    
    for date, row in df.iterrows():
        price = row['Actual']
        signal = row['Signal']
        
        # Execute trade
        if signal == 'Buy' and capital >= price:
            shares_to_buy = int(capital // price)
            shares += shares_to_buy
            capital -= shares_to_buy * price
        elif signal == 'Sell' and shares > 0:
            capital += shares * price
            shares = 0
            
        portfolio_value = capital + (shares * price)
        portfolio.append({
            'Date': date,
            'Price': price,
            'Signal': signal,
            'Capital': capital,
            'Shares': shares,
            'Portfolio_Value': portfolio_value
        })
        
    portfolio_df = pd.DataFrame(portfolio).set_index('Date')
    return portfolio_df

def compute_metrics(portfolio_df: pd.DataFrame) -> dict:
    """
    Computes performance metrics.
    """
    initial_val = portfolio_df['Portfolio_Value'].iloc[0]
    final_val = portfolio_df['Portfolio_Value'].iloc[-1]
    
    total_return = ((final_val / initial_val) - 1) * 100
    
    days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
    years = days / 365.25
    annualized_return = ((final_val / initial_val) ** (1 / years) - 1) * 100 if years > 0 else 0
    
    rolling_max = portfolio_df['Portfolio_Value'].cummax()
    drawdown = (portfolio_df['Portfolio_Value'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    daily_returns = portfolio_df['Portfolio_Value'].pct_change().dropna()
    if daily_returns.std() != 0:
        sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std())
    else:
        sharpe_ratio = 0
        
    start_price = portfolio_df['Price'].iloc[0]
    end_price = portfolio_df['Price'].iloc[-1]
    buy_hold_return = ((end_price / start_price) - 1) * 100
    
    trades = 0
    winning_trades = 0
    buy_price = 0
    
    for idx, row in portfolio_df.iterrows():
        if row['Signal'] == 'Buy' and buy_price == 0:
            buy_price = row['Price']
        elif row['Signal'] == 'Sell' and buy_price != 0:
            trades += 1
            if row['Price'] > buy_price:
                winning_trades += 1
            buy_price = 0
            
    win_rate = (winning_trades / trades * 100) if trades > 0 else 0
    
    return {
        "Total_Return_Pct": round(total_return, 2),
        "Annualized_Return_Pct": round(annualized_return, 2),
        "Max_Drawdown_Pct": round(max_drawdown, 2),
        "Sharpe_Ratio": round(sharpe_ratio, 2),
        "Win_Rate_Pct": round(win_rate, 2),
        "Buy_and_Hold_Return_Pct": round(buy_hold_return, 2)
    }

def save_results(metrics: dict, ticker: str) -> None:
    """
    Saves to outputs/results/ as CSV and JSON.
    """
    json_path = OUTPUTS_RESULTS_DIR / f"{ticker}_metrics.json"
    csv_path = OUTPUTS_RESULTS_DIR / f"{ticker}_metrics.csv"
    
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)

def run_backtest(ticker: str, signal_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Full backtest pipeline
    """
    from model.config.settings import INITIAL_CAPITAL
    portfolio_df = simulate_trades(signal_df, INITIAL_CAPITAL)
    metrics = compute_metrics(portfolio_df)
    save_results(metrics, ticker)
    return portfolio_df, metrics
