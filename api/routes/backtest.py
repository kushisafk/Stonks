from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json
import pandas as pd
from model.config.settings import TICKERS, OUTPUTS_RESULTS_DIR, START_DATE, END_DATE, INITIAL_CAPITAL
from model.portfolio_summary import compute_portfolio_summary
from model.src.data_pipeline import run_data_pipeline
from model.src.forester_model import run_forester
from model.src.golden_ratio import run_golden_ratio
from model.src.backtester import simulate_trades, extract_trade_log

router = APIRouter()

class PortfolioSummaryResponse(BaseModel):
    avg_total_return_pct: float
    avg_annualized_return_pct: float
    avg_sharpe_ratio: float
    avg_max_drawdown_pct: float
    avg_win_rate_pct: float
    avg_buy_and_hold_return_pct: float
    tickers: List[str]

class EquityCurvePoint(BaseModel):
    date: str
    portfolio_value: float
    buy_hold_value: float

class TradeLogItem(BaseModel):
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl: float
    result: str

class BacktestTickerResponse(BaseModel):
    ticker: str
    metrics: Dict[str, Any]
    equity_curve: List[EquityCurvePoint]
    trade_log: List[TradeLogItem]

@router.get("/summary", response_model=PortfolioSummaryResponse)
def get_backtest_summary():
    ticker_metrics = {}
    for ticker in TICKERS:
        json_path = OUTPUTS_RESULTS_DIR / f"{ticker}_metrics.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Backtest results not found. Run training pipeline first.")
        
        with open(json_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            ticker_metrics[ticker] = metrics
            
    summary = compute_portfolio_summary(ticker_metrics)
    return PortfolioSummaryResponse(**summary)

@router.get("/{ticker}", response_model=BacktestTickerResponse)
def get_backtest_ticker(ticker: str):
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Invalid ticker.")
        
    json_path = OUTPUTS_RESULTS_DIR / f"{ticker}_metrics.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Backtest results not found. Run training pipeline first.")
        
    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    try:
        raw_df, X, y = run_data_pipeline(ticker, START_DATE, END_DATE)
        model, model_metrics, predictions_df = run_forester(ticker, X, y)
        signal_df = run_golden_ratio(ticker, predictions_df)
        portfolio_df = simulate_trades(signal_df, INITIAL_CAPITAL)
        
        raw_trade_log = extract_trade_log(portfolio_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    trade_log = []
    for item in raw_trade_log:
        trade_log.append(TradeLogItem(
            entry_date=item["entry_date"],
            exit_date=item["exit_date"],
            entry_price=round(item["entry_price"], 2),
            exit_price=round(item["exit_price"], 2),
            pnl=round(item["pnl"], 2),
            result=item["result"]
        ))
        
    first_price = float(portfolio_df['Price'].iloc[0])
    
    equity_curve = []
    for date, row in portfolio_df.iterrows():
        date_str = date.date().isoformat() if isinstance(date, pd.Timestamp) else str(date)
        price = float(row['Price'])
        port_val = round(float(row['Portfolio_Value']), 2)
        buy_hold = round((INITIAL_CAPITAL / first_price) * price, 2) if first_price != 0 else 0.0
        
        equity_curve.append(EquityCurvePoint(
            date=date_str,
            portfolio_value=port_val,
            buy_hold_value=buy_hold
        ))
        
    # ensure metric floats are rounded to 2 decimal places
    for k, v in metrics.items():
        if isinstance(v, float):
            metrics[k] = round(v, 2)
            
    return BacktestTickerResponse(
        ticker=ticker,
        metrics=metrics,
        equity_curve=equity_curve,
        trade_log=trade_log
    )
