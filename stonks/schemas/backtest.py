from pydantic import BaseModel, Field
from typing import Dict

class MLMetrics(BaseModel):
    accuracy: float = Field(..., description="Proportion of correct price direction predictions")
    precision: float = Field(..., description="Precision of price increase predictions")
    recall: float = Field(..., description="Recall of price increase predictions")
    f1: float = Field(..., description="Harmonic mean of precision and recall")

class TradingMetrics(BaseModel):
    starting_capital: float = Field(..., description="Initial capital allocated to strategy")
    ending_capital: float = Field(..., description="Final cash and asset value at simulation end")
    strategy_return: float = Field(..., description="Total return of the strategy (%)")
    buy_and_hold_return: float = Field(..., description="Total return of friction-adjusted buy and hold (%)")
    win_rate: float = Field(..., description="Percentage of executed trades that closed at a net profit")
    total_trades: int = Field(..., description="Total number of round-trip trades executed")
    annualized_volatility: float = Field(..., description="Annualized standard deviation of strategy returns")
    sharpe_ratio: float = Field(..., description="Annualized Sharpe Ratio based on a risk-free rate")
    max_drawdown: float = Field(..., description="Maximum peak-to-trough drop in portfolio equity (%)")

class BacktestResponse(BaseModel):
    symbol: str = Field(..., description="Asset ticker symbol backtested")
    ml_metrics: MLMetrics = Field(..., description="Machine learning performance classification scores")
    trading_metrics: TradingMetrics = Field(..., description="Quantitative portfolio execution scores")
