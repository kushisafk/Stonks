from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class TradingStyle(str, Enum):
    INVESTOR = "Investor"
    SWING_TRADER = "Swing Trader"
    DAY_TRADER = "Day Trader"
    SCALPER = "Scalper"
    CUSTOM = "Custom"

class RiskProfile(str, Enum):
    CONSERVATIVE = "Conservative"
    BALANCED = "Balanced"
    AGGRESSIVE = "Aggressive"
    CUSTOM = "Custom"

class PositionType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(str, Enum):
    OPEN = "OPEN"
    PARTIAL = "PARTIAL"
    CLOSED = "CLOSED"

class UserProfile(BaseModel):
    username: str = Field(default="Default User")
    trading_style: TradingStyle = Field(default=TradingStyle.SWING_TRADER)
    risk_profile: RiskProfile = Field(default=RiskProfile.BALANCED)
    default_capital: float = Field(default=100000.0)
    base_currency: str = Field(default="USD")
    timezone: str = Field(default="UTC")
    preferred_notification_channels: List[str] = Field(default_factory=list)

class Preferences(BaseModel):
    polling_interval: int = Field(default=24)  # in hours
    theme: str = Field(default="dark")
    default_report_format: str = Field(default="markdown")
    preferred_output_style: str = Field(default="detailed")
    threshold_profile: Dict[str, float] = Field(default_factory=lambda: {"buy": 0.70, "sell": 0.40})
    auto_optimization_enabled: bool = Field(default=True)
    preferred_ml_model: str = Field(default="catboost")
    preferred_language: str = Field(default="en")

class WatchlistItem(BaseModel):
    ticker: str
    date_added: str
    tags: List[str] = Field(default_factory=list)
    notes: str = Field(default="")
    priority: int = Field(default=2)  # 1 (low), 2 (medium), 3 (high)
    target_price: Optional[float] = Field(default=None)

class Watchlist(BaseModel):
    name: str
    items: Dict[str, WatchlistItem] = Field(default_factory=dict)

class Position(BaseModel):
    ticker: str
    position_type: PositionType
    entry_price: float
    quantity: float
    entry_date: str
    current_stop_loss: Optional[float] = Field(default=None)
    current_take_profit: Optional[float] = Field(default=None)
    status: PositionStatus = Field(default=PositionStatus.OPEN)
    realized_pl: float = Field(default=0.0)
    sector: Optional[str] = Field(default=None)
    industry: Optional[str] = Field(default=None)

class Portfolio(BaseModel):
    cash_balance: float = Field(default=100000.0)
    buying_power: float = Field(default=100000.0)
    open_equity: float = Field(default=0.0)
    total_equity: float = Field(default=100000.0)
    portfolio_value: float = Field(default=100000.0)
    long_exposure: float = Field(default=0.0)
    short_exposure: float = Field(default=0.0)
    net_exposure: float = Field(default=0.0)
    largest_position: Optional[str] = Field(default=None)
    daily_pl: float = Field(default=0.0)
    unrealized_pl: float = Field(default=0.0)
    realized_pl: float = Field(default=0.0)

class DecisionRecord(BaseModel):
    timestamp: str
    ticker: str
    prediction: str
    probability: float
    risk_score: int
    confidence_tier: str
    recommendation: str
    reasoning: str
    warnings: List[str] = Field(default_factory=list)
    market_regime: str
    sentiment: float
    model_used: str
    decision_outcome: Optional[str] = Field(default=None)

class Alert(BaseModel):
    timestamp: str
    rule_type: str
    ticker: str
    message: str
    triggered: bool = Field(default=True)

class SessionState(BaseModel):
    schema_version: int = Field(default=1)
    user_profile: UserProfile = Field(default_factory=UserProfile)
    preferences: Preferences = Field(default_factory=Preferences)
    watchlists: Dict[str, Watchlist] = Field(default_factory=dict)
    positions: Dict[str, Position] = Field(default_factory=dict)
    portfolio: Portfolio = Field(default_factory=Portfolio)
    history: List[DecisionRecord] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)
