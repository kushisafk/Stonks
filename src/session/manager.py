from pathlib import Path
from typing import Dict, List, Optional, Any
from src.session.schemas import SessionState, Watchlist, Position, Portfolio, Preferences, DecisionRecord, Alert, TradingStyle, RiskProfile
from src.session.persistence import PersistenceManager
from src.session.watchlist import WatchlistService
from src.session.positions import PositionService
from src.session.portfolio import PortfolioService
from src.session.history import HistoryService
from src.session.alerts import AlertService
from src.session.preferences import PreferencesService
from src.session.exceptions import InvalidStateError

class TradingSessionManager:
    """Unified Facade controlling state changes, service orchestration, and automatic persistence."""
    
    def __init__(self, filepath: Path):
        self.persistence = PersistenceManager(filepath)
        self.state: Optional[SessionState] = None
        
        self.watchlist_service = WatchlistService()
        self.position_service = PositionService()
        self.portfolio_service = PortfolioService()
        self.history_service = HistoryService()
        self.alert_service = AlertService()
        self.preferences_service = PreferencesService()
        
    def _ensure_session(self) -> None:
        """Helper to assert that a session is loaded before any state operation."""
        if self.state is None:
            raise InvalidStateError("No active session loaded. Create or load a session first.")
            
    def create_session(self) -> None:
        """Initializes a new SessionState block and persists it immediately."""
        self.state = SessionState()
        self.save_session()
        
    def load_session(self) -> None:
        """Loads and recovers session state from disk, recalculating portfolio balance metrics."""
        self.state = self.persistence.load()
        self.portfolio_service.update_portfolio_metrics(self.state)
        
    def save_session(self) -> None:
        """Forces atomic persistence of current session state."""
        self._ensure_session()
        self.persistence.save(self.state)
        
    # --- Watchlist API ---
    def get_watchlists(self) -> Dict[str, Watchlist]:
        self._ensure_session()
        return self.state.watchlists
        
    def add_watchlist(self, name: str) -> None:
        self._ensure_session()
        self.watchlist_service.add_watchlist(self.state, name)
        self.save_session()
        
    def rename_watchlist(self, old_name: str, new_name: str) -> None:
        self._ensure_session()
        self.watchlist_service.rename_watchlist(self.state, old_name, new_name)
        self.save_session()
        
    def track_symbol(
        self, 
        watchlist_name: str, 
        ticker: str, 
        tags: List[str] = None, 
        notes: str = "", 
        priority: int = 2, 
        target_price: Optional[float] = None
    ) -> None:
        self._ensure_session()
        self.watchlist_service.add_ticker(self.state, watchlist_name, ticker, tags, notes, priority, target_price)
        self.save_session()
        
    def untrack_symbol(self, watchlist_name: str, ticker: str) -> None:
        self._ensure_session()
        self.watchlist_service.remove_ticker(self.state, watchlist_name, ticker)
        self.save_session()
        
    # --- Position API ---
    def open_long_position(
        self, 
        ticker: str, 
        entry_price: float, 
        quantity: float, 
        stop_loss: Optional[float] = None, 
        take_profit: Optional[float] = None
    ) -> None:
        self._ensure_session()
        self.position_service.open_long(self.state, ticker, entry_price, quantity, stop_loss, take_profit)
        self.portfolio_service.update_portfolio_metrics(self.state)
        self.save_session()
        
    def open_short_position(
        self, 
        ticker: str, 
        entry_price: float, 
        quantity: float, 
        stop_loss: Optional[float] = None, 
        take_profit: Optional[float] = None
    ) -> None:
        self._ensure_session()
        self.position_service.open_short(self.state, ticker, entry_price, quantity, stop_loss, take_profit)
        self.portfolio_service.update_portfolio_metrics(self.state)
        self.save_session()
        
    def close_position(self, ticker: str, exit_price: float) -> None:
        self._ensure_session()
        self.position_service.close_position(self.state, ticker, exit_price)
        self.portfolio_service.update_portfolio_metrics(self.state)
        self.save_session()
        
    def partial_close_position(self, ticker: str, exit_price: float, quantity: float) -> None:
        self._ensure_session()
        self.position_service.partial_close(self.state, ticker, exit_price, quantity)
        self.portfolio_service.update_portfolio_metrics(self.state)
        self.save_session()
        
    def update_stop_loss(self, ticker: str, stop_loss: Optional[float]) -> None:
        self._ensure_session()
        self.position_service.update_stop_loss(self.state, ticker, stop_loss)
        self.save_session()
        
    def update_take_profit(self, ticker: str, take_profit: Optional[float]) -> None:
        self._ensure_session()
        self.position_service.update_take_profit(self.state, ticker, take_profit)
        self.save_session()
        
    # --- Portfolio API ---
    def get_portfolio(self) -> Portfolio:
        self._ensure_session()
        return self.state.portfolio
        
    def update_portfolio_metrics(self, current_prices: Optional[Dict[str, float]] = None) -> None:
        self._ensure_session()
        self.portfolio_service.update_portfolio_metrics(self.state, current_prices)
        self.save_session()
        
    def get_positions(self) -> Dict[str, Position]:
        self._ensure_session()
        return self.state.positions
        
    def get_sector_exposures(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        self._ensure_session()
        return self.portfolio_service.get_sector_exposures(self.state, current_prices)
        
    # --- History API ---
    def record_decision(self, record: DecisionRecord) -> None:
        self._ensure_session()
        self.history_service.record_decision(self.state, record)
        self.save_session()
        
    def get_recent_decisions(self, limit: int = 100) -> List[DecisionRecord]:
        self._ensure_session()
        return self.history_service.get_recent_decisions(self.state, limit)
        
    def get_ticker_history(self, ticker: str) -> List[DecisionRecord]:
        self._ensure_session()
        return self.history_service.get_ticker_history(self.state, ticker)
        
    def get_recommendation_history(self, recommendation: str) -> List[DecisionRecord]:
        self._ensure_session()
        return self.history_service.get_recommendation_history(self.state, recommendation)
        
    # --- Alert API ---
    def record_alert(self, alert: Alert) -> None:
        self._ensure_session()
        self.alert_service.record_alert(self.state, alert)
        self.save_session()
        
    def get_alerts(self, triggered_only: bool = True) -> List[Alert]:
        self._ensure_session()
        return self.alert_service.get_alerts(self.state, triggered_only)
        
    def clear_alerts(self) -> None:
        self._ensure_session()
        self.alert_service.clear_alerts(self.state)
        self.save_session()
        
    # --- Preferences API ---
    def get_preferences(self) -> Preferences:
        self._ensure_session()
        return self.state.preferences
        
    def update_preferences(self, **kwargs) -> None:
        self._ensure_session()
        self.preferences_service.update_preferences(self.state, **kwargs)
        self.save_session()
        
    def update_profile(self, **kwargs) -> None:
        self._ensure_session()
        self.preferences_service.update_profile(self.state, **kwargs)
        self.save_session()
