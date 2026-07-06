from typing import List
from src.session.schemas import SessionState, DecisionRecord

class HistoryService:
    """Manages the persistence and retrieval of trading decision records inside the session state."""
    
    def record_decision(self, state: SessionState, record: DecisionRecord) -> None:
        """Appends a new decision record to the history list."""
        state.history.append(record)
        
    def get_recent_decisions(self, state: SessionState, limit: int = 100) -> List[DecisionRecord]:
        """Returns the most recent decision records up to the specified limit."""
        return state.history[-limit:]
        
    def get_ticker_history(self, state: SessionState, ticker: str) -> List[DecisionRecord]:
        """Filters and returns all decision records matching a specific ticker symbol."""
        ticker = ticker.strip().upper()
        return [r for r in state.history if r.ticker == ticker]
        
    def get_recommendation_history(self, state: SessionState, recommendation: str) -> List[DecisionRecord]:
        """Filters and returns all decision records matching a specific recommendation type (e.g. BUY)."""
        recommendation = recommendation.strip().upper()
        return [r for r in state.history if r.recommendation.upper() == recommendation]
