from datetime import datetime
from typing import Dict, List, Optional
from stonks.session.schemas import SessionState, Watchlist, WatchlistItem
from stonks.session.exceptions import WatchlistNotFoundError, InvalidStateError

class WatchlistService:
    """Manages watchlist groupings and symbol items in the session state."""
    
    def add_watchlist(self, state: SessionState, name: str) -> None:
        """Creates a new named watchlist grouping."""
        name = name.strip()
        if not name:
            raise InvalidStateError("Watchlist name cannot be empty.")
        if name in state.watchlists:
            raise InvalidStateError(f"Watchlist '{name}' already exists.")
        state.watchlists[name] = Watchlist(name=name)
        
    def rename_watchlist(self, state: SessionState, old_name: str, new_name: str) -> None:
        """Renames an existing watchlist grouping."""
        old_name = old_name.strip()
        new_name = new_name.strip()
        if old_name not in state.watchlists:
            raise WatchlistNotFoundError(f"Watchlist '{old_name}' not found.")
        if not new_name:
            raise InvalidStateError("New watchlist name cannot be empty.")
        if new_name in state.watchlists:
            raise InvalidStateError(f"Watchlist '{new_name}' already exists.")
            
        watchlist = state.watchlists.pop(old_name)
        watchlist.name = new_name
        state.watchlists[new_name] = watchlist
        
    def add_ticker(
        self, 
        state: SessionState, 
        watchlist_name: str, 
        ticker: str, 
        tags: List[str] = None, 
        notes: str = "", 
        priority: int = 2, 
        target_price: Optional[float] = None
    ) -> None:
        """Adds a ticker to a specific watchlist with metadata tags and priority settings."""
        watchlist_name = watchlist_name.strip()
        ticker = ticker.strip().upper()
        
        if watchlist_name not in state.watchlists:
            raise WatchlistNotFoundError(f"Watchlist '{watchlist_name}' not found.")
            
        watchlist = state.watchlists[watchlist_name]
        item = WatchlistItem(
            ticker=ticker,
            date_added=datetime.now().isoformat(),
            tags=tags or [],
            notes=notes,
            priority=priority,
            target_price=target_price
        )
        watchlist.items[ticker] = item
        
    def remove_ticker(self, state: SessionState, watchlist_name: str, ticker: str) -> None:
        """Removes a ticker from a specific watchlist."""
        watchlist_name = watchlist_name.strip()
        ticker = ticker.strip().upper()
        
        if watchlist_name not in state.watchlists:
            raise WatchlistNotFoundError(f"Watchlist '{watchlist_name}' not found.")
            
        watchlist = state.watchlists[watchlist_name]
        if ticker not in watchlist.items:
            raise InvalidStateError(f"Ticker '{ticker}' not found in watchlist '{watchlist_name}'.")
        watchlist.items.pop(ticker)
