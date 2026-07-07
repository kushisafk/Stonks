import uuid
from datetime import datetime
from typing import Dict, Any, Optional

class Event:
    """Base event representation containing common metadata."""
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self.event_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.data = data or {}
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.event_id}, time={self.timestamp}, data={self.data})"

class TimerEvent(Event):
    """Triggered on scheduled interval beats."""
    pass

class PriceUpdateEvent(Event):
    """Triggered when fresh market prices are retrieved."""
    pass

class NewsUpdateEvent(Event):
    """Triggered when fresh articles are indexed."""
    pass

class RecommendationChangedEvent(Event):
    """Triggered when an asset recommendation shifts (e.g. HOLD -> BUY)."""
    pass

class AlertTriggeredEvent(Event):
    """Triggered when an alert constraint is hit."""
    pass

class PositionOpenedEvent(Event):
    """Triggered when a LONG or SHORT position is opened."""
    pass

class PositionClosedEvent(Event):
    """Triggered when a position is closed."""
    pass

class WatchlistUpdatedEvent(Event):
    """Triggered when watchlist items are added or removed."""
    pass

class PortfolioChangedEvent(Event):
    """Triggered when cash, equity, or allocations update."""
    pass

class SessionLoadedEvent(Event):
    """Triggered when state is loaded from session.json."""
    pass

class SessionSavedEvent(Event):
    """Triggered when state is persisted to disk."""
    pass

class RuntimeStartedEvent(Event):
    """Triggered when the central STONKS runtime starts."""
    pass

class RuntimeStoppedEvent(Event):
    """Triggered when the runtime stops."""
    pass
