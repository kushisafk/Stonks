class SessionError(Exception):
    """Base exception for all session-related operations in STONKS."""
    pass

class SessionLoadError(SessionError):
    """Raised when the session cannot be loaded or deserialized."""
    pass

class SessionSaveError(SessionError):
    """Raised when the session cannot be serialized or saved to disk."""
    pass

class InvalidStateError(SessionError):
    """Raised when an operation is requested on an invalid state (e.g. closing an already closed position)."""
    pass

class PositionNotFoundError(SessionError):
    """Raised when a requested position ticker is not found in the open positions."""
    pass

class WatchlistNotFoundError(SessionError):
    """Raised when a requested named watchlist does not exist."""
    pass
