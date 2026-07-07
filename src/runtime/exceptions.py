class RuntimeException(Exception):
    """Base exception for all STONKS runtime modules."""
    pass

class EventBusError(RuntimeException):
    """Raised when event publishing or subscription fails."""
    pass

class WorkerPoolError(RuntimeException):
    """Raised when job submission or execution fails in worker pool."""
    pass

class SchedulerError(RuntimeException):
    """Raised when job scheduling or interval parsing fails."""
    pass

class AgentRegistrationError(RuntimeException):
    """Raised when registering a target agent fails."""
    pass
