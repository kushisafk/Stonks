from enum import Enum

class StateType(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SHUTTING_DOWN = "SHUTTING_DOWN"

class RuntimeState:
    """Synchronized state indicator tracking active lifecycle phases of the runtime."""
    
    def __init__(self):
        self._state = StateType.STOPPED
        
    def set(self, new_state: StateType) -> None:
        self._state = new_state
        
    def get(self) -> StateType:
        return self._state
        
    @property
    def is_active(self) -> bool:
        return self._state in (StateType.RUNNING, StateType.STARTING)
