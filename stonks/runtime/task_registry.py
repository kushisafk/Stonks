from typing import Dict, Callable, Any, Tuple

class TaskRegistry:
    """Central registry of named tasks that can be mapped to scheduled jobs."""
    
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        
    def register(self, name: str, func: Callable, default_interval: int = 60, priority: int = 5) -> None:
        """Registers a named task function and its default scheduling configurations."""
        self._tasks[name] = {
            "func": func,
            "default_interval": default_interval,
            "priority": priority
        }
        
    def get(self, name: str) -> Dict[str, Any]:
        """Retrieves task configurations by name."""
        return self._tasks.get(name)
        
    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """Lists all registered task templates."""
        return dict(self._tasks)
