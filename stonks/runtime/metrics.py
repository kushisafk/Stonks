import threading
from typing import Dict, Any

class RuntimeMetrics:
    """Thread-safe collector gathering execution counts, latencies, and queue lengths."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {
            "jobs_executed": 0,
            "jobs_failed": 0,
            "analysis_time": 0.0,
            "queue_length": 0,
            "events_published": 0,
            "events_processed": 0,
            "recommendation_time": 0.0,
            "uptime_seconds": 0
        }
        self._lock = threading.Lock()
        
    def increment(self, name: str, count: int = 1) -> None:
        """Increments a numeric metric counter."""
        with self._lock:
            if name in self._metrics:
                # Ensure the field is numeric before incrementing
                self._metrics[name] = self._metrics[name] + count
                
    def record_value(self, name: str, value: Any) -> None:
        """Records a specific metric value (overwriting previous value)."""
        with self._lock:
            self._metrics[name] = value
            
    def get_all(self) -> Dict[str, Any]:
        """Returns a copy of all current metric values."""
        with self._lock:
            return dict(self._metrics)
