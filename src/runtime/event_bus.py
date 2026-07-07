import threading
from typing import Dict, List, Callable, Type
from src.runtime.events import Event
from src.logging.logger import logger

class EventBus:
    """Thread-safe publish/subscribe event bus coordinating message dispatching."""
    
    def __init__(self):
        self._listeners: Dict[Type[Event], List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()
        
    def subscribe(self, event_type: Type[Event], listener: Callable[[Event], None]) -> None:
        """Subscribes a listener callback to an event type."""
        with self._lock:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            if listener not in self._listeners[event_type]:
                self._listeners[event_type].append(listener)
                
    def unsubscribe(self, event_type: Type[Event], listener: Callable[[Event], None]) -> None:
        """Removes a listener callback subscription."""
        with self._lock:
            if event_type in self._listeners and listener in self._listeners[event_type]:
                self._listeners[event_type].remove(listener)
                
    def publish(self, event: Event) -> None:
        """Publishes an event to all registered listener callbacks thread-safely."""
        event_type = type(event)
        listeners_to_call = []
        
        with self._lock:
            listeners_to_call.extend(self._listeners.get(event_type, []))
            # Also notify subscribers to the base Event class
            if Event in self._listeners and event_type != Event:
                listeners_to_call.extend(self._listeners.get(Event, []))
                
        for listener in listeners_to_call:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"EventBus: Error calling listener {listener.__name__ if hasattr(listener, '__name__') else listener} for event {event_type.__name__}: {e}")
