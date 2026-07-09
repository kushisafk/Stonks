"""
STONKS Background Runtime Demo

Demonstrates how to initialize the Session Manager, start the event-driven 
background runtime thread pool, register custom listeners, and stop it gracefully.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stonks.session.manager import TradingSessionManager
from stonks.runtime.runtime import StonksRuntime
from stonks.runtime.events import PriceUpdateEvent, RecommendationChangedEvent
from stonks.logging.logger import logger

def custom_price_listener(event):
    """Callback function triggered when PriceUpdateEvent is published."""
    print(f"\n[EVENT DETECTED] Price Update: {event.data.get('ticker')} is currently ${event.data.get('price'):.2f}\n")

def main():
    # Load default session file in temp directory for testing
    session_file = Path("models_data/session.json")
    manager = TradingSessionManager(session_file)
    manager.load_session()
    
    # Initialize runtime orchestrator
    runtime = StonksRuntime(manager)
    
    # Register a custom listener to the Event Bus dynamically
    runtime.event_bus.subscribe(PriceUpdateEvent, custom_price_listener)
    
    logger.info("Starting background runtime threads...")
    runtime.start()
    
    try:
        print("\n" + "=" * 50)
        print("Runtime is running in the background.")
        print("Metrics and heartbeats will log every few seconds.")
        print("Waiting 15 seconds before terminating...")
        print("=" * 50 + "\n")
        
        # Simulate active events trigger
        time.sleep(3.0)
        print("[Inference Simulator] Publishing Mock Price Update event...")
        runtime.event_bus.publish(PriceUpdateEvent(data={"ticker": "AAPL", "price": 182.40}))
        
        time.sleep(12.0)
        
    except KeyboardInterrupt:
        pass
        
    print("\n" + "=" * 50)
    logger.info("Stopping background runtime threads...")
    runtime.stop()
    print("=" * 50 + "\n")

if __name__ == "__main__":
    main()
