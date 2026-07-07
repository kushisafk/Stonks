import time
import threading
from typing import Dict, Any, Optional
from src.session.manager import TradingSessionManager
from src.runtime.events import (
    RuntimeStartedEvent, RuntimeStoppedEvent, 
    RecommendationChangedEvent, PriceUpdateEvent, PortfolioChangedEvent, SessionSavedEvent
)
from src.runtime.event_bus import EventBus
from src.runtime.worker_pool import WorkerPool
from src.runtime.scheduler import RuntimeScheduler
from src.runtime.agent_manager import AgentManager
from src.runtime.task_registry import TaskRegistry
from src.runtime.runtime_state import RuntimeState, StateType
from src.runtime.metrics import RuntimeMetrics
from src.runtime.heartbeat import RuntimeHeartbeat
from src.runtime.dispatcher import EventDispatcher
from src.logging.logger import logger

class StonksRuntime:
    """The central operating runtime engine, managing scheduler, worker pool, event bus, and agents."""
    
    def __init__(self, session_manager: TradingSessionManager):
        self.manager = session_manager
        
        # Initialize core runtime modules
        self.state = RuntimeState()
        self.metrics = RuntimeMetrics()
        self.event_bus = EventBus()
        self.worker_pool = WorkerPool(num_workers=4, metrics=self.metrics)
        self.scheduler = RuntimeScheduler(self.worker_pool)
        self.agent_manager = AgentManager()
        self.task_registry = TaskRegistry()
        self.dispatcher = EventDispatcher(self.manager, self.event_bus)
        
        # Track start time and active heartbeat
        self.start_time: float = 0.0
        self.heartbeat: Optional[RuntimeHeartbeat] = None
        
        # In-memory memory recommendation cache to detect changes
        self._last_recs: Dict[str, str] = {}
        
    def start(self) -> None:
        """Starts the event bus, worker threads, scheduler, registers jobs, and runs the runtime loop."""
        if self.state.is_active:
            logger.warning("StonksRuntime: Runtime is already active.")
            return
            
        self.state.set(StateType.STARTING)
        self.start_time = time.time()
        logger.info("StonksRuntime: Initializing Event-driven Trading operating system runtime...")
        
        # 1. Start Worker Pool
        self.worker_pool.start()
        
        # 2. Register Dispatcher Listeners
        self.dispatcher.register_listeners()
        
        # 3. Load previous recommendation states from session history to avoid false alert triggers
        self._init_recommendation_cache()
        
        # 4. Register Schedulable default tasks
        self.task_registry.register("watchlist_sweep", self._run_watchlist_sweep, default_interval=300, priority=3)
        self.task_registry.register("session_autosave", self._run_session_autosave, default_interval=60, priority=9)
        self.task_registry.register("heartbeat", self._run_heartbeat, default_interval=10, priority=10)
        
        # Add tasks to scheduler
        for name, task in self.task_registry.list_all().items():
            self.scheduler.add_job(
                func=task["func"],
                interval_seconds=task["default_interval"],
                priority=task["priority"]
            )
            
        # 5. Start Scheduler thread
        self.scheduler.start()
        
        # 6. Initialize Heartbeat monitor
        self.heartbeat = RuntimeHeartbeat(self.manager, self.metrics, self.start_time)
        
        # Transition state and publish start event
        self.state.set(StateType.RUNNING)
        self.event_bus.publish(RuntimeStartedEvent())
        logger.info("StonksRuntime: Engine successfully started and waiting for events.")
        
    def stop(self) -> None:
        """Stops scheduler, worker threads, and publishes stop event."""
        if not self.state.is_active:
            return
            
        self.state.set(StateType.SHUTTING_DOWN)
        logger.info("StonksRuntime: Shuting down operating runtime engine...")
        
        # Publish Stopped Event
        self.event_bus.publish(RuntimeStoppedEvent())
        
        # Stop Scheduler and Worker Threads
        self.scheduler.stop()
        self.worker_pool.shutdown()
        
        self.state.set(StateType.STOPPED)
        logger.info("StonksRuntime: Stopped successfully.")
        
    def _init_recommendation_cache(self) -> None:
        """Initializes last recommendation dictionary using decision history records."""
        self._last_recs.clear()
        history = self.manager.get_recent_decisions(limit=100)
        for record in history:
            self._last_recs[record.ticker] = record.recommendation
            
    def _run_watchlist_sweep(self) -> None:
        """Worker task analyzing all tracked watchlists and positions tickers."""
        tickers = set()
        
        # Fetch watchlists
        for wl in self.manager.get_watchlists().values():
            tickers.update(wl.items.keys())
            
        # Fetch active positions
        for ticker, pos in self.manager.get_positions().items():
            if pos.status.value != "CLOSED":
                tickers.add(ticker)
                
        if not tickers:
            return
            
        logger.info(f"Runtime Sweep: Running background evaluation for tickers: {sorted(list(tickers))}")
        
        from src.agent.pipeline import trading_agent
        for ticker in sorted(list(tickers)):
            try:
                start_t = time.time()
                res = trading_agent.run_pipeline(ticker, force_train=False)
                duration = time.time() - start_t
                
                self.metrics.record_value("analysis_time", duration)
                self.metrics.increment("events_processed")
                
                # Fetch output metrics
                intel = res["intelligence"]["json_report"]
                price = float(intel.get("price", 0.0))
                if price > 0:
                    self.event_bus.publish(PriceUpdateEvent(data={"ticker": ticker, "price": price}))
                    
                new_rec = intel.get("recommendation", "HOLD")
                old_rec = self._last_recs.get(ticker, "HOLD")
                
                if new_rec != old_rec:
                    self._last_recs[ticker] = new_rec
                    self.event_bus.publish(RecommendationChangedEvent(data={
                        "ticker": ticker,
                        "old_recommendation": old_rec,
                        "new_recommendation": new_rec
                    }))
                    
            except Exception as e:
                logger.error(f"Runtime Sweep: Failed to analyze {ticker}: {e}")
                
        # Trigger portfolio metrics change check
        self.manager.update_portfolio_metrics()
        self.event_bus.publish(PortfolioChangedEvent())
        
    def _run_session_autosave(self) -> None:
        """Autosave task writing in-memory mutations to disk."""
        logger.info("Runtime: Auto-saving active trading session...")
        try:
            self.manager.save_session()
            self.event_bus.publish(SessionSavedEvent())
        except Exception as e:
            logger.error(f"Runtime: Autosave failed: {e}")
            
    def _run_heartbeat(self) -> None:
        """Periodic heartbeat logger task."""
        if self.heartbeat:
            self.heartbeat.log_heartbeat()
            # Record current queue depth metric
            self.metrics.record_value("queue_length", self.worker_pool.queue.qsize())
