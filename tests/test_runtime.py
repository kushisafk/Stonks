import time
import pytest
from src.session.manager import TradingSessionManager
from src.runtime.runtime import StonksRuntime
from src.runtime.events import Event, PriceUpdateEvent, RecommendationChangedEvent, PortfolioChangedEvent
from src.runtime.job_queue import PrioritizedJob
from src.runtime.exceptions import WorkerPoolError, AgentRegistrationError
from src.session.schemas import Position, PositionType, PositionStatus

class CustomTestEvent(Event):
    pass

def test_event_bus():
    from src.runtime.event_bus import EventBus
    bus = EventBus()
    
    received = []
    def listener(event):
        received.append(event.data.get("val"))
        
    # Subscribe to custom event
    bus.subscribe(CustomTestEvent, listener)
    
    # Publish event
    bus.publish(CustomTestEvent(data={"val": 123}))
    assert len(received) == 1
    assert received[0] == 123
    
    # Unsubscribe
    bus.unsubscribe(CustomTestEvent, listener)
    bus.publish(CustomTestEvent(data={"val": 456}))
    assert len(received) == 1  # Unsubscribed, so not received

def test_worker_pool_priorities_and_fault_isolation():
    from src.runtime.worker_pool import WorkerPool
    from src.runtime.metrics import RuntimeMetrics
    
    metrics = RuntimeMetrics()
    pool = WorkerPool(num_workers=2, metrics=metrics)
    pool.start()
    
    execution_order = []
    
    # Define tasks with different priorities
    def task_a():
        execution_order.append("A")
        
    def task_b():
        execution_order.append("B")
        
    def task_crashing():
        raise ValueError("Simulated crash")
        
    # Submit crashing task and normal tasks
    pool.submit(PrioritizedJob(func=task_crashing, priority=1, max_retries=0))
    pool.submit(PrioritizedJob(func=task_b, priority=3))
    pool.submit(PrioritizedJob(func=task_a, priority=2))
    
    # Wait for execution
    time.sleep(0.5)
    pool.shutdown()
    
    # Crash isolation verification
    assert metrics.get_all()["jobs_failed"] == 1
    assert "A" in execution_order
    assert "B" in execution_order

def test_scheduler_loop():
    from src.runtime.worker_pool import WorkerPool
    from src.runtime.scheduler import RuntimeScheduler
    
    pool = WorkerPool(num_workers=1)
    pool.start()
    
    scheduler = RuntimeScheduler(pool)
    scheduler.start()
    
    counter = []
    def increment():
        counter.append(1)
        
    # Add a repeating job every 0.1 seconds
    scheduler.add_job(func=increment, interval_seconds=0.1, one_shot=False)
    
    time.sleep(0.35)
    scheduler.stop()
    pool.shutdown()
    
    # Repeating job should have run at least 2-3 times
    assert len(counter) >= 2

def test_dispatcher_alerts_generation(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    manager.create_session()
    
    # Mock active position for AAPL with a Stop Loss at $150
    # Deduct cash to simulate entry
    manager.state.portfolio.cash_balance = 50000.0
    manager.state.portfolio.buying_power = 50000.0
    
    manager.open_long_position(ticker="AAPL", entry_price=160.0, quantity=100.0, stop_loss=150.0)
    assert "AAPL" in manager.get_positions()
    
    # Start runtime facade to hook events
    runtime = StonksRuntime(manager)
    runtime.start()
    
    # Publish price update breaching stop loss ($145)
    runtime.event_bus.publish(PriceUpdateEvent(data={"ticker": "AAPL", "price": 145.0}))
    
    # Wait for event dispatcher callbacks to process inside worker pool
    time.sleep(0.5)
    runtime.stop()
    
    # Alert should have been generated and position closed
    alerts = manager.get_alerts()
    assert len(alerts) >= 1
    assert alerts[0].rule_type == "Stop Loss Triggered"
    assert manager.get_positions()["AAPL"].status == PositionStatus.CLOSED
