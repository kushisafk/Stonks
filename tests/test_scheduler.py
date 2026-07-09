import pytest
from unittest.mock import patch, MagicMock
from stonks.scheduler.scheduler import TradingScheduler

def test_scheduler_lifecycle():
    """Verify that start and stop lifecycle updates is_running state correctly."""
    scheduler = TradingScheduler(tickers=["MSFT"], interval_hours=1)
    assert not scheduler.is_running
    
    # Start scheduler background threads
    scheduler.start()
    assert scheduler.is_running
    
    # Stop scheduler background threads
    scheduler.stop()
    assert not scheduler.is_running

@patch("stonks.scheduler.scheduler.trading_agent.run_pipeline")
def test_scheduler_job_execution(mock_pipeline):
    """Verify that the scheduled background job triggers run_pipeline for all configured tickers."""
    scheduler = TradingScheduler(tickers=["AAPL", "TSLA"], interval_hours=24)
    
    # Trigger the scheduled function manually to verify iteration
    scheduler._execute_pipeline_job()
    
    # Verify run_pipeline was called once for each ticker
    assert mock_pipeline.call_count == 2
    mock_pipeline.assert_any_call("AAPL", force_train=False)
    mock_pipeline.assert_any_call("TSLA", force_train=False)
