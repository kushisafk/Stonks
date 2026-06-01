import logging
from typing import List, Optional
from apscheduler.schedulers.background import BackgroundScheduler
from src.config.settings import settings
from src.logging.logger import logger
from src.agent.pipeline import trading_agent

class TradingScheduler:
    """Manages background scheduled executions of the trading agent pipeline for all configured tickers."""
    
    def __init__(self, tickers: Optional[List[str]] = None, interval_hours: Optional[int] = None):
        self.scheduler = BackgroundScheduler()
        self.tickers = tickers or settings.DEFAULT_TICKERS
        self.interval_hours = interval_hours if interval_hours is not None else settings.SCHEDULER_INTERVAL_HOURS
        self._is_running = False
        
    def _execute_pipeline_job(self) -> None:
        """Scheduled task that runs the coordinated pipeline for each ticker."""
        logger.info(f"Scheduled Job Triggered: Running pipeline for tickers: {self.tickers}")
        for symbol in self.tickers:
            try:
                # Do not force train during recurring scheduled prediction runs (uses cached models)
                trading_agent.run_pipeline(symbol, force_train=False)
            except Exception as e:
                logger.error(f"Scheduled Job Error: Pipeline execution failed for {symbol}: {e}")
                
    def start(self) -> None:
        """Starts the background scheduler thread."""
        if self._is_running:
            logger.warning("TradingScheduler is already running.")
            return
            
        logger.info(f"Starting TradingScheduler: execution interval set to {self.interval_hours} hours.")
        # Schedule the recurring job
        self.scheduler.add_job(
            func=self._execute_pipeline_job,
            trigger="interval",
            hours=self.interval_hours,
            id="stonks_pipeline_daily_job",
            replace_existing=True
        )
        self.scheduler.start()
        self._is_running = True
        logger.info("TradingScheduler started successfully.")
        
    def stop(self) -> None:
        """Gracefully terminates the background scheduler thread."""
        if not self._is_running:
            logger.warning("TradingScheduler is not running.")
            return
            
        logger.info("Stopping TradingScheduler...")
        self.scheduler.shutdown(wait=True)
        self._is_running = False
        logger.info("TradingScheduler stopped successfully.")
        
    @property
    def is_running(self) -> bool:
        """Returns True if the scheduler background threads are active."""
        return self._is_running

# Global scheduler instance
trading_scheduler = TradingScheduler()
