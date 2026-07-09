import time
import threading
from typing import List, Callable, Optional, Tuple, Dict, Any
from stonks.runtime.job_queue import PrioritizedJob
from stonks.logging.logger import logger

class SchedulerJob:
    """Descriptor encapsulating scheduled job tasks and tracking execution intervals."""
    
    def __init__(
        self, 
        func: Callable, 
        interval_seconds: float, 
        one_shot: bool = False, 
        priority: int = 5, 
        args: Tuple = None, 
        kwargs: Dict[str, Any] = None
    ):
        self.func = func
        self.interval = interval_seconds
        self.one_shot = one_shot
        self.priority = priority
        self.args = args or ()
        self.kwargs = kwargs or {}
        # Delayed one-shots trigger in the future, standard loops execute immediately
        self.next_run = time.time() + (interval_seconds if one_shot else 0.0)

class RuntimeScheduler:
    """Manages scheduled background task triggers, dispatching jobs to the worker pool without busy waiting."""
    
    def __init__(self, worker_pool):
        self.worker_pool = worker_pool
        self._jobs: List[SchedulerJob] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False
        self._condition = threading.Condition()
        
    def add_job(
        self, 
        func: Callable, 
        interval_seconds: float, 
        one_shot: bool = False, 
        priority: int = 5, 
        args: Tuple = None, 
        kwargs: Dict[str, Any] = None
    ) -> None:
        """Registers a job for scheduled executions."""
        job = SchedulerJob(func, interval_seconds, one_shot, priority, args, kwargs)
        with self._lock:
            self._jobs.append(job)
        with self._condition:
            self._condition.notify_all()
            
    def start(self) -> None:
        """Starts the scheduler thread loop."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop, 
            name="StonksScheduler", 
            daemon=True
        )
        self._thread.start()
        logger.info("RuntimeScheduler: Started scheduler thread.")
        
    def stop(self) -> None:
        """Stops the scheduler thread loop."""
        if not self._is_running:
            return
        self._is_running = False
        with self._condition:
            self._condition.notify_all()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("RuntimeScheduler: Scheduler shutdown complete.")
        
    def _scheduler_loop(self) -> None:
        """REPL loop determining nearest execution trigger times and waiting on condition sleeps."""
        while self._is_running:
            now = time.time()
            sleep_time = 1.0  # Default sleep window if no jobs registered
            jobs_to_run = []
            
            with self._lock:
                # Filter due jobs
                for job in self._jobs:
                    if now >= job.next_run:
                        jobs_to_run.append(job)
                        
                # Update next execution times or remove finished one-shots
                for job in jobs_to_run:
                    if job.one_shot:
                        self._jobs.remove(job)
                    else:
                        job.next_run = now + job.interval
                        
                # Calculate sleep duration to the nearest due job
                if self._jobs:
                    next_run_times = [j.next_run for j in self._jobs]
                    sleep_time = max(0.005, min(next_run_times) - now)
                    
            # Submit due jobs to the concurrent worker pool
            for job in jobs_to_run:
                p_job = PrioritizedJob(
                    func=job.func,
                    priority=job.priority,
                    args=job.args,
                    kwargs=job.kwargs
                )
                try:
                    self.worker_pool.submit(p_job)
                except Exception as e:
                    logger.error(f"RuntimeScheduler: Failed to submit job: {e}")
                    
            # Wait on condition to prevent CPU spinning
            with self._condition:
                if self._is_running:
                    self._condition.wait(timeout=sleep_time)
