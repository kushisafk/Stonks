import threading
from queue import PriorityQueue, Empty
from typing import List, Callable, Set, Optional, Dict
from stonks.runtime.job_queue import PrioritizedJob
from stonks.runtime.exceptions import WorkerPoolError
from stonks.logging.logger import logger

class WorkerPool:
    """ThreadPool container executing concurrent jobs with fault tolerance and retries."""
    
    def __init__(self, num_workers: int = 4, metrics=None):
        self.num_workers = num_workers
        self.metrics = metrics
        self.queue: PriorityQueue = PriorityQueue()
        self.threads: List[threading.Thread] = []
        self._is_running = False
        
        # Thread safety locks for cancelled job trackers
        self._cancelled_jobs: Set[str] = set()
        self._cancelled_lock = threading.Lock()
        
    def start(self) -> None:
        """Starts the background worker threads."""
        if self._is_running:
            return
        self._is_running = True
        self.threads.clear()
        
        for idx in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_loop, 
                name=f"StonksWorker-{idx}", 
                daemon=True
            )
            self.threads.append(t)
            t.start()
            
        logger.info(f"WorkerPool: Started {self.num_workers} worker threads.")
        
    def submit(self, job: PrioritizedJob) -> None:
        """Submits a job to the priority queue."""
        if not self._is_running:
            raise WorkerPoolError("WorkerPool is not running.")
        self.queue.put(job)
        if self.metrics:
            self.metrics.increment("events_published") # event trigger tracker
            
    def cancel_job(self, job_id: str) -> None:
        """Marks a job ID as cancelled so it is skipped when popped."""
        with self._cancelled_lock:
            self._cancelled_jobs.add(job_id)
            
    def shutdown(self) -> None:
        """Shuts down all worker threads gracefully by injecting termination sentinels."""
        if not self._is_running:
            return
        self._is_running = False
        
        # Inject shutdown job sentinels to wake up sleeping threads
        for _ in range(self.num_workers):
            self.queue.put(PrioritizedJob(func=lambda: None, priority=999999, job_id="SHUTDOWN_SENTINEL"))
            
        for t in self.threads:
            t.join(timeout=3.0)
            
        logger.info("WorkerPool: Shutdown complete.")
        
    def _worker_loop(self) -> None:
        """Continuous execution loop pulled by each worker thread."""
        while True:
            try:
                # Wait for next item
                job = self.queue.get(block=True, timeout=1.0)
                
                # Check for shutdown sentinel
                if job.job_id == "SHUTDOWN_SENTINEL":
                    self.queue.task_done()
                    break
                    
                # Check for cancellation
                with self._cancelled_lock:
                    if job.job_id in self._cancelled_jobs:
                        self._cancelled_jobs.remove(job.job_id)
                        self.queue.task_done()
                        continue
                        
                # Execute the job with exception isolation
                self._execute_job(job)
                self.queue.task_done()
                
            except Empty:
                # Loop round to check if running state changed
                if not self._is_running:
                    break
            except Exception as e:
                logger.error(f"WorkerPool: Critical error in worker loop: {e}")
                
    def _execute_job(self, job: PrioritizedJob) -> None:
        """Executes a single job, tracking metrics and retrying if necessary."""
        try:
            # Run target callable function
            job.func(*job.args, **job.kwargs)
            if self.metrics:
                self.metrics.increment("jobs_executed")
                
        except Exception as e:
            logger.error(f"WorkerPool: Job {job.job_id} failed: {e}")
            if self.metrics:
                self.metrics.increment("jobs_failed")
                
            # Perform automatic retry logic
            if job.retries < job.max_retries:
                job.retries += 1
                logger.info(f"WorkerPool: Retrying job {job.job_id} (Attempt {job.retries}/{job.max_retries})")
                self.queue.put(job)
            else:
                logger.error(f"WorkerPool: Job {job.job_id} exceeded maximum retries ({job.max_retries}). Aborting.")
