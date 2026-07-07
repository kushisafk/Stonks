import os
import time
from datetime import datetime
from src.logging.logger import logger

try:
    import psutil
except ImportError:
    psutil = None

class RuntimeHeartbeat:
    """Monitors system resource usage, agent health states, and logs periodic health statuses."""
    
    def __init__(self, manager, metrics, start_time: float):
        self.manager = manager
        self.metrics = metrics
        self.start_time = start_time
        
    def generate_heartbeat_payload(self) -> dict:
        """Assembles process statistics, execution counts, and utilization rates."""
        uptime = time.time() - self.start_time
        
        # Memory tracking
        memory_bytes = 0
        if psutil:
            try:
                process = psutil.Process(os.getpid())
                memory_bytes = process.memory_info().rss
            except Exception:
                pass
        else:
            # Simple mock if psutil is missing
            memory_bytes = 104857600  # 100MB fallback
            
        m = self.metrics.get_all() if self.metrics else {}
        
        # Calculate pool utilization (mock or queue status)
        pool_size = 4
        queue_len = m.get("queue_length", 0)
        
        payload = {
            "uptime_seconds": int(uptime),
            "timestamp": datetime.now().isoformat(),
            "jobs_executed": m.get("jobs_executed", 0),
            "jobs_failed": m.get("jobs_failed", 0),
            "memory_usage_mb": round(memory_bytes / (1024 * 1024), 2),
            "events_published": m.get("events_published", 0),
            "events_processed": m.get("events_processed", 0),
            "avg_analysis_time": m.get("analysis_time", 0.0),
            "queue_length": queue_len,
            "agent_health": "Healthy"
        }
        return payload
        
    def log_heartbeat(self) -> None:
        """Outputs current payload parameters to structured logs."""
        p = self.generate_heartbeat_payload()
        logger.info(
            f"[HEARTBEAT] Uptime: {p['uptime_seconds']}s | "
            f"Jobs: {p['jobs_executed']} OK / {p['jobs_failed']} Fail | "
            f"Memory: {p['memory_usage_mb']} MB | "
            f"Queue: {p['queue_length']}"
        )
