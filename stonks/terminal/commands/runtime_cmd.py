from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter

class RuntimeCommands:
    """Handles the 'runtime' command namespace actions."""
    
    def __init__(self, manager, runtime):
        self.manager = manager
        self.runtime = runtime
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: runtime <subcommand> [args]\nSubcommands: start, stop, restart, status, metrics, jobs, events, heartbeat, config")
            
        subcmd = args[0].lower()
        
        if subcmd == "start":
            if self.runtime.state.is_active:
                print("Runtime is already running.")
                return
            print("Starting background operating runtime engine...")
            self.runtime.start()
            print(f"{TextFormatter.SUCCESS} Background runtime started successfully.")
            
        elif subcmd == "stop":
            if not self.runtime.state.is_active:
                print("Runtime is already stopped.")
                return
            print("Stopping background operating runtime engine...")
            self.runtime.stop()
            print(f"{TextFormatter.SUCCESS} Background runtime stopped.")
            
        elif subcmd == "restart":
            print("Restarting background operating runtime engine...")
            self.runtime.stop()
            self.runtime.start()
            print(f"{TextFormatter.SUCCESS} Background runtime restarted.")
            
        elif subcmd == "status":
            state_val = self.runtime.state.get().value
            uptime_sec = 0
            if self.runtime.state.is_active and self.runtime.heartbeat:
                payload = self.runtime.heartbeat.generate_heartbeat_payload()
                uptime_sec = payload["uptime_seconds"]
                
            lines = [
                f"Runtime State  : {state_val}",
                f"Uptime         : {uptime_sec}s",
                f"Worker Threads : {self.runtime.worker_pool.num_workers}",
                f"Queued Jobs    : {self.runtime.worker_pool.queue.qsize()}",
                f"Active Agents  : {len(self.runtime.agent_manager.list_agents())}"
            ]
            print(TextFormatter.to_panel("Runtime Status", lines))
            
        elif subcmd == "metrics":
            m = self.runtime.metrics.get_all()
            headers = ["Metric Name", "Value"]
            rows = [
                ["Jobs Executed", str(m.get("jobs_executed", 0))],
                ["Jobs Failed", str(m.get("jobs_failed", 0))],
                ["Events Published", str(m.get("events_published", 0))],
                ["Events Processed", str(m.get("events_processed", 0))],
                ["Average Analysis Latency", f"{m.get('analysis_time', 0.0):.2f}s"],
                ["Queue Depth", str(m.get("queue_length", 0))]
            ]
            print(f"\n{TextFormatter.bold('Runtime Health Metrics')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "jobs":
            # Lists scheduled jobs
            jobs = self.runtime.scheduler._jobs
            if not jobs:
                print("No scheduled jobs registered.")
                return
                
            headers = ["Task Name", "Interval", "Priority", "Next Run In"]
            import time
            now = time.time()
            rows = []
            for j in jobs:
                time_rem = max(0.0, j.next_run - now)
                rows.append([
                    j.func.__name__,
                    f"{j.interval}s",
                    str(j.priority),
                    f"{time_rem:.1f}s"
                ])
            print(f"\n{TextFormatter.bold('Scheduler Active Task Queue')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "events":
            # List event listeners
            listeners = self.runtime.event_bus._listeners
            headers = ["Event Type", "Subscriber Callables Count"]
            rows = []
            for k, v in listeners.items():
                rows.append([k.__name__, str(len(v))])
            print(f"\n{TextFormatter.bold('Event Bus Active Subscriptions')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "heartbeat":
            if not self.runtime.heartbeat:
                print("Heartbeat manager is not initialized (start runtime first).")
                return
            payload = self.runtime.heartbeat.generate_heartbeat_payload()
            lines = [
                f"Timestamp   : {payload['timestamp']}",
                f"Memory RSS  : {payload['memory_usage_mb']} MB",
                f"Jobs run    : {payload['jobs_executed']} OK / {payload['jobs_failed']} Fail",
                f"Queue size  : {payload['queue_length']}",
                f"Agent Health: {payload['agent_health']}"
            ]
            print(TextFormatter.to_panel("System Heartbeat Monitor", lines))
            
        elif subcmd == "config":
            tasks = self.runtime.task_registry.list_all()
            headers = ["Task Name", "Default Interval", "Default Priority"]
            rows = []
            for name, t in tasks.items():
                rows.append([name, f"{t['default_interval']}s", str(t['priority'])])
            print(f"\n{TextFormatter.bold('Runtime Scheduler Config')}")
            print(TextFormatter.to_table(headers, rows))
            
        else:
            raise CommandError(f"Unknown runtime subcommand '{subcmd}'. Type 'help' for options.")
