from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter

class AlertCommands:
    """Handles the 'alerts' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: alerts <subcommand> [args]\nSubcommands: list, clear, acknowledge")
            
        subcmd = args[0].lower()
        
        if subcmd == "list":
            # List all logged alerts
            alerts = self.manager.get_alerts(triggered_only=False)
            if not alerts:
                print("No alerts recorded.")
                return
                
            headers = ["Timestamp", "Type", "Ticker", "Message", "Triggered"]
            rows = []
            for a in alerts:
                trig_str = TextFormatter.red("ACTIVE") if a.triggered else TextFormatter.green("ACK")
                rows.append([
                    a.timestamp.split("T")[1][:8] if "T" in a.timestamp else a.timestamp,
                    a.rule_type,
                    a.ticker,
                    a.message,
                    trig_str
                ])
            print(f"\n{TextFormatter.bold('Logged Alerts History')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "clear":
            self.manager.clear_alerts()
            print(f"{TextFormatter.SUCCESS} Alert history cleared.")
            
        elif subcmd in ("acknowledge", "ack"):
            alerts = self.manager.get_alerts(triggered_only=True)
            if not alerts:
                print("No active alerts to acknowledge.")
                return
            for a in alerts:
                a.triggered = False
            self.manager.save_session()
            print(f"{TextFormatter.SUCCESS} Acknowledged {len(alerts)} alert(s).")
            
        else:
            raise CommandError(f"Unknown alerts subcommand '{subcmd}'. Type 'help alerts' for options.")
