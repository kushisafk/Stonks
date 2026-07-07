import os
from typing import List
from src.terminal.errors import UsageError, CommandError
from src.terminal.formatter import TextFormatter

class SessionCommands:
    """Handles the 'session' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: session <subcommand> [args]\nSubcommands: status, save, reload, reset")
            
        subcmd = args[0].lower()
        
        if subcmd == "status":
            state = self.manager.state
            path = self.manager.persistence.filepath
            size_kb = os.path.getsize(path) / 1024.0 if path.exists() else 0.0
            
            lines = [
                f"Session File Path : {path}",
                f"File Size         : {size_kb:.2f} KB",
                f"Schema Version    : v{state.schema_version}",
                f"Active User       : {state.user_profile.username}",
                f"Active Watchlists : {len(state.watchlists)} list(s)",
                f"Active Positions  : {len([t for t, p in state.positions.items() if p.status.value != 'CLOSED'])} open position(s)",
                f"Log History Count : {len(state.history)} decision(s)",
                f"Total Alerts Count: {len(state.alerts)} alert(s)"
            ]
            print(TextFormatter.to_panel("Trading Session Status", lines))
            
        elif subcmd == "save":
            print("Forcing atomic save of session state...")
            try:
                self.manager.save_session()
                print(f"{TextFormatter.SUCCESS} Session state saved successfully.")
            except Exception as e:
                raise CommandError(f"Failed to save session: {e}")
                
        elif subcmd == "reload":
            print("Reloading session state from disk...")
            try:
                self.manager.load_session()
                print(f"{TextFormatter.SUCCESS} Session state reloaded successfully.")
            except Exception as e:
                raise CommandError(f"Failed to reload session: {e}")
                
        elif subcmd == "reset":
            confirm = input("Are you absolutely sure you want to RESET the entire session state? (yes/no): ").strip().lower()
            if confirm == "yes":
                print("Resetting session state to clean defaults...")
                try:
                    self.manager.create_session()
                    print(f"{TextFormatter.SUCCESS} Session state has been reset successfully.")
                except Exception as e:
                    raise CommandError(f"Failed to reset session: {e}")
            else:
                print("Reset aborted.")
                
        else:
            raise CommandError(f"Unknown session subcommand '{subcmd}'. Type 'help session' for options.")
