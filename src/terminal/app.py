import sys
from pathlib import Path
from src.session.manager import TradingSessionManager
from src.terminal.shell import TerminalShell
from src.terminal.formatter import TextFormatter
from src.config.settings import settings

def display_welcome_banner(manager: TradingSessionManager) -> None:
    """Renders the standard ASCII welcome banner showing system status."""
    state = manager.state
    
    # Calculate statistics
    active_model = state.preferences.preferred_ml_model
    watchlists_count = len(state.watchlists)
    positions_count = len([t for t, p in state.positions.items() if p.status.value != "CLOSED"])
    alerts_count = len(state.alerts)
    
    banner = f"""
╔════════════════════════════════════╗
║            STONKS                 ║
║ AI Trading Operating System       ║
╚════════════════════════════════════╝

Workspace : Default
Model     : {active_model.capitalize()}
Watchlists: {watchlists_count}
Positions : {positions_count}
Alerts    : {alerts_count}
"""
    print(banner.strip())

def main() -> None:
    """Configures environment and kicks off shell loop."""
    # Ensure stdout/stderr reconfigured for UTF-8 support on Windows consoles
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
        
    session_file = settings.MODEL_DIR / "session.json"
    
    # Initialize Session Manager facade
    manager = TradingSessionManager(session_file)
    manager.load_session()
    
    display_welcome_banner(manager)
    
    # Launch Terminal REPL
    shell = TerminalShell(manager)
    shell.run()

if __name__ == "__main__":
    main()
