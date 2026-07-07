import os
import sys
from pathlib import Path
from typing import Dict, Any

try:
    import readline
except ImportError:
    readline = None

from src.terminal.parser import CommandParser
from src.terminal.completion import TerminalCompleter
from src.terminal.formatter import TextFormatter
from src.terminal.errors import TerminalError, CommandError, UsageError
from src.terminal.commands.market import MarketCommands
from src.terminal.commands.positions import PositionCommands
from src.terminal.commands.portfolio import PortfolioCommands
from src.terminal.commands.watchlists import WatchlistCommands
from src.terminal.commands.research import ResearchCommands
from src.terminal.commands.profile import ProfileCommands
from src.terminal.commands.session import SessionCommands
from src.terminal.commands.alerts import AlertCommands
from src.terminal.commands.system import SystemCommands
from src.terminal.commands.runtime_cmd import RuntimeCommands
from src.runtime.runtime import StonksRuntime

class TerminalShell:
    """Orchestrates the REPL shell loop, manages command history, and routes user inputs to subcommands."""
    
    def __init__(self, manager):
        self.manager = manager
        self.parser = CommandParser()
        
        # Instantiate continuous background runtime
        self.runtime = StonksRuntime(manager)
        self.completer = TerminalCompleter(manager)
        
        # Instantiate namespace command handlers
        self.market_cmd = MarketCommands(manager)
        self.position_cmd = PositionCommands(manager)
        self.portfolio_cmd = PortfolioCommands(manager)
        self.watchlist_cmd = WatchlistCommands(manager)
        self.research_cmd = ResearchCommands(manager)
        self.profile_cmd = ProfileCommands(manager)
        self.session_cmd = SessionCommands(manager)
        self.alerts_cmd = AlertCommands(manager)
        self.system_cmd = SystemCommands(manager)
        self.runtime_cmd = RuntimeCommands(manager, self.runtime)
        
        # History setup
        self.history_file = Path(".stonks_history").resolve()
        
    def setup_readline(self) -> None:
        """Configures history persistence and autocomplete completer hooks."""
        if not readline:
            return
            
        # Hook autocomplete
        readline.set_completer(self.completer.complete)
        readline.parse_and_bind("tab: complete")
        # Treat spaces, tabs, quotes properly
        readline.set_completer_delims(" \t\n`@$><=;|&(")
        
        # Load history
        if self.history_file.exists():
            try:
                readline.read_history_file(str(self.history_file))
            except Exception:
                pass
                
    def save_history(self) -> None:
        """Saves command history to disk."""
        if not readline:
            return
        try:
            # Set history size to avoid unlimited growth
            readline.set_history_length(1000)
            readline.write_history_file(str(self.history_file))
        except Exception:
            pass
            
    def run(self) -> None:
        """Starts the interactive CLI loop."""
        self.setup_readline()
        
        print("\nType \"help\" to begin.\n")
        
        while True:
            try:
                # Prompt with color
                prompt = TextFormatter.colorize("stonks> ", TextFormatter.BOLD + TextFormatter.CYAN)
                line = input(prompt)
                
                # Parse command line tokens
                tokens = self.parser.parse(line)
                if not tokens:
                    continue
                    
                cmd = tokens[0].lower()
                args = tokens[1:]
                
                # Handle Exit commands directly
                if cmd in ("exit", "quit"):
                    print("Exiting STONKS terminal. Auto-saving active session...")
                    if self.runtime.state.is_active:
                        self.runtime.stop()
                    self.manager.save_session()
                    self.save_history()
                    break
                    
                # Route commands by namespace
                if cmd == "market":
                    self.market_cmd.execute(args)
                elif cmd == "position":
                    self.position_cmd.execute(args)
                elif cmd == "portfolio":
                    self.portfolio_cmd.execute(args)
                elif cmd == "watchlist":
                    self.watchlist_cmd.execute(args)
                elif cmd == "research":
                    self.research_cmd.execute(args)
                elif cmd == "profile":
                    self.profile_cmd.execute(args)
                elif cmd == "session":
                    self.session_cmd.execute(args)
                elif cmd == "alerts":
                    self.alerts_cmd.execute(args)
                elif cmd == "runtime":
                    self.runtime_cmd.execute(args)
                elif cmd in ("help", "version", "clear"):
                    self.system_cmd.execute(cmd, args)
                else:
                    # Search for closest namespace command suggestion
                    suggestions = [
                        n for n in self.completer.namespaces 
                        if n.startswith(cmd[:2]) or cmd[:2] in n
                    ]
                    sug_str = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                    print(f"{TextFormatter.WARNING} Unknown command namespace '{cmd}'.{sug_str}")
                    
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\n[Operation cancelled. Type 'exit' to quit.]")
            except EOFError:
                # Handle Ctrl+D gracefully
                print("\nExiting STONKS terminal. Auto-saving active session...")
                if self.runtime.state.is_active:
                    self.runtime.stop()
                self.manager.save_session()
                self.save_history()
                break
            except TerminalError as e:
                # Handle CLI command errors gracefully without crashing the shell loop
                print(f"{TextFormatter.WARNING} {e}")
            except Exception as e:
                # Safety fallback to prevent crashes from bugs or data corruption
                print(f"{TextFormatter.ALERT} System Error: {e}")
