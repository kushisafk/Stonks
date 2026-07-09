from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter
from stonks.session.exceptions import SessionError

class WatchlistCommands:
    """Handles the 'watchlist' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: watchlist <subcommand> [args]\nSubcommands: list, create, delete, add, remove, rename")
            
        subcmd = args[0].lower()
        
        if subcmd == "list":
            watchlists = self.manager.get_watchlists()
            if not watchlists:
                print("No watchlists defined.")
                return
                
            for name, wl in watchlists.items():
                print(f"\nWatchlist: {TextFormatter.bold(name)}")
                if not wl.items:
                    print("  (Empty)")
                    continue
                    
                headers = ["Ticker", "Date Added", "Notes", "Priority", "Target Price"]
                rows = []
                for ticker, item in wl.items.items():
                    rows.append([
                        ticker,
                        item.date_added.split("T")[0] if "T" in item.date_added else item.date_added,
                        item.notes or "-",
                        str(item.priority),
                        f"${item.target_price:.2f}" if item.target_price else "-"
                    ])
                print(TextFormatter.to_table(headers, rows))
                
        elif subcmd == "create":
            if len(args) < 2:
                raise UsageError("Usage: watchlist create <name>")
            name = args[1]
            try:
                self.manager.add_watchlist(name)
                print(f"{TextFormatter.SUCCESS} Created watchlist '{name}' successfully.")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "delete":
            if len(args) < 2:
                raise UsageError("Usage: watchlist delete <name>")
            name = args[1]
            try:
                # Direct deletion of named watchlist dict block
                self.manager.state.watchlists.pop(name)
                self.manager.save_session()
                print(f"{TextFormatter.SUCCESS} Deleted watchlist '{name}'.")
            except KeyError:
                raise CommandError(f"Watchlist '{name}' not found.")
                
        elif subcmd == "add":
            if len(args) < 3:
                raise UsageError("Usage: watchlist add <name> <ticker> [notes] [priority] [target_price]")
            name = args[1]
            ticker = args[2].upper()
            
            notes = args[3] if len(args) >= 4 else ""
            priority = int(args[4]) if len(args) >= 5 else 2
            target_price = float(args[5]) if len(args) >= 6 else None
            
            try:
                self.manager.track_symbol(name, ticker, tags=[], notes=notes, priority=priority, target_price=target_price)
                print(f"{TextFormatter.SUCCESS} Added {ticker} to watchlist '{name}'.")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "remove":
            if len(args) < 3:
                raise UsageError("Usage: watchlist remove <name> <ticker>")
            name = args[1]
            ticker = args[2].upper()
            try:
                self.manager.untrack_symbol(name, ticker)
                print(f"{TextFormatter.SUCCESS} Removed {ticker} from watchlist '{name}'.")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "rename":
            if len(args) < 3:
                raise UsageError("Usage: watchlist rename <old_name> <new_name>")
            old_name, new_name = args[1], args[2]
            try:
                self.manager.rename_watchlist(old_name, new_name)
                print(f"{TextFormatter.SUCCESS} Renamed watchlist '{old_name}' to '{new_name}'.")
            except SessionError as e:
                raise CommandError(str(e))
                
        else:
            raise CommandError(f"Unknown watchlist subcommand '{subcmd}'. Type 'help watchlist' for options.")
