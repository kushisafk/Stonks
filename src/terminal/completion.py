import sys
from typing import List, Optional, Dict

try:
    import readline
except ImportError:
    readline = None

class TerminalCompleter:
    """Readline-compatible completion helper mapping namespaces, subcommands, and tickers dynamically."""
    
    def __init__(self, manager=None):
        self.manager = manager
        
        # Subcommand trees
        self.tree = {
            "market": ["analyze", "compare", "research", "news", "inspect", "explain", "chart"],
            "position": ["list", "open long", "open short", "close", "reduce", "increase", "review", "update-stop", "update-target"],
            "portfolio": ["summary", "exposure", "performance", "sectors", "risk", "history"],
            "watchlist": ["list", "create", "delete", "add", "remove", "rename"],
            "research": ["benchmark", "thresholds", "features", "models", "importance", "history"],
            "session": ["status", "save", "reload", "reset"],
            "profile": ["show", "edit", "risk", "capital", "preferences"],
            "alerts": ["list", "clear", "acknowledge"]
        }
        self.namespaces = list(self.tree.keys()) + ["help", "version", "clear", "exit", "quit"]
        
    def complete(self, text: str, state: int) -> Optional[str]:
        """
        Completes CLI input tokens based on namespace and watchlist context.
        Readline standard signature.
        """
        if not readline:
            return None
            
        buffer = readline.get_line_buffer()
        words = buffer.split()
        
        # Candidates list to select from based on tab context
        candidates = []
        
        if not buffer or buffer.endswith(" "):
            # If the user typed space, show what matches the next command depth
            if len(words) == 1 and words[0] in self.tree:
                candidates = self.tree[words[0]]
            elif len(words) == 2 and words[0] == "position" and words[1] == "open":
                candidates = ["long", "short"]
            elif len(words) >= 2 and words[0] in ("market", "position", "watchlist"):
                # Suggest tickers after subcommand verbs
                candidates = self._get_tickers()
        else:
            # We are completing a partially typed word
            current_word = words[-1]
            if len(words) == 1:
                candidates = [n for n in self.namespaces if n.startswith(current_word.lower())]
            elif len(words) == 2:
                ns = words[0].lower()
                if ns in self.tree:
                    candidates = [sub for sub in self.tree[ns] if sub.startswith(current_word.lower())]
            elif len(words) == 3 and words[0].lower() == "position" and words[1].lower() == "open":
                candidates = [x for x in ["long", "short"] if x.startswith(current_word.lower())]
            elif len(words) >= 2:
                # Complete ticker symbols
                candidates = [t for t in self._get_tickers() if t.startswith(current_word.upper())]
                
        if state < len(candidates):
            # Readline appends a space when completion succeeds
            return candidates[state]
        return None
        
    def _get_tickers(self) -> List[str]:
        """Queries the session manager for tracked symbols to complete."""
        tickers = set()
        if not self.manager or not self.manager.state:
            return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            
        for wl in self.manager.state.watchlists.values():
            tickers.update(wl.items.keys())
            
        tickers.update(self.manager.state.positions.keys())
        
        if not tickers:
            return ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            
        return sorted(list(tickers))
