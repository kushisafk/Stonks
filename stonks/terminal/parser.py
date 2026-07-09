import shlex
from typing import List
from stonks.terminal.errors import ParserError

class CommandParser:
    """Parses raw text strings into command tokens, handling quotes, flags, and namespaces."""
    
    ALIASES = {
        "p": "position",
        "w": "watchlist",
        "port": "portfolio",
        "m": "market",
        "h": "help",
        "q": "exit",
        "r": "research",
        "s": "session",
        "a": "alerts"
    }
    
    def parse(self, text: str) -> List[str]:
        """
        Splits command string into tokens, resolving namespace aliases.
        
        Args:
            text: Raw input command string
            
        Returns:
            List[str]: Split and normalized arguments
        """
        text = text.strip()
        if not text:
            return []
        try:
            # shlex automatically strips quotes and handles escaped spaces
            tokens = shlex.split(text)
        except ValueError as e:
            raise ParserError(f"Syntax error: {e}")
            
        if tokens and tokens[0].lower() in self.ALIASES:
            tokens[0] = self.ALIASES[tokens[0].lower()]
            
        return tokens
