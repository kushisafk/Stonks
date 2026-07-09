class TerminalError(Exception):
    """Base exception for all CLI terminal issues."""
    pass

class ParserError(TerminalError):
    """Raised when tokens cannot be parsed or quote mismatches occur."""
    pass

class CommandError(TerminalError):
    """Raised when command execution fails due to runtime rules."""
    pass

class UsageError(TerminalError):
    """Raised when syntax parameters or argument types are incorrect."""
    pass
