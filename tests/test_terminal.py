import pytest
from pathlib import Path
from stonks.terminal.parser import CommandParser
from stonks.terminal.completion import TerminalCompleter
from stonks.terminal.errors import ParserError, UsageError, CommandError
from stonks.session.manager import TradingSessionManager

def test_command_parser():
    parser = CommandParser()
    
    # 1. Standard token parsing
    tokens = parser.parse("watchlist create Growth")
    assert tokens == ["watchlist", "create", "Growth"]
    
    # 2. Alias resolution
    tokens_alias = parser.parse("p list")
    assert tokens_alias == ["position", "list"]
    
    # 3. Quoted arguments handling
    tokens_quotes = parser.parse('watchlist add Technology "NVDA" "NVIDIA Corp"')
    assert tokens_quotes == ["watchlist", "add", "Technology", "NVDA", "NVIDIA Corp"]
    
    # 4. Empty parsing
    assert parser.parse("") == []
    
    # 5. Mismatched quotes error
    with pytest.raises(ParserError):
        parser.parse('watchlist add Technology "NVDA')

def test_completer_suggestions():
    completer = TerminalCompleter()
    
    # Set mock readline library binding
    import sys
    class MockReadline:
        def __init__(self):
            self.buffer = ""
        def get_line_buffer(self):
            return self.buffer
            
    import stonks.terminal.completion
    stonks.terminal.completion.readline = MockReadline()
    
    # Mock line buffer
    stonks.terminal.completion.readline.buffer = "mark"
    res = completer.complete("mark", 0)
    assert res == "market"
    
    # Subcommands completion
    stonks.terminal.completion.readline.buffer = "market "
    res = completer.complete("", 0)
    assert res == "analyze"
    
    stonks.terminal.completion.readline = None  # Reset readline state representation

def test_commands_execution_flow(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    manager.create_session()
    
    from stonks.terminal.commands.watchlists import WatchlistCommands
    watchlist_cmd = WatchlistCommands(manager)
    
    # Test watchlist creation execution
    watchlist_cmd.execute(["create", "Dividend"])
    assert "Dividend" in manager.get_watchlists()
    
    # Test validation error (empty watchlist create)
    with pytest.raises(UsageError):
        watchlist_cmd.execute(["create"])
        
    # Test invalid subcommand error
    with pytest.raises(CommandError):
        watchlist_cmd.execute(["invalid_action"])
        
    # Test portfolio cash execution
    from stonks.terminal.commands.portfolio import PortfolioCommands
    portfolio_cmd = PortfolioCommands(manager)
    portfolio_cmd.execute(["cash", "50000.0"])
    assert manager.get_portfolio().cash_balance == 50000.0
