import pytest
import json
from pathlib import Path
from stonks.session.manager import TradingSessionManager
from stonks.session.schemas import TradingStyle, RiskProfile, DecisionRecord, Alert
from stonks.session.exceptions import InvalidStateError, PositionNotFoundError, WatchlistNotFoundError

def test_session_lifecycle_and_persistence(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    
    # 1. Create session
    manager.create_session()
    assert manager.state is not None
    assert session_file.exists()
    
    # 2. Update preferences
    manager.update_preferences(theme="light", preferred_language="fr")
    manager.update_profile(username="Alice", trading_style=TradingStyle.SCALPER)
    
    # 3. Reload session
    manager2 = TradingSessionManager(session_file)
    manager2.load_session()
    assert manager2.state.user_profile.username == "Alice"
    assert manager2.state.user_profile.trading_style == TradingStyle.SCALPER
    assert manager2.state.preferences.theme == "light"
    assert manager2.state.preferences.preferred_language == "fr"

def test_watchlist_operations(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    manager.create_session()
    
    # Add watchlist
    manager.add_watchlist("Growth")
    assert "Growth" in manager.get_watchlists()
    
    # Track symbols
    manager.track_symbol(
        watchlist_name="Growth",
        ticker="AAPL",
        tags=["tech", "megacap"],
        notes="Earnings breakout candidate",
        priority=3,
        target_price=200.0
    )
    watchlist = manager.get_watchlists()["Growth"]
    assert "AAPL" in watchlist.items
    assert watchlist.items["AAPL"].notes == "Earnings breakout candidate"
    assert watchlist.items["AAPL"].target_price == 200.0
    
    # Rename watchlist
    manager.rename_watchlist("Growth", "High Growth")
    assert "High Growth" in manager.get_watchlists()
    assert "Growth" not in manager.get_watchlists()
    assert "AAPL" in manager.get_watchlists()["High Growth"].items
    
    # Untrack symbol
    manager.untrack_symbol("High Growth", "AAPL")
    assert "AAPL" not in manager.get_watchlists()["High Growth"].items

def test_positions_and_portfolio_auto_math(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    manager.create_session()
    manager.set_cash_balance(100000.0)
    
    port = manager.get_portfolio()
    assert port.cash_balance == 100000.0
    assert port.total_equity == 100000.0
    
    # 1. Open Long Position: BUY 100 AAPL at $150
    manager.open_long_position(ticker="AAPL", entry_price=150.0, quantity=100.0)
    assert "AAPL" in manager.get_positions()
    
    port = manager.get_portfolio()
    assert port.cash_balance == 85000.0
    assert port.long_exposure == 15000.0
    assert port.total_equity == 100000.0
    
    # 2. Price increase: Update portfolio metrics with AAPL at $160
    manager.update_portfolio_metrics(current_prices={"AAPL": 160.0})
    port = manager.get_portfolio()
    assert port.cash_balance == 85000.0
    assert port.long_exposure == 16000.0
    assert port.unrealized_pl == 1000.0
    assert port.total_equity == 101000.0
    
    # 3. Partial Close: Sell 50 shares at $170
    manager.partial_close_position(ticker="AAPL", exit_price=170.0, quantity=50.0)
    port = manager.get_portfolio()
    assert port.cash_balance == 93500.0
    assert port.realized_pl == 1000.0
    
    manager.update_portfolio_metrics(current_prices={"AAPL": 170.0})
    port = manager.get_portfolio()
    assert port.cash_balance == 93500.0
    assert port.long_exposure == 8500.0
    assert port.unrealized_pl == 1000.0
    assert port.total_equity == 102000.0
    
    # 4. Enforce Budget constraints
    with pytest.raises(InvalidStateError):
        manager.open_long_position(ticker="MSFT", entry_price=1000.0, quantity=500.0)
        
    # 5. Open Short Position: Sell 10 shares of TSLA at $200
    manager.open_short_position(ticker="TSLA", entry_price=200.0, quantity=10.0)
    port = manager.get_portfolio()
    assert port.cash_balance == 95500.0
    assert port.short_exposure == 2000.0
    
    # Update price of TSLA to $180 (profit for short)
    manager.update_portfolio_metrics(current_prices={"AAPL": 170.0, "TSLA": 180.0})
    port = manager.get_portfolio()
    assert port.unrealized_pl == 1200.0
    assert port.short_exposure == 1800.0
    assert port.total_equity == 95500.0 + 8500.0 - 1800.0
    
    # Close short position at $190
    manager.close_position(ticker="TSLA", exit_price=190.0)
    port = manager.get_portfolio()
    assert port.cash_balance == 93600.0
    assert port.realized_pl == 1100.0

def test_decision_history_and_alerts(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    manager.create_session()
    
    # Record Decision
    record = DecisionRecord(
        timestamp="2026-06-28T16:00:00",
        ticker="NVDA",
        prediction="BUY",
        probability=0.74,
        risk_score=45,
        confidence_tier="High",
        recommendation="BUY",
        reasoning="Strong technicals",
        warnings=["High volatility"],
        market_regime="Bullish",
        sentiment=0.25,
        model_used="catboost"
    )
    manager.record_decision(record)
    assert len(manager.get_recent_decisions()) == 1
    assert manager.get_ticker_history("NVDA")[0].prediction == "BUY"
    assert len(manager.get_recommendation_history("BUY")) == 1
    
    # Record Alert
    alert = Alert(
        timestamp="2026-06-28T16:05:00",
        rule_type="Stop Loss Trigger",
        ticker="AAPL",
        message="Stop loss triggered at $145.00"
    )
    manager.record_alert(alert)
    assert len(manager.get_alerts()) == 1
    assert manager.get_alerts()[0].ticker == "AAPL"
    
    manager.clear_alerts()
    assert len(manager.get_alerts()) == 0

def test_corrupted_file_recovery(tmp_path):
    session_file = tmp_path / "session.json"
    manager = TradingSessionManager(session_file)
    manager.create_session()
    
    # Mutate state so we know it's not clean
    manager.update_preferences(theme="emerald")
    manager.save_session()
    
    # Corrupt the main session file with garbage data
    with open(session_file, "w") as f:
        f.write("{{INVALID CORRUPT JSON{")
        
    # Attempt to load session. It should recover from session.backup
    manager_recover = TradingSessionManager(session_file)
    manager_recover.load_session()
    assert manager_recover.state.preferences.theme == "emerald"
    
    # Corrupt both main and backup files
    with open(session_file, "w") as f:
        f.write("GARBAGE")
    with open(manager_recover.persistence.backup_path, "w") as f:
        f.write("GARBAGE")
        
    # Attempt to load should initialize clean state instead of crashing
    manager_clean = TradingSessionManager(session_file)
    manager_clean.load_session()
    assert manager_clean.state.preferences.theme == "dark" # Default theme
