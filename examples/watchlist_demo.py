"""
STONKS Watchlist Management Demo

Demonstrates how to create watchlist folders, track tickers with priority keys 
and target prices, and rename watchlists.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stonks.session.manager import TradingSessionManager

def main():
    session_file = Path("models_data/session.json")
    manager = TradingSessionManager(session_file)
    manager.create_session() # Fresh slate
    
    # 1. Create a watchlist
    watchlist_name = "Growth Picks"
    print(f"Creating watchlist: '{watchlist_name}'...")
    manager.add_watchlist(watchlist_name)
    
    # 2. Track symbols
    print("Tracking symbols: NVDA, TSLA...")
    manager.track_symbol(
        watchlist_name=watchlist_name,
        ticker="NVDA",
        tags=["tech", "semiconductor"],
        notes="High volume consolidation",
        priority=3,
        target_price=135.00
    )
    manager.track_symbol(
        watchlist_name=watchlist_name,
        ticker="TSLA",
        tags=["automotive", "ev"],
        notes="Range breakout candidate",
        priority=2
    )
    
    # 3. View watchlists
    print("\nWatchlist Contents:")
    watchlists = manager.get_watchlists()
    for name, wl in watchlists.items():
        print(f"Watchlist: {name}")
        for ticker, item in wl.items.items():
            print(f"  - {ticker} | Priority: {item.priority} | Target Price: {item.target_price} | Notes: {item.notes}")
            
    # 4. Rename watchlist
    print(f"\nRenaming watchlist '{watchlist_name}' to 'Tech Watch'...")
    manager.rename_watchlist(watchlist_name, "Tech Watch")
    
    # 5. Untrack a symbol
    print("Removing TSLA from Tech Watch...")
    manager.untrack_symbol("Tech Watch", "TSLA")
    
    print("\nUpdated Watchlist Contents:")
    for name, wl in manager.get_watchlists().items():
        print(f"Watchlist: {name}")
        for ticker in wl.items.keys():
            print(f"  - {ticker}")
    print()

if __name__ == "__main__":
    main()
