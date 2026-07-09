"""
STONKS Portfolio Management Demo

Demonstrates how to open positions (long/short), update stop losses, 
and recompute allocations/equity balances dynamically.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stonks.session.manager import TradingSessionManager
from stonks.logging.logger import logger

def main():
    # Use temporary session file in memory or local test directory
    session_file = Path("models_data/session.json")
    manager = TradingSessionManager(session_file)
    manager.create_session() # Start with clean slate
    
    print("\n" + "=" * 50)
    print("Initial Portfolio Balance:")
    port = manager.get_portfolio()
    print(f"Cash Balance: ${port.cash_balance:,.2f}")
    print(f"Total Equity: ${port.total_equity:,.2f}")
    print("=" * 50 + "\n")
    
    # 1. Open Long Position: AAPL 100 shares @ $150
    logger.info("Opening LONG Position: AAPL 100 shares @ $150...")
    manager.open_long_position(ticker="AAPL", entry_price=150.0, quantity=100.0, stop_loss=145.0, take_profit=170.0)
    
    # 2. Open Short Position: TSLA 10 shares @ $220
    logger.info("Opening SHORT Position: TSLA 10 shares @ $220...")
    manager.open_short_position(ticker="TSLA", entry_price=220.0, quantity=10.0)
    
    # 3. View position metrics
    print("\nActive Open Positions:")
    positions = manager.get_positions()
    for ticker, p in positions.items():
        if p.status.value != "CLOSED":
            print(f"  {ticker} | Type: {p.position_type.value} | Qty: {p.quantity} | Entry: ${p.entry_price:.2f} | Stop: ${p.current_stop_loss}")
            
    # 4. Trigger price revaluation: AAPL increases to $160, TSLA drops to $200 (profit on both!)
    logger.info("Market Shift: Recalculating allocations at new market prices (AAPL: $160, TSLA: $200)...")
    manager.update_portfolio_metrics(current_prices={"AAPL": 160.0, "TSLA": 200.0})
    
    print("\nUpdated Portfolio Balances:")
    port = manager.get_portfolio()
    print(f"Cash Balance   : ${port.cash_balance:,.2f}")
    print(f"Open Equity    : ${port.open_equity:,.2f}")
    print(f"Total Equity   : ${port.total_equity:,.2f}")
    print(f"Unrealized P/L : ${port.unrealized_pl:+,.2f}")
    print(f"Largest holding: {port.largest_position}")
    print("=" * 50 + "\n")
    
    # 5. Liquidate AAPL position
    logger.info("Liquidating AAPL position @ $162...")
    manager.close_position("AAPL", 162.0)
    
    port = manager.get_portfolio()
    print(f"Final Cash Balance: ${port.cash_balance:,.2f}")
    print(f"Final Realized P/L: ${port.realized_pl:+,.2f}\n")

if __name__ == "__main__":
    main()
