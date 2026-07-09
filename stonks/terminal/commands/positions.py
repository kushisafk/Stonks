from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter
from stonks.session.exceptions import SessionError

class PositionCommands:
    """Handles the 'position' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: position <subcommand> [args]\nSubcommands: list, open long, open short, close, reduce, increase, review, update-stop, update-target")
            
        subcmd = args[0].lower()
        
        if subcmd == "list":
            positions = self.manager.get_positions()
            active = {t: p for t, p in positions.items() if p.status.value != "CLOSED"}
            if not active:
                print("No open positions.")
                return
                
            headers = ["Ticker", "Type", "Qty", "Entry Price", "Stop Loss", "Take Profit", "Unrealized P/L"]
            rows = []
            for ticker, p in active.items():
                pl = p.realized_pl # placeholder or we can print current P/L if we had prices.
                # Let's check if we can fetch current price for active positions
                from stonks.data.market_data import market_data_service
                try:
                    df = market_data_service.fetch_data(ticker)
                    curr_price = float(df.iloc[-1]["Close"])
                except Exception:
                    curr_price = p.entry_price
                    
                if p.position_type.value == "LONG":
                    unrealized = (curr_price - p.entry_price) * p.quantity
                else:
                    unrealized = (p.entry_price - curr_price) * p.quantity
                    
                pl_str = f"${unrealized:+,.2f}"
                if unrealized > 0:
                    pl_str = TextFormatter.green(pl_str)
                elif unrealized < 0:
                    pl_str = TextFormatter.red(pl_str)
                    
                rows.append([
                    ticker,
                    p.position_type.value,
                    f"{p.quantity:,.2f}",
                    f"${p.entry_price:,.2f}",
                    f"${p.current_stop_loss:,.2f}" if p.current_stop_loss else "-",
                    f"${p.current_take_profit:,.2f}" if p.current_take_profit else "-",
                    pl_str
                ])
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "open":
            if len(args) < 3:
                raise UsageError("Usage: position open <long/short> <ticker> [qty] [price]")
            direction = args[1].lower()
            ticker = args[2].upper()
            
            qty = None
            price = None
            if len(args) >= 5:
                try:
                    qty = float(args[3])
                    price = float(args[4])
                except ValueError:
                    raise UsageError("Quantity and price must be valid numbers.")
                    
            if qty is None or price is None:
                try:
                    qty = float(input(f"Enter quantity for {ticker}: "))
                    price = float(input(f"Enter entry price for {ticker}: "))
                except ValueError:
                    raise UsageError("Invalid quantity or price input.")
                    
            try:
                if direction == "long":
                    self.manager.open_long_position(ticker, price, qty)
                    print(f"{TextFormatter.SUCCESS} Opened LONG position: {qty} {ticker} @ ${price:.2f}")
                elif direction == "short":
                    self.manager.open_short_position(ticker, price, qty)
                    print(f"{TextFormatter.SUCCESS} Opened SHORT position: {qty} {ticker} @ ${price:.2f}")
                else:
                    raise UsageError("Direction must be 'long' or 'short'.")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "close":
            if len(args) < 2:
                raise UsageError("Usage: position close <ticker> [exit_price]")
            ticker = args[1].upper()
            
            exit_price = None
            if len(args) >= 3:
                try:
                    exit_price = float(args[2])
                except ValueError:
                    raise UsageError("Exit price must be a valid number.")
                    
            if exit_price is None:
                try:
                    exit_price = float(input(f"Enter exit price for {ticker}: "))
                except ValueError:
                    raise UsageError("Invalid exit price input.")
                    
            try:
                self.manager.close_position(ticker, exit_price)
                print(f"{TextFormatter.SUCCESS} Closed position for {ticker} @ ${exit_price:.2f}")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "reduce":
            if len(args) < 3:
                raise UsageError("Usage: position reduce <ticker> <percent> [exit_price]")
            ticker = args[1].upper()
            try:
                pct = float(args[2].replace("%", "")) / 100.0
            except ValueError:
                raise UsageError("Percent must be a valid number (e.g. 50 or 50%).")
                
            if pct <= 0 or pct > 1.0:
                raise UsageError("Percent must be between 1% and 100%.")
                
            exit_price = None
            if len(args) >= 4:
                try:
                    exit_price = float(args[3])
                except ValueError:
                    raise UsageError("Exit price must be a valid number.")
                    
            if exit_price is None:
                try:
                    exit_price = float(input(f"Enter exit price for {ticker}: "))
                except ValueError:
                    raise UsageError("Invalid exit price input.")
                    
            positions = self.manager.get_positions()
            if ticker not in positions or positions[ticker].status.value == "CLOSED":
                raise CommandError(f"No active position for {ticker}.")
                
            qty_to_reduce = positions[ticker].quantity * pct
            try:
                self.manager.partial_close_position(ticker, exit_price, qty_to_reduce)
                print(f"{TextFormatter.SUCCESS} Reduced position for {ticker} by {pct:.1%} ({qty_to_reduce:.2f} shares) @ ${exit_price:.2f}")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "increase":
            if len(args) < 2:
                raise UsageError("Usage: position increase <ticker> [qty] [price]")
            ticker = args[1].upper()
            
            qty = None
            price = None
            if len(args) >= 4:
                try:
                    qty = float(args[2])
                    price = float(args[3])
                except ValueError:
                    raise UsageError("Quantity and price must be valid numbers.")
                    
            if qty is None or price is None:
                try:
                    qty = float(input(f"Enter quantity to add for {ticker}: "))
                    price = float(input(f"Enter entry price for additional shares: "))
                except ValueError:
                    raise UsageError("Invalid quantity or price input.")
                    
            try:
                # Re-route to standard increase position
                pos = self.manager.get_positions().get(ticker)
                if not pos or pos.status.value == "CLOSED":
                    raise CommandError(f"No active position for {ticker}. Open a position first.")
                if pos.position_type.value == "LONG":
                    self.manager.open_long_position(ticker, price, qty)
                else:
                    self.manager.open_short_position(ticker, price, qty)
                print(f"{TextFormatter.SUCCESS} Increased position for {ticker} by {qty} shares @ ${price:.2f}")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "review":
            if len(args) < 2:
                raise UsageError("Usage: position review <ticker>")
            ticker = args[1].upper()
            pos = self.manager.get_positions().get(ticker)
            if not pos:
                raise CommandError(f"No position history found for {ticker}.")
                
            lines = [
                f"Ticker: {pos.ticker}",
                f"Type: {pos.position_type.value}",
                f"Status: {pos.status.value}",
                f"Quantity: {pos.quantity:,.2f}",
                f"Entry Price: ${pos.entry_price:,.2f}",
                f"Stop Loss: ${pos.current_stop_loss:,.2f}" if pos.current_stop_loss else "Stop Loss: -",
                f"Take Profit: ${pos.current_take_profit:,.2f}" if pos.current_take_profit else "Take Profit: -",
                f"Realized P/L: ${pos.realized_pl:+,.2f}"
            ]
            print(TextFormatter.to_panel(f"Position Review: {ticker}", lines))
            
        elif subcmd == "update-stop":
            if len(args) < 3:
                raise UsageError("Usage: position update-stop <ticker> <price/none>")
            ticker = args[1].upper()
            price_str = args[2].lower()
            price = None if price_str == "none" else float(price_str)
            try:
                self.manager.update_stop_loss(ticker, price)
                print(f"{TextFormatter.SUCCESS} Updated Stop Loss for {ticker} to {f'${price:.2f}' if price else 'None'}")
            except SessionError as e:
                raise CommandError(str(e))
                
        elif subcmd == "update-target":
            if len(args) < 3:
                raise UsageError("Usage: position update-target <ticker> <price/none>")
            ticker = args[1].upper()
            price_str = args[2].lower()
            price = None if price_str == "none" else float(price_str)
            try:
                self.manager.update_take_profit(ticker, price)
                print(f"{TextFormatter.SUCCESS} Updated Take Profit for {ticker} to {f'${price:.2f}' if price else 'None'}")
            except SessionError as e:
                raise CommandError(str(e))
                
        else:
            raise CommandError(f"Unknown position subcommand '{subcmd}'. Type 'help position' for options.")
