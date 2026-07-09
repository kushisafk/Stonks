from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter

class PortfolioCommands:
    """Handles the 'portfolio' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: portfolio <subcommand> [args]\nSubcommands: summary, exposure, performance, sectors, risk, history, cash")
            
        subcmd = args[0].lower()
        
        # Recalculate metrics first to ensure absolute accuracy
        self.manager.update_portfolio_metrics()
        port = self.manager.get_portfolio()
        
        if subcmd == "summary":
            pl_str = f"${port.unrealized_pl:+,.2f}"
            if port.unrealized_pl > 0:
                pl_str = TextFormatter.green(pl_str)
            elif port.unrealized_pl < 0:
                pl_str = TextFormatter.red(pl_str)
                
            real_str = f"${port.realized_pl:+,.2f}"
            if port.realized_pl > 0:
                real_str = TextFormatter.green(real_str)
            elif port.realized_pl < 0:
                real_str = TextFormatter.red(real_str)
                
            lines = [
                f"Cash Balance    : ${port.cash_balance:,.2f}",
                f"Buying Power    : ${port.buying_power:,.2f}",
                f"Open Equity     : ${port.open_equity:,.2f}",
                f"Total Equity    : ${port.total_equity:,.2f}",
                f"Portfolio Value : ${port.portfolio_value:,.2f}",
                f"Unrealized P/L  : {pl_str}",
                f"Realized P/L    : {real_str}",
                f"Largest Position: {port.largest_position or 'None'}"
            ]
            print(TextFormatter.to_panel("Portfolio Summary", lines))
            
        elif subcmd == "exposure":
            net_str = f"${port.net_exposure:+,.2f}"
            if port.net_exposure > 0:
                net_str = TextFormatter.green(net_str)
            elif port.net_exposure < 0:
                net_str = TextFormatter.red(net_str)
                
            lines = [
                f"Long Exposure   : ${port.long_exposure:,.2f}",
                f"Short Exposure  : ${port.short_exposure:,.2f}",
                f"Net Exposure    : {net_str}"
            ]
            print(TextFormatter.to_panel("Portfolio Exposure", lines))
            
        elif subcmd in ("performance", "history"):
            positions = self.manager.get_positions()
            closed = {t: p for t, p in positions.items() if p.status.value == "CLOSED"}
            if not closed:
                print("No closed position history recorded.")
                return
                
            headers = ["Ticker", "Type", "Entry Price", "Realized P/L"]
            rows = []
            for t, p in closed.items():
                pl_str = f"${p.realized_pl:+,.2f}"
                if p.realized_pl > 0:
                    pl_str = TextFormatter.green(pl_str)
                elif p.realized_pl < 0:
                    pl_str = TextFormatter.red(pl_str)
                rows.append([t, p.position_type.value, f"${p.entry_price:,.2f}", pl_str])
            print(f"\n{TextFormatter.bold('Closed Trade History')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "sectors":
            exposures = self.manager.get_sector_exposures()
            if not exposures:
                print("No active sector exposures to display.")
                return
            headers = ["Sector", "Allocation %"]
            rows = []
            for sect, pct in exposures.items():
                rows.append([sect, f"{pct:.2%}"])
            print(f"\n{TextFormatter.bold('Sector Allocation Matrix')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "risk":
            # Risk panel showing positions and standard warnings
            positions = self.manager.get_positions()
            active = {t: p for t, p in positions.items() if p.status.value != "CLOSED"}
            if not active:
                print("No active positions to assess risk.")
                return
                
            headers = ["Ticker", "Stop Loss Status", "Take Profit Status", "Exposure Risk"]
            rows = []
            for t, p in active.items():
                stop_str = TextFormatter.green("Set") if p.current_stop_loss else TextFormatter.red("NOT SET")
                target_str = TextFormatter.green("Set") if p.current_take_profit else TextFormatter.yellow("NOT SET")
                
                exp_ratio = (p.quantity * p.entry_price) / port.total_equity if port.total_equity > 0 else 0.0
                if exp_ratio > 0.20:
                    risk_label = TextFormatter.red("CONCENTRATED (High)")
                else:
                    risk_label = TextFormatter.green("Balanced")
                rows.append([t, stop_str, target_str, risk_label])
            print(f"\n{TextFormatter.bold('Portfolio Risk Control Assessment')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "cash":
            if len(args) < 2:
                raise UsageError("Usage: portfolio cash <amount>")
            try:
                amount = float(args[1])
            except ValueError:
                raise UsageError("Cash amount must be a valid number.")
            if amount < 0:
                raise UsageError("Cash balance cannot be negative.")
                
            self.manager.set_cash_balance(amount)
            print(f"{TextFormatter.SUCCESS} Portfolio cash balance updated to ${amount:,.2f}")
            
        else:
            raise CommandError(f"Unknown portfolio subcommand '{subcmd}'. Type 'help portfolio' for options.")
