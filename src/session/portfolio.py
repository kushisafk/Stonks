from typing import Dict, Any, Optional
from src.session.schemas import SessionState, PositionStatus, PositionType

class PortfolioService:
    """Computes and maintains portfolio value, buying power, exposures, and P/L metrics dynamically."""
    
    def update_portfolio_metrics(self, state: SessionState, current_prices: Optional[Dict[str, float]] = None) -> None:
        """
        Recomputes portfolio metrics based on active positions and current prices.
        
        Args:
            state: The active SessionState instance
            current_prices: Dictionary of current prices for active positions.
                            Falls back to position entry price if not provided.
        """
        prices = current_prices or {}
        
        active_positions = {
            ticker: pos 
            for ticker, pos in state.positions.items() 
            if pos.status in (PositionStatus.OPEN, PositionStatus.PARTIAL) and pos.quantity > 0.0
        }
        
        lmv = 0.0  # Long Market Value
        smv = 0.0  # Short Market Value (Liability)
        unrealized_pl = 0.0
        largest_position_ticker = None
        largest_position_value = -1.0
        
        for ticker, pos in active_positions.items():
            curr_price = prices.get(ticker, pos.entry_price)
            pos_val = pos.quantity * curr_price
            
            if pos.position_type == PositionType.LONG:
                lmv += pos_val
                pl = (curr_price - pos.entry_price) * pos.quantity
            else:
                smv += pos_val
                pl = (pos.entry_price - curr_price) * pos.quantity
                
            unrealized_pl += pl
            
            if pos_val > largest_position_value:
                largest_position_value = pos_val
                largest_position_ticker = ticker
                
        port = state.portfolio
        port.open_equity = lmv - smv
        port.unrealized_pl = unrealized_pl
        port.total_equity = port.cash_balance + port.open_equity
        port.portfolio_value = port.total_equity
        
        # Exposure metrics
        port.long_exposure = lmv
        port.short_exposure = smv
        port.net_exposure = lmv - smv
        
        port.buying_power = max(0.0, port.cash_balance)
        port.largest_position = largest_position_ticker
        port.daily_pl = port.unrealized_pl + port.realized_pl
        
    def get_sector_exposures(self, state: SessionState, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Aggregates and returns the percentage sector exposure of the portfolio."""
        prices = current_prices or {}
        active_positions = {
            ticker: pos 
            for ticker, pos in state.positions.items() 
            if pos.status in (PositionStatus.OPEN, PositionStatus.PARTIAL) and pos.quantity > 0.0
        }
        
        sectors = {}
        total_exp = 0.0
        
        for ticker, pos in active_positions.items():
            curr_price = prices.get(ticker, pos.entry_price)
            pos_val = pos.quantity * curr_price
            sector = pos.sector or "Unknown"
            sectors[sector] = sectors.get(sector, 0.0) + pos_val
            total_exp += pos_val
            
        if total_exp > 0:
            return {sect: val / total_exp for sect, val in sectors.items()}
        return {}
