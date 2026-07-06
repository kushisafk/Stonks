from datetime import datetime
from typing import Dict, List, Optional
from src.session.schemas import SessionState, Position, PositionType, PositionStatus
from src.session.exceptions import PositionNotFoundError, InvalidStateError

class PositionService:
    """Manages the lifecycle of LONG and SHORT trading positions in the session state."""
    
    def open_long(
        self, 
        state: SessionState, 
        ticker: str, 
        entry_price: float, 
        quantity: float, 
        stop_loss: Optional[float] = None, 
        take_profit: Optional[float] = None
    ) -> None:
        """Opens a new LONG position or increases an existing one."""
        ticker = ticker.strip().upper()
        if entry_price <= 0 or quantity <= 0:
            raise InvalidStateError("Price and quantity must be positive.")
            
        cost = entry_price * quantity
        if cost > state.portfolio.cash_balance:
            raise InvalidStateError(f"Insufficient cash balance. Required: ${cost:,.2f}, Available: ${state.portfolio.cash_balance:,.2f}")
            
        if ticker in state.positions and state.positions[ticker].status != PositionStatus.CLOSED:
            pos = state.positions[ticker]
            if pos.position_type != PositionType.LONG:
                raise InvalidStateError(f"Cannot open LONG position for {ticker} when a SHORT position is active.")
            self.increase_position(state, ticker, entry_price, quantity)
        else:
            state.portfolio.cash_balance -= cost
            state.positions[ticker] = Position(
                ticker=ticker,
                position_type=PositionType.LONG,
                entry_price=entry_price,
                quantity=quantity,
                entry_date=datetime.now().isoformat(),
                current_stop_loss=stop_loss,
                current_take_profit=take_profit,
                status=PositionStatus.OPEN
            )
            
    def open_short(
        self, 
        state: SessionState, 
        ticker: str, 
        entry_price: float, 
        quantity: float, 
        stop_loss: Optional[float] = None, 
        take_profit: Optional[float] = None
    ) -> None:
        """Opens a new SHORT position or increases an existing one."""
        ticker = ticker.strip().upper()
        if entry_price <= 0 or quantity <= 0:
            raise InvalidStateError("Price and quantity must be positive.")
            
        cost = entry_price * quantity
        if cost > state.portfolio.buying_power:
            raise InvalidStateError(f"Insufficient buying power for short sale. Required: ${cost:,.2f}, Available: ${state.portfolio.buying_power:,.2f}")
            
        if ticker in state.positions and state.positions[ticker].status != PositionStatus.CLOSED:
            pos = state.positions[ticker]
            if pos.position_type != PositionType.SHORT:
                raise InvalidStateError(f"Cannot open SHORT position for {ticker} when a LONG position is active.")
            self.increase_position(state, ticker, entry_price, quantity)
        else:
            state.portfolio.cash_balance += cost
            state.positions[ticker] = Position(
                ticker=ticker,
                position_type=PositionType.SHORT,
                entry_price=entry_price,
                quantity=quantity,
                entry_date=datetime.now().isoformat(),
                current_stop_loss=stop_loss,
                current_take_profit=take_profit,
                status=PositionStatus.OPEN
            )
            
    def close_position(self, state: SessionState, ticker: str, exit_price: float) -> None:
        """Closes an open position entirely and realizes P/L."""
        ticker = ticker.strip().upper()
        if ticker not in state.positions or state.positions[ticker].status == PositionStatus.CLOSED:
            raise PositionNotFoundError(f"No active position found for ticker {ticker}.")
            
        pos = state.positions[ticker]
        if exit_price <= 0:
            raise InvalidStateError("Exit price must be positive.")
            
        if pos.position_type == PositionType.LONG:
            proceeds = pos.quantity * exit_price
            state.portfolio.cash_balance += proceeds
            pl = (exit_price - pos.entry_price) * pos.quantity
        else:
            cover_cost = pos.quantity * exit_price
            state.portfolio.cash_balance -= cover_cost
            pl = (pos.entry_price - exit_price) * pos.quantity
            
        pos.realized_pl += pl
        state.portfolio.realized_pl += pl
        pos.quantity = 0.0
        pos.status = PositionStatus.CLOSED
        
    def partial_close(self, state: SessionState, ticker: str, exit_price: float, quantity: float) -> None:
        """Closes a portion of an open position."""
        ticker = ticker.strip().upper()
        if ticker not in state.positions or state.positions[ticker].status == PositionStatus.CLOSED:
            raise PositionNotFoundError(f"No active position found for ticker {ticker}.")
            
        pos = state.positions[ticker]
        if quantity <= 0 or exit_price <= 0:
            raise InvalidStateError("Quantity and exit price must be positive.")
        if quantity > pos.quantity:
            raise InvalidStateError(f"Cannot partial close {quantity} shares. Active position quantity is {pos.quantity}.")
            
        if quantity == pos.quantity:
            self.close_position(state, ticker, exit_price)
            return
            
        if pos.position_type == PositionType.LONG:
            proceeds = quantity * exit_price
            state.portfolio.cash_balance += proceeds
            pl = (exit_price - pos.entry_price) * quantity
        else:
            cover_cost = quantity * exit_price
            state.portfolio.cash_balance -= cover_cost
            pl = (pos.entry_price - exit_price) * quantity
            
        pos.realized_pl += pl
        state.portfolio.realized_pl += pl
        pos.quantity -= quantity
        pos.status = PositionStatus.PARTIAL
        
    def increase_position(self, state: SessionState, ticker: str, price: float, quantity: float) -> None:
        """Increases size of an active position (averages entry price)."""
        ticker = ticker.strip().upper()
        if ticker not in state.positions or state.positions[ticker].status == PositionStatus.CLOSED:
            raise PositionNotFoundError(f"No active position found for ticker {ticker}.")
            
        pos = state.positions[ticker]
        cost = price * quantity
        
        if pos.position_type == PositionType.LONG:
            if cost > state.portfolio.cash_balance:
                raise InvalidStateError("Insufficient cash balance to increase long position.")
            state.portfolio.cash_balance -= cost
        else:
            if cost > state.portfolio.buying_power:
                raise InvalidStateError("Insufficient buying power to increase short position.")
            state.portfolio.cash_balance += cost
            
        total_qty = pos.quantity + quantity
        pos.entry_price = ((pos.quantity * pos.entry_price) + (quantity * price)) / total_qty
        pos.quantity = total_qty
        pos.status = PositionStatus.OPEN
        
    def reduce_position(self, state: SessionState, ticker: str, price: float, quantity: float) -> None:
        """Reduces size of an active position (realizes partial P/L)."""
        self.partial_close(state, ticker, price, quantity)
        
    def update_stop_loss(self, state: SessionState, ticker: str, stop_loss: Optional[float]) -> None:
        """Updates the stop loss value for a position."""
        ticker = ticker.strip().upper()
        if ticker not in state.positions or state.positions[ticker].status == PositionStatus.CLOSED:
            raise PositionNotFoundError(f"No active position found for ticker {ticker}.")
        state.positions[ticker].current_stop_loss = stop_loss
        
    def update_take_profit(self, state: SessionState, ticker: str, take_profit: Optional[float]) -> None:
        """Updates the take profit value for a position."""
        ticker = ticker.strip().upper()
        if ticker not in state.positions or state.positions[ticker].status == PositionStatus.CLOSED:
            raise PositionNotFoundError(f"No active position found for ticker {ticker}.")
        state.positions[ticker].current_take_profit = take_profit
