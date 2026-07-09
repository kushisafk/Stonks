from datetime import datetime
from stonks.runtime.events import (
    Event, RecommendationChangedEvent, PriceUpdateEvent, 
    PortfolioChangedEvent, AlertTriggeredEvent, PositionClosedEvent
)
from stonks.session.schemas import Alert
from stonks.logging.logger import logger

class EventDispatcher:
    """Subscribes to Event Bus and triggers session actions like Alert generation."""
    
    def __init__(self, manager, event_bus):
        self.manager = manager
        self.event_bus = event_bus
        
    def register_listeners(self) -> None:
        """Hooks handler callables to respective Event classes on the Event Bus."""
        self.event_bus.subscribe(RecommendationChangedEvent, self._handle_recommendation_change)
        self.event_bus.subscribe(PriceUpdateEvent, self._handle_price_update)
        self.event_bus.subscribe(PortfolioChangedEvent, self._handle_portfolio_change)
        
    def _handle_recommendation_change(self, event: RecommendationChangedEvent) -> None:
        """Creates alert entry when an asset recommendation shifts."""
        ticker = event.data.get("ticker")
        old_rec = event.data.get("old_recommendation")
        new_rec = event.data.get("new_recommendation")
        
        msg = f"Recommendation shift for {ticker}: {old_rec} -> {new_rec}"
        logger.info(f"Dispatcher: {msg}")
        
        alert = Alert(
            timestamp=datetime.now().isoformat(),
            rule_type="Recommendation Change",
            ticker=ticker,
            message=msg
        )
        self.manager.record_alert(alert)
        
    def _handle_price_update(self, event: PriceUpdateEvent) -> None:
        """Checks if current price actions trigger stop loss or take profit targets."""
        ticker = event.data.get("ticker")
        curr_price = event.data.get("price")
        if not ticker or not curr_price:
            return
            
        positions = self.manager.get_positions()
        if ticker in positions:
            p = positions[ticker]
            if p.status.value == "CLOSED":
                return
                
            # Check Stop Loss
            if p.current_stop_loss:
                is_breached = False
                if p.position_type.value == "LONG" and curr_price <= p.current_stop_loss:
                    is_breached = True
                elif p.position_type.value == "SHORT" and curr_price >= p.current_stop_loss:
                    is_breached = True
                    
                if is_breached:
                    msg = f"STOP LOSS breached for {ticker} at ${curr_price:.2f} (Target stop: ${p.current_stop_loss:.2f})"
                    logger.warning(f"Dispatcher: {msg}")
                    
                    alert = Alert(
                        timestamp=datetime.now().isoformat(),
                        rule_type="Stop Loss Triggered",
                        ticker=ticker,
                        message=msg
                    )
                    self.manager.record_alert(alert)
                    
                    # Auto-close position in session
                    self.manager.close_position(ticker, curr_price)
                    
                    # Publish triggered events
                    self.event_bus.publish(AlertTriggeredEvent(data={"ticker": ticker, "reason": "stop_loss"}))
                    self.event_bus.publish(PositionClosedEvent(data={"ticker": ticker, "price": curr_price}))
                    return
                    
            # Check Take Profit Target
            if p.current_take_profit:
                is_breached = False
                if p.position_type.value == "LONG" and curr_price >= p.current_take_profit:
                    is_breached = True
                elif p.position_type.value == "SHORT" and curr_price <= p.current_take_profit:
                    is_breached = True
                    
                if is_breached:
                    msg = f"TAKE PROFIT target reached for {ticker} at ${curr_price:.2f} (Target profit: ${p.current_take_profit:.2f})"
                    logger.info(f"Dispatcher: {msg}")
                    
                    alert = Alert(
                        timestamp=datetime.now().isoformat(),
                        rule_type="Take Profit Triggered",
                        ticker=ticker,
                        message=msg
                    )
                    self.manager.record_alert(alert)
                    
                    # Auto-close position in session
                    self.manager.close_position(ticker, curr_price)
                    
                    # Publish triggered events
                    self.event_bus.publish(AlertTriggeredEvent(data={"ticker": ticker, "reason": "take_profit"}))
                    self.event_bus.publish(PositionClosedEvent(data={"ticker": ticker, "price": curr_price}))
                    
    def _handle_portfolio_change(self, event: PortfolioChangedEvent) -> None:
        """Assesses portfolio concentration risks."""
        # Simple concentration warning check
        port = self.manager.get_portfolio()
        if port.total_equity <= 0:
            return
            
        for ticker, pos in self.manager.get_positions().items():
            if pos.status.value != "CLOSED":
                alloc = (pos.quantity * pos.entry_price) / port.total_equity
                if alloc > 0.30:  # Concentrated if allocation is greater than 30%
                    msg = f"High portfolio concentration detected for {ticker}: {alloc:.1%}"
                    logger.warning(f"Dispatcher: {msg}")
                    
                    alert = Alert(
                        timestamp=datetime.now().isoformat(),
                        rule_type="Risk Level Alert",
                        ticker=ticker,
                        message=msg
                    )
                    self.manager.record_alert(alert)
