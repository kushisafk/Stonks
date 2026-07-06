from typing import List
from src.session.schemas import SessionState, Alert

class AlertService:
    """Manages trigger rules and persistent logging of trade alerts inside the session state."""
    
    def record_alert(self, state: SessionState, alert: Alert) -> None:
        """Appends a new alert to the session's alert history."""
        state.alerts.append(alert)
        
    def get_alerts(self, state: SessionState, triggered_only: bool = True) -> List[Alert]:
        """Retrieves alerts, optionally filtering by triggered status."""
        if triggered_only:
            return [a for a in state.alerts if a.triggered]
        return state.alerts
        
    def clear_alerts(self, state: SessionState) -> None:
        """Clears all logged alerts from the state."""
        state.alerts.clear()
