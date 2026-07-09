from typing import Dict, Any, List, Optional
from stonks.session.schemas import SessionState, TradingStyle, RiskProfile

class PreferencesService:
    """Manages modifications to the UserProfile and system Preferences configurations."""
    
    def update_profile(
        self, 
        state: SessionState, 
        username: Optional[str] = None,
        trading_style: Optional[TradingStyle] = None,
        risk_profile: Optional[RiskProfile] = None,
        default_capital: Optional[float] = None,
        base_currency: Optional[str] = None,
        timezone: Optional[str] = None,
        notification_channels: Optional[List[str]] = None
    ) -> None:
        """Modifies fields inside the user profile state block."""
        prof = state.user_profile
        if username is not None:
            prof.username = username.strip()
        if trading_style is not None:
            prof.trading_style = trading_style
        if risk_profile is not None:
            prof.risk_profile = risk_profile
        if default_capital is not None:
            if default_capital < 0:
                raise ValueError("Default capital cannot be negative.")
            prof.default_capital = default_capital
        if base_currency is not None:
            prof.base_currency = base_currency.strip().upper()
        if timezone is not None:
            prof.timezone = timezone.strip()
        if notification_channels is not None:
            prof.preferred_notification_channels = notification_channels

    def update_preferences(
        self, 
        state: SessionState, 
        polling_interval: Optional[int] = None,
        theme: Optional[str] = None,
        default_report_format: Optional[str] = None,
        preferred_output_style: Optional[str] = None,
        threshold_profile: Optional[Dict[str, float]] = None,
        auto_optimization_enabled: Optional[bool] = None,
        preferred_ml_model: Optional[str] = None,
        preferred_language: Optional[str] = None
    ) -> None:
        """Modifies fields inside the system preferences state block."""
        pref = state.preferences
        if polling_interval is not None:
            if polling_interval <= 0:
                raise ValueError("Polling interval must be positive.")
            pref.polling_interval = polling_interval
        if theme is not None:
            pref.theme = theme.strip()
        if default_report_format is not None:
            pref.default_report_format = default_report_format.strip()
        if preferred_output_style is not None:
            pref.preferred_output_style = preferred_output_style.strip()
        if threshold_profile is not None:
            pref.threshold_profile = threshold_profile
        if auto_optimization_enabled is not None:
            pref.auto_optimization_enabled = auto_optimization_enabled
        if preferred_ml_model is not None:
            pref.preferred_ml_model = preferred_ml_model.strip().lower()
        if preferred_language is not None:
            pref.preferred_language = preferred_language.strip()
