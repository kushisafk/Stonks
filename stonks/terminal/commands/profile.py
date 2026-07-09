from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter
from stonks.session.schemas import TradingStyle, RiskProfile

class ProfileCommands:
    """Handles the 'profile' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: profile <subcommand> [args]\nSubcommands: show, edit, risk, capital, preferences")
            
        subcmd = args[0].lower()
        
        if subcmd == "show":
            prof = self.manager.state.user_profile
            lines = [
                f"Username        : {prof.username}",
                f"Trading Style   : {prof.trading_style.value}",
                f"Risk Profile    : {prof.risk_profile.value}",
                f"Default Capital : ${prof.default_capital:,.2f}",
                f"Base Currency   : {prof.base_currency}",
                f"Timezone        : {prof.timezone}",
                f"Alert Channels  : {', '.join(prof.preferred_notification_channels) if prof.preferred_notification_channels else 'None'}"
            ]
            print(TextFormatter.to_panel("User Trading Profile", lines))
            
        elif subcmd == "edit":
            if len(args) < 3:
                raise UsageError("Usage: profile edit <field> <value>\nFields: username, style, timezone, currency")
            field = args[1].lower()
            val = args[2]
            
            try:
                if field == "username":
                    self.manager.update_profile(username=val)
                elif field == "style":
                    # Parse trading style
                    try:
                        style_enum = TradingStyle(val)
                    except ValueError:
                        valid = [x.value for x in TradingStyle]
                        raise UsageError(f"Invalid Trading Style. Valid options: {valid}")
                    self.manager.update_profile(trading_style=style_enum)
                elif field == "timezone":
                    self.manager.update_profile(timezone=val)
                elif field == "currency":
                    self.manager.update_profile(base_currency=val)
                else:
                    raise UsageError(f"Unknown profile field '{field}'.")
                print(f"{TextFormatter.SUCCESS} Updated profile field '{field}' successfully.")
            except Exception as e:
                raise CommandError(str(e))
                
        elif subcmd == "risk":
            if len(args) < 2:
                raise UsageError("Usage: profile risk <Conservative/Balanced/Aggressive>")
            risk_str = args[1].capitalize()
            try:
                risk_enum = RiskProfile(risk_str)
                self.manager.update_profile(risk_profile=risk_enum)
                print(f"{TextFormatter.SUCCESS} Updated Risk Profile to '{risk_str}'.")
            except ValueError:
                valid = [x.value for x in RiskProfile]
                raise UsageError(f"Invalid Risk Profile. Valid options: {valid}")
                
        elif subcmd == "capital":
            if len(args) < 2:
                raise UsageError("Usage: profile capital <amount>")
            try:
                cap = float(args[1])
            except ValueError:
                raise UsageError("Capital must be a valid number.")
            try:
                self.manager.update_profile(default_capital=cap)
                print(f"{TextFormatter.SUCCESS} Updated default capital allocation to ${cap:,.2f}")
            except Exception as e:
                raise CommandError(str(e))
                
        elif subcmd == "preferences":
            if len(args) >= 3:
                # Setting a preference
                pref_field = args[1].lower()
                pref_val = args[2]
                
                try:
                    if pref_field == "theme":
                        self.manager.update_preferences(theme=pref_val)
                    elif pref_field == "model":
                        self.manager.update_preferences(preferred_ml_model=pref_val)
                    elif pref_field == "language":
                        self.manager.update_preferences(preferred_language=pref_val)
                    elif pref_field == "format":
                        self.manager.update_preferences(default_report_format=pref_val)
                    else:
                        raise UsageError(f"Unknown preference field '{pref_field}'.")
                    print(f"{TextFormatter.SUCCESS} Updated preference '{pref_field}' successfully.")
                except Exception as e:
                    raise CommandError(str(e))
            else:
                # Show preferences
                pref = self.manager.get_preferences()
                lines = [
                    f"Theme            : {pref.theme}",
                    f"Language         : {pref.preferred_language}",
                    f"Preferred Model  : {pref.preferred_ml_model}",
                    f"Report Format    : {pref.default_report_format}",
                    f"Polling Interval : {pref.polling_interval} hours",
                    f"Thresholds       : BUY >= {pref.threshold_profile.get('buy', 0.70):.2f}, SELL <= {pref.threshold_profile.get('sell', 0.40):.2f}"
                ]
                print(TextFormatter.to_panel("System Preferences", lines))
                
        else:
            raise CommandError(f"Unknown profile subcommand '{subcmd}'. Type 'help profile' for options.")
