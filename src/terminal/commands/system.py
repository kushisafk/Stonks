import os
from typing import List, Dict
from src.terminal.errors import UsageError, CommandError
from src.terminal.formatter import TextFormatter

class SystemCommands:
    """Handles the global system namespace commands (help, version, clear)."""
    
    def __init__(self, manager):
        self.manager = manager
        
        # General Help Descriptions
        self.help_namespaces = {
            "general": [
                "STONKS Command Line Interface (CLI)",
                "Syntax: <namespace> <subcommand> [arguments] [flags]",
                "",
                "Namespaces:",
                "  market     - Run price predictions, sentiment sweeps, and technical research.",
                "  position   - Manage LONG and SHORT trading positions in the session state.",
                "  portfolio  - Display cash balances, exposures, sector allocation, and P/L.",
                "  watchlist  - Maintain custom symbol watchlist groupings.",
                "  research   - Audits ML model benchmarking leaderboards, thresholds, and features.",
                "  profile    - Show/Edit trading preferences and risk thresholds.",
                "  session    - Status check, save, reload, or reset session configurations.",
                "  alerts     - Query trigger history or acknowledge stop loss triggers.",
                "  runtime    - Command background orchestrator threads.",
                "",
                "System:",
                "  help       - Display help (or 'help <namespace>' for subcommand syntax).",
                "  clear      - Clear the terminal screen.",
                "  version    - Output software version metadata.",
                "  exit/quit  - Terminate the active shell loop gracefully (auto-saves session)."
            ],
            "market": [
                "Market Command Namespace Subcommands:",
                "  market analyze TICKER       - Run CatBoost prediction and FinBERT sentiment reasoning report.",
                "  market compare TICKER1 TK2  - Side-by-side metric comparison table.",
                "  market explain TICKER       - Generate natural-language technical explainer report.",
                "  market news TICKER          - Display recent articles news titles and publisher details.",
                "  market inspect TICKER       - Dump raw JSON of pipeline prediction payload.",
                "  market research TICKER      - Display recent historical daily candles and volumes.",
                "  market chart TICKER         - Placeholder subcommand for chart rendering."
            ],
            "position": [
                "Position Command Namespace Subcommands:",
                "  position list               - Display tabular view of open/partial positions.",
                "  position open long TICKER   - Buy asset entry (prompts for qty and price if omitted).",
                "  position open short TICKER  - Short sell asset entry.",
                "  position close TICKER       - Liquidate active position at exit price.",
                "  position reduce TICKER %    - Partial close position by percentage (e.g. 50%).",
                "  position increase TICKER    - Scale-in to an active position (averages price).",
                "  position review TICKER      - Panel summarizing entry dates, stop losses, and target prices.",
                "  position update-stop TK PR  - Update stop loss pricing (or 'none' to clear).",
                "  position update-target TK PR- Update take profit target (or 'none' to clear)."
            ],
            "portfolio": [
                "Portfolio Command Namespace Subcommands:",
                "  portfolio summary           - Panel showing total cash, open equity, buying power, P/L.",
                "  portfolio exposure          - Details LMV, SMV, and Net Dollar exposure.",
                "  portfolio sectors           - Summary table of percentage sector allocations.",
                "  portfolio risk              - Assessment of stop-losses and concentration risk constraints.",
                "  portfolio history/performance- List realized P/L of closed trades."
            ],
            "watchlist": [
                "Watchlist Command Namespace Subcommands:",
                "  watchlist list              - Displays all watchlist groups and symbol items.",
                "  watchlist create NAME       - Add a new named watchlist category.",
                "  watchlist delete NAME       - Delete a watchlist category.",
                "  watchlist add NAME TICKER   - Track a symbol in a named watchlist.",
                "  watchlist remove NAME TICKER- Untrack a symbol.",
                "  watchlist rename OLD NEW    - Rename a watchlist category."
            ],
            "research": [
                "Research Command Namespace Subcommands:",
                "  research benchmark          - Display ML benchmarking leaderboard of classical models.",
                "  research thresholds         - Display current probability thresholds for decision routing.",
                "  research features           - Summary of engineered indicators in Feature Store.",
                "  research models             - List registered models in central registry.",
                "  research importance         - Show top 10 feature importances of best-performing model.",
                "  research history            - Summary metrics of decisions count."
            ],
            "session": [
                "Session Command Namespace Subcommands:",
                "  session status              - Summary of session file path, size, schema, user count.",
                "  session save                - Force atomic write serialization to disk.",
                "  session reload              - Force read reload from file.",
                "  session reset               - Reset session state to clean defaults."
            ],
            "profile": [
                "Profile Command Namespace Subcommands:",
                "  profile show                - Show username, timezone, timezone, default capital, currency.",
                "  profile edit FIELD VAL      - Edit username, style (Day Trader, etc.), timezone, currency.",
                "  profile risk RISK_LEVEL     - Modify risk tier (Conservative, Balanced, Aggressive).",
                "  profile capital AMOUNT      - Edit default capital allocation bounds.",
                "  profile preferences         - Display system settings and override thresholds."
            ],
            "alerts": [
                "Alerts Command Namespace Subcommands:",
                "  alerts list                 - Table showing logged triggers history.",
                "  alerts clear                - Delete alert logging records.",
                "  alerts acknowledge          - Mark active alerts as acknowledged."
            ],
            "runtime": [
                "Runtime Command Namespace Subcommands:",
                "  runtime start               - Start background scheduler and workers.",
                "  runtime stop                - Stop background runtime engine.",
                "  runtime restart             - Restart background runtime engine.",
                "  runtime status              - Display running status state.",
                "  runtime metrics             - Print ASCII table of runtime health metrics.",
                "  runtime jobs                - List active scheduled tasks in queue.",
                "  runtime events              - List event subscribers count.",
                "  runtime heartbeat           - Output current process health parameters.",
                "  runtime config              - Show task intervals scheduler configuration."
            ]
        }
        
    def execute(self, cmd: str, args: List[str]) -> None:
        cmd = cmd.lower()
        
        if cmd == "help":
            if args:
                ns = args[0].lower()
                if ns in self.help_namespaces:
                    print(TextFormatter.to_panel(f"Help: {ns.capitalize()}", self.help_namespaces[ns]))
                else:
                    raise UsageError(f"No help topic found for namespace '{ns}'.")
            else:
                print(TextFormatter.to_panel("Help: STONKS Interactive Shell", self.help_namespaces["general"]))
                
        elif cmd == "version":
            print(f"STONKS AI Trading Operating System - Version 1.0.0 (Phase 8 Runtime)")
            
        elif cmd == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            
        else:
            raise CommandError(f"Unknown system command '{cmd}'.")
