from typing import List
from stonks.terminal.errors import UsageError, CommandError
from stonks.terminal.formatter import TextFormatter
from stonks.models.model_registry import list_registered_models, get_best_model
from stonks.config.settings import settings

class ResearchCommands:
    """Handles the 'research' command namespace actions."""
    
    def __init__(self, manager):
        self.manager = manager
        
    def execute(self, args: List[str]) -> None:
        if not args:
            raise UsageError("Usage: research <subcommand> [args]\nSubcommands: benchmark, thresholds, features, models, importance, history")
            
        subcmd = args[0].lower()
        
        if subcmd == "benchmark":
            print("Reading current ML benchmark leaderboard...")
            import json
            leaderboard_path = settings.MODEL_DIR / "leaderboard.json"
            if not leaderboard_path.exists():
                print("No benchmark leaderboard found. Run a benchmark suite sweep first.")
                return
                
            with open(leaderboard_path, "r") as f:
                data = json.load(f)
                
            headers = ["Rank", "Model", "Overall Score", "Avg Sharpe", "Avg Alpha", "Avg Max DD", "Avg Accuracy"]
            rows = []
            for idx, r in enumerate(data, 1):
                rows.append([
                    str(idx),
                    r.get("model", "N/A"),
                    f"{r.get('overall_score', 0.0):.4f}",
                    f"{r.get('avg_sharpe', 0.0):.4f}",
                    f"{r.get('avg_alpha', 0.0):+.2%}",
                    f"{r.get('avg_dd', 0.0):.2%}",
                    f"{r.get('avg_accuracy', 0.0):.2%}"
                ])
            print(f"\n{TextFormatter.bold('Production ML Benchmarking Leaderboard')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "thresholds":
            # Display current universal threshold profile
            lines = [
                f"Active Threshold profile: Universal v1",
                f"BUY_THRESHOLD : {settings.BUY_THRESHOLD:.2f} (probabilities >= 70% support BUY)",
                f"SELL_THRESHOLD: {settings.SELL_THRESHOLD:.2f} (probabilities <= 40% support SELL)"
            ]
            print(TextFormatter.to_panel("Universal Threshold Boundaries", lines))
            
        elif subcmd == "features":
            # Print features list from feature pipeline
            from stonks.features.feature_pipeline import feature_pipeline
            lines = [
                "Total engineered indicators: 42 Features",
                "Categories: Technical Indicators, Market Context, Sentiment Features,",
                "            Volume Intelligence, Relative Strength Trend Features.",
                "",
                "Key Columns: daily_return, rsi, macd, volatility_20d, spy_trend_strength,",
                "             relative_strength_20d, volume_ratio, sentiment_score."
            ]
            print(TextFormatter.to_panel("Feature Store Metrics", lines))
            
        elif subcmd == "models":
            # List registered models
            registered = list_registered_models()
            headers = ["Model Key", "Wrapper Class Name"]
            rows = []
            for k, cls in registered.items():
                rows.append([k, cls.__name__])
            print(f"\n{TextFormatter.bold('Central Model Registry')}")
            print(TextFormatter.to_table(headers, rows))
            
        elif subcmd == "importance":
            # Get best model feature importances
            try:
                best_cls = get_best_model()
                # Load trained model to retrieve weights
                active_model = best_cls()
                model_name = best_cls.__name__.replace("Model", "").lower()
                # Try to load AAPL model weights
                from pathlib import Path
                model_path = settings.MODEL_DIR / f"AAPL_{model_name}.joblib"
                if not model_path.exists():
                    # Fallback to general rf if active catboost model file not trained yet for AAPL
                    model_path = settings.MODEL_DIR / "AAPL_rf.joblib"
                
                if model_path.exists():
                    active_model.load(model_path)
                    importances = active_model.feature_importances
                    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
                    
                    headers = ["Feature Name", "Relative Importance %"]
                    rows = []
                    for name, val in sorted_imp:
                        rows.append([name, f"{val:.2%}"])
                    print(f"\n{TextFormatter.bold('Top Feature Importances (Model: ' + best_cls.__name__ + ')')}")
                    print(TextFormatter.to_table(headers, rows))
                else:
                    print("No trained model weights found to extract feature importances. Run a pipeline training first.")
            except Exception as e:
                raise CommandError(f"Could not retrieve feature importances: {e}")
                
        elif subcmd == "history":
            # Display decision statistics
            decisions = self.manager.get_recent_decisions()
            if not decisions:
                print("No historical decision records logged in this session.")
                return
            buy_cnt = sum(1 for d in decisions if d.recommendation == "BUY")
            sell_cnt = sum(1 for d in decisions if d.recommendation == "SELL")
            hold_cnt = sum(1 for d in decisions if d.recommendation == "HOLD")
            lines = [
                f"Total Logged Decisions: {len(decisions)}",
                f"BUY Actions           : {buy_cnt}",
                f"SELL Actions          : {sell_cnt}",
                f"HOLD Actions          : {hold_cnt}"
            ]
            print(TextFormatter.to_panel("Decision History Metrics", lines))
            
        else:
            raise CommandError(f"Unknown research subcommand '{subcmd}'. Type 'help research' for options.")
