import os
import sys
import json
from pathlib import Path

# Force append project source path to sys.path
project_path = Path(__file__).resolve().parent.parent
sys.path.append(str(project_path))

from src.agent.pipeline import trading_agent
from src.config.settings import settings

TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
        
    print("="*90)
    print("STONKS PHASE 5 DEMONSTRATION RUN: GENERATING INTELLIGENT RECOMMENDATIONS...")
    print("="*90)
    
    # Configure defaults to ensure CatBoost leaderboard winner is selected
    settings.MODEL = "catboost"
    
    results = {}
    
    for symbol in TICKERS:
        print(f"\nProcessing symbol: {symbol}...")
        try:
            # Run agent pipeline (leverage cache to be fast)
            output = trading_agent.run_pipeline(symbol, force_train=False)
            
            intel = output["intelligence"]
            results[symbol] = intel["json_report"]
            
            # Print Markdown report directly to console
            print(intel["markdown_report"])
            print("-" * 80)
            
        except Exception as e:
            print(f"Error running pipeline for {symbol}: {e}")
            
    # Save the consolidated JSON to scratch
    output_json_path = project_path / "scratch" / "demonstration_results.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nConsolidated JSON reports saved successfully to: {output_json_path}")
    print("="*90)

if __name__ == "__main__":
    main()
