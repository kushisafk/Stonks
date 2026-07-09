"""
STONKS Basic Usage Example

Demonstrates how to import the TradingAgent pipeline, run predictive evaluations 
for a symbol (e.g. AAPL), and access natural-language explanations.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from stonks.agent.pipeline import trading_agent
from stonks.logging.logger import logger

def main():
    ticker = "AAPL"
    logger.info(f"Running STONKS predictive analysis for {ticker}...")
    
    try:
        # Run pipeline inference (uses cached model weights or trains if missing)
        result = trading_agent.run_pipeline(ticker, force_train=False)
        
        # Access outputs
        intel = result["intelligence"]["json_report"]
        prediction = intel["prediction"]
        probability = intel["probability"]
        recommendation = intel["recommendation"]
        
        print("\n" + "=" * 50)
        print(f"Prediction for {ticker}: {prediction} ({probability})")
        print(f"Final Recommendation : {recommendation}")
        print("=" * 50)
        
        # Output natural-language reasoning
        print("\nDecision Reasoning Explanation:")
        print(result["explanation"])
        print("=" * 50 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to run basic usage example: {e}")

if __name__ == "__main__":
    main()
