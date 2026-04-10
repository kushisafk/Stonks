"""
Main execution script for Stonks model.
Runs the full pipeline for all configured tickers.
"""
import sys
import os
import pandas as pd

# Ensure imports work from project root (Stonks/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.config.settings import TICKERS, START_DATE, END_DATE

from model.src.data_pipeline import run_data_pipeline
from model.src.forester_model import run_forester
from model.src.golden_ratio import run_golden_ratio
from model.src.backtester import run_backtest
from model.src.visualizer import plot_all

def main():
    print("Starting Stonks Model Pipeline...")
    
    summary_metrics = []
    
    for ticker in TICKERS:
        try:
            print(f"\n[{ticker}] Starting pipeline...")
            
            # 1. Data Pipeline
            print(f"[{ticker}] Fetching and processing data...")
            raw_df, X, y = run_data_pipeline(ticker, START_DATE, END_DATE)
            
            if len(X) < 100:
                print(f"[{ticker}] Not enough data to proceed.")
                continue
                
            # 2. Forester Model
            print(f"[{ticker}] Training Forester model...")
            model, model_metrics, predictions_df = run_forester(ticker, X, y)
            print(f"[{ticker}] Model metrics: RMSE: {model_metrics['RMSE']:.2f}, R2: {model_metrics['R2']:.2f}")
            
            # 3. Golden Ratio Signals
            print(f"[{ticker}] Generating Golden Ratio signals...")
            signal_df = run_golden_ratio(ticker, predictions_df)
            
            # 4. Backtesting
            print(f"[{ticker}] Running backtest...")
            portfolio_df, backtest_metrics = run_backtest(ticker, signal_df)
            
            backtest_metrics['Ticker'] = ticker
            summary_metrics.append(backtest_metrics)
            
            # 5. Visualization
            print(f"[{ticker}] Generating plots...")
            plot_all(ticker, signal_df, portfolio_df)
            
            print(f"[{ticker}] Pipeline complete.")
            
        except Exception as e:
            import traceback
            print(f"[{ticker}] FAILED: {str(e)}")
            traceback.print_exc()
            
    # Print summary table
    if summary_metrics:
        print("\n" + "="*50)
        print("PIPELINE SUMMARY METRICS")
        print("="*50)
        summary_df = pd.DataFrame(summary_metrics)
        cols = ['Ticker'] + [c for c in summary_df.columns if c != 'Ticker']
        print(summary_df[cols].to_string(index=False))
        print("="*50)
    else:
        print("\nNo metrics to display. All pipelines failed.")

if __name__ == "__main__":
    main()
