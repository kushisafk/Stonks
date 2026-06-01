import sys
import argparse
import time
import uvicorn
from pathlib import Path
from src.config.settings import settings
from src.logging.logger import logger
from src.agent.pipeline import trading_agent
from src.backtesting.backtester import Backtester
from src.scheduler.scheduler import trading_scheduler
from src.data.market_data import market_data_service

def run_api_server() -> None:
    """Starts the FastAPI application using uvicorn."""
    logger.info("Starting STONKS API Server...")
    # Bind to standard port 8000
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

def run_scheduler_mode() -> None:
    """Starts the scheduler in the foreground, blocking until interrupted."""
    logger.info("Starting STONKS Automated Scheduler Daemon...")
    trading_scheduler.start()
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        trading_scheduler.stop()
        logger.info("STONKS Scheduler daemon terminated gracefully.")

def execute_single_prediction(symbol: str, force_train: bool) -> None:
    """Executes a single end-to-end coordinated pipeline pass for a stock ticker."""
    try:
        result = trading_agent.run_pipeline(symbol, force_train=force_train)
        print("\n" + "=" * 55)
        print(f" STONKS TRADING AGENT SIGNAL REPORT: {symbol.upper()}")
        print("=" * 55)
        print(f"Timestamp:    {result['timestamp']}")
        print(f"Signal:       {result['signal']}")
        print(f"Confidence:   {result['confidence']:.2%}")
        print(f"Close Price:  ${result['close_price']:.2f}")
        print("-" * 55)
        print(f"Explanation:  {result['explanation']}")
        print("=" * 55 + "\n")
    except Exception as e:
        logger.error(f"Single prediction pipeline pass failed for {symbol}: {e}")
        sys.exit(1)

def execute_backtest(symbol: str) -> None:
    """Executes a rolling walk-forward backtest on the stock ticker and prints a report."""
    try:
        raw_df = market_data_service.fetch_data(symbol)
        backtester = Backtester()
        results = backtester.run_walk_forward(symbol, raw_df)
        
        ml = results["ml_metrics"]
        tr = results["trading_metrics"]
        
        print("\n" + "=" * 55)
        print(f" STONKS WALK-FORWARD RESEARCH PERFORMANCE: {symbol.upper()}")
        print("=" * 55)
        print(" 1. MACHINE LEARNING DIRECTIONAL CLASSIFICATION:")
        print(f"    Out-of-Sample Accuracy:   {ml['accuracy']:.2%}")
        print(f"    Directional Precision:    {ml['precision']:.2%}")
        print(f"    Directional Recall:       {ml['recall']:.2%}")
        print(f"    Directional F1-Score:     {ml['f1']:.2%}")
        print("-" * 55)
        print(" 2. PORTFOLIO TRADING METRICS (Walk-Forward Out-of-Sample):")
        print(f"    Starting Capital:         ${tr['starting_capital']:,.2f}")
        print(f"    Ending Capital:           ${tr['ending_capital']:,.2f}")
        print(f"    Strategy Net Return:      {tr['strategy_return']:.2%}")
        print(f"    Buy & Hold Net Return:    {tr['buy_and_hold_return']:.2%}")
        alpha = tr['strategy_return'] - tr['buy_and_hold_return']
        print(f"    Net Outperformance (Alpha): {alpha:+.2%}")
        print(f"    Win Rate on Executed Trades: {tr['win_rate']:.2%}")
        print(f"    Total Completed Trades:   {tr['total_trades']}")
        print(f"    Annualized Volatility:    {tr['annualized_volatility']:.2%}")
        print(f"    Annualized Sharpe Ratio:  {tr['sharpe_ratio']:.2f}")
        print(f"    Maximum Portfolio Drawdown: {tr['max_drawdown']:.2%}")
        print("=" * 55 + "\n")
    except Exception as e:
        logger.error(f"Backtesting research run failed for {symbol}: {e}")
        sys.exit(1)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="STONKS: A Production-Grade AI Trading Decision Platform."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--api", action="store_true", help="Start the FastAPI app server using Uvicorn")
    group.add_argument("--scheduler", action="store_true", help="Start the APScheduler background daemon")
    group.add_argument("--predict", type=str, metavar="TICKER", help="Execute a single prediction pipeline pass on ticker")
    group.add_argument("--backtest", type=str, metavar="TICKER", help="Execute a rolling walk-forward backtest research run")
    
    parser.add_argument(
        "--force-train", 
        action="store_true", 
        help="Forces retraining of the base model (only applicable with --predict)"
    )
    
    args = parser.parse_args()
    
    if args.api:
        run_api_server()
    elif args.scheduler:
        run_scheduler_mode()
    elif args.predict:
        execute_single_prediction(args.predict, args.force_train)
    elif args.backtest:
        execute_backtest(args.backtest)

if __name__ == "__main__":
    main()
