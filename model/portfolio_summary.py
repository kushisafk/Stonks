def compute_portfolio_summary(ticker_metrics: dict[str, dict]) -> dict:
    """
    Computes an aggregated summary of metrics across all backtested tickers.
    """
    tickers = list(ticker_metrics.keys())
    if not tickers:
        return {
            "avg_total_return_pct": 0.0,
            "avg_annualized_return_pct": 0.0,
            "avg_sharpe_ratio": 0.0,
            "avg_max_drawdown_pct": 0.0,
            "avg_win_rate_pct": 0.0,
            "avg_buy_and_hold_return_pct": 0.0,
            "tickers": []
        }
        
    total_metrics = {
        "Total_Return_Pct": 0.0,
        "Annualized_Return_Pct": 0.0,
        "Sharpe_Ratio": 0.0,
        "Max_Drawdown_Pct": 0.0,
        "Win_Rate_Pct": 0.0,
        "Buy_and_Hold_Return_Pct": 0.0
    }
    
    for t, m in ticker_metrics.items():
        total_metrics["Total_Return_Pct"] += m.get("Total_Return_Pct", 0)
        total_metrics["Annualized_Return_Pct"] += m.get("Annualized_Return_Pct", 0)
        total_metrics["Sharpe_Ratio"] += m.get("Sharpe_Ratio", 0)
        total_metrics["Max_Drawdown_Pct"] += m.get("Max_Drawdown_Pct", 0)
        total_metrics["Win_Rate_Pct"] += m.get("Win_Rate_Pct", 0)
        total_metrics["Buy_and_Hold_Return_Pct"] += m.get("Buy_and_Hold_Return_Pct", 0)
        
    n = len(tickers)
    return {
        "avg_total_return_pct": round(total_metrics["Total_Return_Pct"] / n, 2),
        "avg_annualized_return_pct": round(total_metrics["Annualized_Return_Pct"] / n, 2),
        "avg_sharpe_ratio": round(total_metrics["Sharpe_Ratio"] / n, 2),
        "avg_max_drawdown_pct": round(total_metrics["Max_Drawdown_Pct"] / n, 2),
        "avg_win_rate_pct": round(total_metrics["Win_Rate_Pct"] / n, 2),
        "avg_buy_and_hold_return_pct": round(total_metrics["Buy_and_Hold_Return_Pct"] / n, 2),
        "tickers": tickers
    }
