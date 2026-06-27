import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Force append project source path to sys.path
project_path = Path(__file__).resolve().parent.parent
sys.path.append(str(project_path))

from src.data.market_data import market_data_service
from src.features.feature_pipeline import feature_pipeline
from src.backtesting.backtester import Backtester
from src.decision.decision_engine import decision_engine
from src.config.settings import settings

# Define feature lists
base_feature_cols = [
    "daily_return", "return_5d", "return_20d",
    "ma10", "ma20", "ma50", "ema20", "ema50",
    "volatility_20d", "rsi", "macd", "macd_signal",
    "bb_middle", "bb_upper", "bb_lower",
    "skew_20d", "kurt_20d"
]

sentiment_cols = [
    "sentiment_score", "positive_news_ratio", "negative_news_ratio",
    "neutral_news_ratio", "article_count", "average_sentiment",
    "weighted_sentiment", "recency_weighted_sentiment"
]

market_context_cols = [
    "spy_return_1d", "spy_return_5d", "spy_return_20d", "spy_rsi", "spy_macd",
    "spy_volatility_20d", "spy_trend_strength", "relative_strength_5d",
    "relative_strength_20d", "relative_strength_50d", "relative_momentum_score",
    "volume_sma_20", "volume_ratio", "volume_momentum", "volume_trend",
    "abnormal_volume_flag", "market_regime"
]

# Set logging level to WARNING to avoid long output logs
import logging
logging.getLogger("src.logging.logger").setLevel(logging.WARNING)

# 1. Custom backtest executor that selects a specific subset of features
def run_custom_backtest(symbol, raw_df, feature_subset, train_window=250, test_window=50, model_mode="ensemble"):
    # Ingest and generate all features
    X_full, y_full = feature_pipeline.get_features(symbol, raw_df, is_training=True, use_store=False)
    
    # Filter X to only include the chosen feature subset
    X = X_full[feature_subset]
    y = y_full
    
    total_len = len(X)
    if total_len < train_window + test_window:
        raise ValueError("Insufficient data")
        
    all_preds = []
    all_probs = []
    all_targets = []
    test_indices = []
    
    for i in range(train_window, total_len, test_window):
        fold_test_end = min(total_len, i + test_window)
        X_train = X.iloc[i - train_window:i]
        y_train = y.iloc[i - train_window:i]
        X_test = X.iloc[i:fold_test_end]
        y_test = y.iloc[i:fold_test_end]
        
        if len(X_test) == 0:
            break
            
        # Train Random Forest
        from src.models.random_forest import RandomForestModel
        rf_model = RandomForestModel(n_estimators=100, max_depth=8, random_state=42)
        rf_model.train(X_train, y_train)
        
        if model_mode == "rf":
            probs = rf_model.predict_proba(X_test)
            preds = rf_model.predict(X_test)
        else:
            from src.ensemble.weighted_voting import WeightedEnsemble
            from src.models.finbert import FinBERTModel
            
            ensemble = WeightedEnsemble()
            ensemble.register_model("rf", rf_model)
            
            finbert_model = FinBERTModel()
            ensemble.register_model("finbert", finbert_model)
            
            ensemble.set_weight("rf", 0.70)
            ensemble.set_weight("finbert", 0.30)
            
            probs = ensemble.predict_proba(X_test)
            preds = ensemble.predict(X_test)
            
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_targets.extend(y_test)
        test_indices.extend(X_test.index)
        
    eval_df = pd.DataFrame({
        "probability": all_probs,
        "prediction": all_preds,
        "target": all_targets
    }, index=test_indices)
    
    eval_df = eval_df.join(raw_df["Close"], how="left")
    
    # Run strategy simulation using global backtester
    backtester = Backtester()
    sim_metrics = backtester.simulate_strategy(eval_df)
    ml_metrics = backtester.calculate_ml_metrics(np.array(all_preds), np.array(all_targets))
    
    return {
        "symbol": symbol,
        "ml_metrics": ml_metrics,
        "trading_metrics": sim_metrics,
        "eval_df": eval_df
    }

tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

# Pre-fetch raw dataframes
raw_dfs = {}
for s in tickers:
    raw_dfs[s] = market_data_service.fetch_data(s)

# ==============================================================================
# PHASE 1: RUN THE ABLATION STUDY
# ==============================================================================
print("\n" + "="*80)
# Configure universal thresholds
decision_engine.buy_threshold = 0.70
decision_engine.sell_threshold = 0.40
print(f"RUNNING ABLATION STUDY (Using universal thresholds: BUY=70%, SELL=40%)")
print("="*80)

ablation_results = []

for symbol in tickers:
    print(f"Evaluating {symbol}...")
    raw_df = raw_dfs[symbol]
    
    # Model A: Technical Only (Phase 1)
    res_a = run_custom_backtest(symbol, raw_df, base_feature_cols, model_mode="rf")
    
    # Model B: Technical + Sentiment (Phase 2)
    res_b = run_custom_backtest(symbol, raw_df, base_feature_cols + sentiment_cols, model_mode="ensemble")
    
    # Model C: Technical + Sentiment + Market Context (Phase 3 Full)
    res_c = run_custom_backtest(symbol, raw_df, base_feature_cols + sentiment_cols + market_context_cols, model_mode="ensemble")
    
    ablation_results.append({
        "ticker": symbol,
        "model_a": res_a,
        "model_b": res_b,
        "model_c": res_c
    })

# Output Ablation Study Markdown Table
print("\n" + "="*80)
print("ABLATION STUDY REPORT (Phase 1 vs. Phase 2 vs. Phase 3):")
print("="*80)
print("| Ticker | Model Group | Accuracy | Sharpe Ratio | CAGR / Strategy Return | Max Drawdown | Net Alpha |")
print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

for r in ablation_results:
    ticker = r["ticker"]
    bh_ret = r["model_a"]["trading_metrics"]["buy_and_hold_return"]
    
    for label, res in [("Model A (Technical Only)", r["model_a"]), 
                       ("Model B (Tech + Sentiment)", r["model_b"]), 
                       ("Model C (Tech + Sent + Market - Phase 3)", r["model_c"])]:
        ml = res["ml_metrics"]
        tr = res["trading_metrics"]
        
        acc = f"{ml['accuracy']:.2%}"
        sharpe = f"{tr['sharpe_ratio']:.4f}"
        strat_ret = f"{tr['strategy_return']:.2%}"
        dd = f"{tr['max_drawdown']:.2%}"
        alpha = tr['strategy_return'] - bh_ret
        alpha_str = f"{alpha:+.2%}"
        
        # Make Phase 3 bold for visual highlight
        if "Phase 3" in label:
            print(f"| **{ticker}** | **{label}** | **{acc}** | **{sharpe}** | **{strat_ret}** | **{dd}** | **{alpha_str}** |")
        else:
            print(f"| {ticker} | {label} | {acc} | {sharpe} | {strat_ret} | {dd} | {alpha_str} |")
    print("|---|---|---|---|---|---|---|")
print("="*80)


# ==============================================================================
# PHASE 2: RUN THE THRESHOLD RE-EVALUATION SWEEP
# ==============================================================================
print("\n" + "="*80)
print(f"RUNNING PARAMETERS THRESHOLD RE-EVALUATION SWEEP ON PHASE 3 FEATURES")
print("="*80)

buy_thresholds = [0.55, 0.60, 0.65, 0.70]
sell_thresholds = [0.30, 0.35, 0.40, 0.45]

sweep_results = []

# Precompute Phase 3 evaluation dataframes for fast simulation
eval_dfs = {}
for symbol in tickers:
    raw_df = raw_dfs[symbol]
    res_c = run_custom_backtest(symbol, raw_df, base_feature_cols + sentiment_cols + market_context_cols, model_mode="ensemble")
    eval_dfs[symbol] = res_c["eval_df"]

for buy_t in buy_thresholds:
    for sell_t in sell_thresholds:
        # Update global decision engine thresholds
        decision_engine.buy_threshold = buy_t
        decision_engine.sell_threshold = sell_t
        
        sharpes = []
        alphas = []
        dds = []
        returns = []
        
        for symbol, df in eval_dfs.items():
            backtester = Backtester()
            metrics = backtester.simulate_strategy(df)
            
            sharpes.append(metrics["sharpe_ratio"])
            alphas.append(metrics["strategy_return"] - metrics["buy_and_hold_return"])
            dds.append(metrics["max_drawdown"])
            returns.append(metrics["strategy_return"])
            
        avg_sharpe = np.mean(sharpes)
        avg_alpha = np.mean(alphas)
        avg_dd = np.mean(dds)
        avg_return = np.mean(returns)
        
        sweep_results.append({
            "buy_t": buy_t,
            "sell_t": sell_t,
            "avg_sharpe": avg_sharpe,
            "avg_alpha": avg_alpha,
            "avg_dd": avg_dd,
            "avg_return": avg_return
        })

# Sort sweep results by Average Sharpe Ratio descending
df_sweep = pd.DataFrame(sweep_results)
df_sweep.sort_values(by="avg_sharpe", ascending=False, inplace=True)

print("\n" + "="*80)
print("UNIVERSAL PARAMETER SWEEP RESULTS FOR PHASE 3 (Sorted by Avg Sharpe Ratio):")
print("="*80)
print("| Rank | Buy Threshold | Sell Threshold | Avg Sharpe Ratio | Avg Net Alpha | Avg Strategy Return | Avg Max Drawdown |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
for idx, (_, r) in enumerate(df_sweep.iterrows(), 1):
    buy_str = f"{int(r['buy_t']*100)}%"
    sell_str = f"{int(r['sell_t']*100)}%"
    sharpe_str = f"{r['avg_sharpe']:.4f}"
    alpha_str = f"{r['avg_alpha']:+.2%}"
    ret_str = f"{r['avg_return']:.2%}"
    dd_str = f"{r['avg_dd']:.2%}"
    
    # Make universal pair bold
    if r['buy_t'] == 0.70 and r['sell_t'] == 0.40:
        print(f"| {idx} | **{buy_str}** | **{sell_str}** | **{sharpe_str}** | **{alpha_str}** | **{ret_str}** | **{dd_str}** | (Recommended) |")
    else:
        print(f"| {idx} | {buy_str} | {sell_str} | {sharpe_str} | {alpha_str} | {ret_str} | {dd_str} |")
print("="*80)
