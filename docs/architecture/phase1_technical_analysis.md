# STONKS Phase 1: Technical Analysis Engine

This document details the initial construction of the Technical Analysis Engine, the foundation of the STONKS platform.

---

## 1. Technical Ingestion Pipeline

Phase 1 established the market data pipeline to ingest historical price feeds and compute traditional statistical indicators:

```mermaid
graph LR
    Ticker[Ticker Entry] --> Fetch[Ingest Candles]
    Fetch --> Clean[Handle NaNs]
    Clean --> Calc[Indicators Calculation]
    Calc --> Store[Feature Store Dataframe]
```

---

## 2. Engineered Indicators (17 Columns)

* **Moving Averages**: 10-day SMA, 20-day SMA, 50-day SMA, 20-day EMA, 50-day EMA.
* **Momentum**: Relative Strength Index (RSI), MACD, MACD Signal.
* **Volatility**: Bollinger Bands (Upper, Middle, Lower), 20-day standard deviation.
* **Statistical Distribution**: daily returns, 5-day return, 20-day return, skewness, kurtosis.

---

## 3. Random Forest Baseline

* **Classifier**: Spawns standard `RandomForestClassifier` with 100 estimators.
* **Evaluation**: Employs out-of-sample chronological backtests, yielding baseline direction accuracies (~48% to ~52%).
* **Feature Importance**: Identified skewness and short-term returns as the most active splitting nodes.
