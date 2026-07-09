# STONKS Prediction Pipeline

This document outlines the predictive intelligence pipeline that converts raw ticker inputs into directional trade signals.

---

## 1. Pipeline Overview

The prediction pipeline ingests market datasets, engineers feature arrays, queries CatBoost, calibrates the output probabilities, and returns buy/sell confidence levels:

```mermaid
graph LR
    Ticker[Ticker Symbol] --> Fetch[Ingest Historical Candles]
    Fetch --> Align[Reindex & Align with SPY Index]
    Align --> FeatureStore[Engineer 42 Features]
    FeatureStore --> CatBoost[Query Calibrated CatBoost]
    CatBoost --> Sig[Calibrated Probability Output]
```

---

## 2. Feature Matrix (42 Columns)

Features are categorized into five groups:
1. **Technical Indicators (17)**: rolling Close price changes, volatility metrics, MACD, RSI, Bollinger Bands, and return statistical distributions.
2. **Sentiment Indicators (8)**: continuous sentiment scores, negative news counts, and FinBERT positive ratio trackers.
3. **Index Dynamics (7)**: S&P 500 ETF (SPY) return offsets, index RSI levels, and trend directions.
4. **Relative Strength (5)**: stocks vs SPY performance differentials over 5-day, 20-day, and 50-day lookback windows.
5. **Volume Intelligence (5)**: volume standard dev ratios, momentum averages, and abnormal volume breakout triggers ($>2.0\text{x}$ 20-day SMAs).

---

## 3. CatBoost Inference & Probability Calibration

* **Symmetric Tree Engine**: CatBoost executes predictions using symmetric trees, which generalize better to out-of-sample data.
* **Calibrated Output**: Standard CatBoost raw outputs can display overconfidence (probabilities clustering near 1.0 or 0.0). STONKS utilizes isotonic calibration wrapper layers (`CalibratedClassifierCV` or direct isotonic transformations) to map the output to precise probability statistics. This guarantees that a 70% probability output represents a true 70% historical success rate.
