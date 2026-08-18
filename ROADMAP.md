# STONKS Project Roadmap

This roadmap outlines planned features, model additions, and platform upgrades for future releases.

---

## [v1.1.0] - Multi-Market Ingestion & Asset Registry
* **🇺🇸 US Stocks**: Complete data scraping and technical indexing for major US equities.
* **🇮🇳 Indian Stocks & ETFs**: Integration of NSE/BSE tickers, Nifty indices, and local liquid exchange-traded funds.
* **Asset Registry**: Central configuration registry of supported assets, metadata, tickers, and trading hours.
* **Custom Asset Additions**: Commands to register custom tickers or CSV feeds dynamically.
* **Per-Asset Models**: Automated per-ticker retraining and optimization schedules for localized CatBoost models.

---

## [v1.2.0] - Deep Sequence Classifiers
* **Deep Neural Classifier Integrations**:
  - Long Short-Term Memory (LSTM) network classifiers for sequential trends.
  - Gated Recurrent Units (GRU) to optimize sequence weights.
  - 1D Temporal Convolutional Networks (TCN) for local multi-scale trend features extraction.
* **Benchmark Evaluation**: Chronological walk-forward sweeps comparing deep models against the tree-based CatBoost baseline.

---

## [v1.3.0] - Advanced Attention Models & Ensembles
* **Transformers**: Ingestion of time-series transformers (e.g. vanilla encoder-decoders, Informer, PatchTST) for long-lookback predictions.
* **Hybrid Ensembles**: Stacking tree models (CatBoost/XGBoost) and attention models (Transformers) to construct meta-ensembles.
* **Context features**: Improved macro indicator correlations, global indices alignment, and yield curve spreads context tracking.

---

## [v1.4.0] - Simulated Paper Trading & Risk Operations
* **Paper Trading Sandbox**: Background simulated execution engine tracking hypothetical entry/exit fill rates without capital risk.
* **Performance Logs**: Tracking portfolio alpha, beta exposures, Sharpe ratios, maximum drawdowns, and historical win-rates in session logs.
* **Advanced Risk & Sizing**: Sizing allocations based on average true range (ATR) volatility and correlation-based concentration limits.

---

## Later - Production Connectors & Dashboards
* **Broker & Exchange APIs**: Optional direct connectivity for trade execution (e.g., Alpaca API, Interactive Brokers API, Zerodha/Kite API).
* **Live Notifications**: Subscribing Slack, Discord, Telegram, or email dispatch agents to Runtime events.
* **Web UI Dashboard**: React/TypeScript web app displaying live portfolio charts, position trackers, and runtime status cards.
