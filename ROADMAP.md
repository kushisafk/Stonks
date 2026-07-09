# STONKS Project Roadmap

This roadmap outlines planned features, model additions, and platform upgrades for future releases.

---

## [v1.1.0] - Deep Sequence Models
* **Ingestion Upgrades**: Transition pipeline sequence outputs to 3D arrays.
* **Deep Classifier Integrations**:
  - Long Short-Term Memory (LSTM) network classifiers.
  - Gated Recurrent Units (GRU).
  - 1D Temporal Convolutional Networks (TCN) for local trend features extraction.
* **Benchmark Sweep**: Compare sequence deep models against the baseline CatBoost.

---

## [v1.2.0] - Attention & Time-Series Foundation Models
* **Transformers**: Integrate vanilla time-series encoder-decoder networks.
* **Advanced Architectures**:
  - Informer / PatchTST for long-sequence prediction.
  - Fine-tune open-weight Time-Series Foundation Models (e.g. TimesFM) via Hugging Face.

---

## [v1.3.0] - Hybrid Ensembles
* **CatBoost + Transformer**: Combine classical tree models (best for tabular allocations) with attention models (best for sequence patterns) into a stacked meta-model.
* **Dynamic Probability Calibrations**: Real-time recalibration under volatile market regimes.

---

## [v1.4.0] - Broker Integrations & Paper Trading
* **Execution Agents**: Support standard broker transaction connections (e.g., Alpaca API, Interactive Brokers API).
* **Live Paper Trading**: Background tracking of actual fill rates without risk.
* **Advanced Analytics**: Sharpe ratio, information ratio, and beta exposures tracked in live session states.

---

## [v1.5.0] - Web Dashboard & Live Notifications
* **React Web App**: Replace or supplement the interactive terminal with a premium React/TypeScript dashboard showing portfolio charts and positions.
* **Notification dispatchers**: Send trade alerts directly to Slack, Discord, or Telegram.
