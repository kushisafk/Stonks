# STONKS v1.0.0 Release Notes

We are thrilled to present the first stable release candidate of **STONKS (AI Trading Operating System) v1.0.0**.

STONKS is an event-driven operating environment designed to run predictive machine learning classifiers, retrieve market sentiment feeds, execute persistent mock portfolios, and run background analysis schedulers.

---

## 🚀 Key Capabilities & Features

### 1. Market Context Ingestion & Alignment
* Ingests chronological price candles and sentiment streams.
* Automatically aligns stock metrics with S&P 500 ETF (`SPY`) index overlays to prevent timezone leakage or look-ahead biases.
* Computes **42 engineered features** covering volume momentum, relative strength index ratios, volatility bands, and SPY index regimes.

### 2. Calibrated Predictive Intelligence
* Implements robust **CatBoost Classifier** tree models, evaluated and selected over five benchmarked classifiers across multi-seed walk-forward validations.
* Features **probability calibration** (isotonic wrappers) to ensure predictions act as mathematically calibrated confidence metrics.

### 3. Decoupled Trading Intelligence Layer
* Separates raw probability outputs from execution routing.
* Employs a stateless intelligence facade running **Market Reasoner**, **Confidence Analyzer**, **Risk Assessor**, and **Trade Planner** checks.
* Integrates a natural-language explainer translating complex ML split parameters into clear English descriptions.

### 4. Persistence & Session Management
* Runs a centralized `TradingSessionManager` acting as the state kernel.
* Implements atomic write sweeps, cache serialization, and automatic recovery rollbacks utilizing backups.

### 5. Interactive Developer Terminal
* A full REPL command shell supporting tokenized syntax parsing, autocomplete, persistent command history, and namespace subcommand shortcuts (e.g. `p list`).

### 6. Event-Driven Background Runtime
* Runs background schedulers and worker threads on a daemon executor thread pool.
* Subscribes dispatcher callbacks to EventBus channels (e.g., stop loss violations trigger position closing).

---

## 🛠 Project Structure

```
STONKS/
├── docs/                      # Architectural overview, guidelines, and command reference
├── examples/                  # Standard standalone Python usage scripts and walk-throughs
├── stonks/                       # Central application packages (agent, session, runtime, terminal)
├── tests/                     # Standard testing suite (pytest)
├── CHANGELOG.md               # Tracking product milestones
├── CONTRIBUTING.md            # Guidelines for coding and branching
├── ROADMAP.md                 # Outline of future deep sequence upgrades
└── SECURITY.md                # Responsible disclosure details and financial disclaimers
```

---

## 🔮 Future Plans
* **v1.1.0**: Sequence deep classifiers (LSTMs, GRUs, 1D TCNs).
* **v1.2.0**: Transformer model architectures and fine-tuning.
* **v1.3.0**: Tabular-sequence stacked hybrid ensembles.
* **v1.4.0**: alpaca/IB broker live connection APIs for mock paper trading.
