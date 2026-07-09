# Changelog

All notable changes to the STONKS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-07
### Added
* First production Release Candidate (RC-1).
* Cleaned codebase including unused imports removal, RLock thread safety, and type hint improvements.
* Dynamic terminal autocompleters and persisted command histories.
* Standardized issue and pull request templates for GitHub collaboration.
* Professional MIT License.

---

## [0.8.0] - 2026-07-06
### Added
* background event-driven `StonksRuntime` scheduler and priority worker pool daemon threads.
* Fault-tolerance mechanisms permitting isolated job crashes without process termination.
* EventBus pub/sub structures linking price changes to Stop-Loss liquidations automatically.

---

## [0.7.0] - 2026-07-02
### Added
* Interactive Terminal operating shell environment.
* Shlex token parsing supporting quoted inputs and namespace command aliases (e.g. `p list`).
* ASCII table formatters and panel layouts.

---

## [0.6.0] - 2026-06-28
### Added
* `TradingSessionManager` facade decoupling active state from local agent memories.
* Atomic serialization writes flushes and backup restore handlers.

---

## [0.5.0] - 2026-06-15
### Added
* Trading Intelligence Layer containing stateless Market Reasoner, Risk Assessor, and Confidence Analyzer.
* AI explanation logics detailing decision outputs.

---

## [0.4.0] - 2026-06-10
### Added
* Central model registry and walk-forward benchmarking suites evaluating classical models.
* Selected CatBoost as the best overall predictive engine.

---

## [0.3.0] - 2026-06-01
### Added
* Market-Context features including S&P 500 ETF (SPY) alignments, relative strength parameters, and volume breakouts.

---

## [0.2.0] - 2026-05-15
### Added
* Sentiment Intelligence processing pipelines integrating Hugging Face FinBERT classifications.

---

## [0.1.0] - 2026-05-01
### Added
* Initial Technical Analysis indicators engine and Random Forest baseline classifier models.
