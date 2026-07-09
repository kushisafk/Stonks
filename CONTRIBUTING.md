# Contributing to STONKS

We welcome contributions from quantitative researchers, software engineers, and machine learning practitioners! This document outlines guidelines for contributing code, documenting features, and reporting bugs.

---

## 1. Development Setup

To configure a local development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/krish/stonks.git
   cd stonks
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install black ruff mypy pytest
   ```

---

## 2. Coding Style & Linting

We enforce clean, standardized formatting rules:
* **Code Formatter**: [Black](https://github.com/psf/black) (`black .`)
* **Linter**: [Ruff](https://github.com/astral-sh/ruff) (`ruff check .`)
* **Static Types**: [Mypy](https://github.com/python/mypy) (`mypy stonks/`)

Always run checks before submitting a PR:
```bash
black --check .
ruff check .
mypy stonks/
```

---

## 3. Running Tests

STONKS uses `pytest` for unit and integration testing. Ensure all tests pass cleanly:
```bash
pytest
```

---

## 4. Git Branching & Commit Guidelines

### Branch Naming Conventions:
* Features: `feature/short-description`
* Bug Fixes: `fix/bug-description`
* Refactoring: `refactor/cleanups`
* Docs: `docs/documentation-update`

### Commit Message Formats:
We follow semantic commit patterns (Angular convention):
* `feat: add CatBoost calibrated inference wrapper`
* `fix: prevent WinError 5 locks in atomic persistence writer`
* `docs: add runtime pipelines diagrams`
* `test: cover priority worker cancellations`
