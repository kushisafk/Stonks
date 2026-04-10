# Stonks — Model

Algorithmic trading model implementation based on Forester and Golden Ratio.

## Based on Paper
"Algorithmic Trading Model for Stock Price Forecasting Integrating Forester with Golden Ratio Strategy" 
(2024 IEEE R10-HTC, DOI: 10.1109/R10-HTC59322.2024.10778666)

## Setup Instructions
```bash
pip install -r requirements.txt
```

## How to Run
```bash
python main.py
```

## Folder Structure
- `config/`: Configuration settings (tickers, dates, model params)
- `data/`: Raw and processed data storage
- `src/`: Core Python modules for data, model, backtest, and visualization
- `notebooks/`: Jupyter notebooks for exploration and demonstration
- `outputs/`: Saved models, generated plots, and backtest results
- `tests/`: Unit tests for the pipeline

## Sample Output
Running `main.py` will fetch historical data for defined NSE tickers, train an ExtraTrees (Forester) model to predict daily closing prices, generate buy/sell signals based on Fibonacci levels (Golden Ratio), and execute a simulated backtest. Outputs include metric CSVs and visualizations in the `outputs/` folder.

## Tech Stack
- Python
- yfinance
- scikit-learn
- pandas & numpy
- matplotlib & plotly
