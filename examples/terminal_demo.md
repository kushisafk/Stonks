# STONKS Terminal Demo Guide

This document presents a mock terminal session demonstrating how users launch, inspect, and evaluate the STONKS Interactive Terminal.

---

## 1. Startup & Status Review

Run the following command in your terminal:
```bash
python -m stonks.terminal.app
```

Output:
```
╔════════════════════════════════════╗
║            STONKS                 ║
║ AI Trading Operating System       ║
╚════════════════════════════════════╝

Workspace : Default
Model     : Catboost
Watchlists: 2
Positions : 0
Alerts    : 0

Type "help" to begin.

stonks>
```

---

## 2. Managing Tickers on a Watchlist

```
stonks> watchlist list
No watchlists defined.

stonks> watchlist create Tech
[✓] Created watchlist 'Tech' successfully.

stonks> watchlist add Tech AAPL "Consolidating near 50 SMA" 3 175.00
[✓] Added AAPL to watchlist 'Tech'.

stonks> watchlist list

Watchlist: Tech
+--------+------------+----------------------------+----------+--------------+
| Ticker | Date Added | Notes                      | Priority | Target Price |
+--------+------------+----------------------------+----------+--------------+
| AAPL   | 2026-07-07 | Consolidating near 50 SMA  | 3        | $175.00      |
+--------+------------+----------------------------+----------+--------------+
```

---

## 3. Running Price Predictions & Market Research

Let's check NVDA:
```
stonks> market analyze NVDA
Running predictive intelligence pipeline for NVDA...

## STONKS prediction report for NVDA
Consensus Recommendation: BUY (78.2% calibrated probability)
Sentiment Index         : +0.45 (positive news bias)
Risk Level              : Moderate (Risk Score: 45)
Explanation             : Directional uptrend supported by abnormal volume breakouts.
```

---

## 4. Execution of Trading Positions

```
stonks> position open long AAPL 100 170.00
[✓] Opened LONG position: 100 AAPL @ $170.00

stonks> position list
+--------+------+--------+-------------+-----------+---------------+----------------+
| Ticker | Type | Qty    | Entry Price | Stop Loss | Take Profit   | Unrealized P/L |
+--------+------+--------+-------------+-----------+---------------+----------------+
| AAPL   | LONG | 100.00 | $170.00     | -         | -             | $0.00          |
+--------+------+--------+-------------+-----------+---------------+----------------+

stonks> position update-stop AAPL 165.00
[✓] Updated Stop Loss for AAPL to $165.00
```

---

## 5. Starting the Background Event Runtime

```
stonks> runtime start
Starting background operating runtime engine...
[✓] Background runtime started successfully.

[HEARTBEAT] Uptime: 0s | Jobs: 0 OK / 0 Fail | Memory: 85.20 MB | Queue: 0
[HEARTBEAT] Uptime: 10s | Jobs: 2 OK / 0 Fail | Memory: 85.45 MB | Queue: 0

stonks> runtime metrics

Runtime Health Metrics
+--------------------------+--------+
| Metric Name              | Value  |
+--------------------------+--------+
| Jobs Executed            | 2      |
| Jobs Failed              | 0      |
| Events Published         | 4      |
| Events Processed         | 2      |
| Average Analysis Latency | 1.84s  |
| Queue Depth              | 0      |
+--------------------------+--------+

stonks> exit
Exiting STONKS terminal. Auto-saving active session...
```
