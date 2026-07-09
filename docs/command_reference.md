# STONKS CLI Command Reference

This document provides a comprehensive command reference for the STONKS Interactive Terminal.

---

## 1. Market Namespace (`market`)

Commands for price analysis, sentiment indexing, news retrieval, and technical explainers.

### `market analyze`
* **Description**: Runs the data, sentiment, and CatBoost prediction pipelines for a ticker, outputting a complete Markdown analysis report.
* **Syntax**: `market analyze <TICKER>`
* **Arguments**:
  - `TICKER`: Asset ticker symbol (e.g. AAPL, TSLA).
* **Example**: `market analyze NVDA`

### `market compare`
* **Description**: Renders a side-by-side comparison matrix comparing calibrated prediction scores, sentiment ratios, and risk bounds for two assets.
* **Syntax**: `market compare <TICKER1> <TICKER2>`
* **Example**: `market compare AAPL MSFT`

### `market explain`
* **Description**: Outputs a detailed technical explainer report describing why a buy/sell/hold consensus signal was generated.
* **Syntax**: `market explain <TICKER>`
* **Example**: `market explain GOOGL`

### `market news`
* **Description**: Retrieves and displays recent news titles, publishers, and links for a symbol.
* **Syntax**: `market news <TICKER>`
* **Example**: `market news TSLA`

### `market inspect`
* **Description**: Dumps the raw prediction pipeline output payload in JSON format.
* **Syntax**: `market inspect <TICKER>`
* **Example**: `market inspect AAPL`

### `market research`
* **Description**: Outputs the five most recent daily candles (Open, High, Low, Close, Volume) for a symbol.
* **Syntax**: `market research <TICKER>`
* **Example**: `market research MSFT`

---

## 2. Position Namespace (`position`)

Commands for managing mock trading positions locally inside the Session Manager.

### `position list`
* **Description**: Lists all open long and short positions, showing quantities, entry prices, stops, targets, and live unrealized P/L.
* **Syntax**: `position list`
* **Example**: `position list`

### `position open long`
* **Description**: Opens a new LONG position (prompts for quantity and price if omitted).
* **Syntax**: `position open long <TICKER> [quantity] [entry_price]`
* **Example**: `position open long AAPL 50 175.50`

### `position open short`
* **Description**: Opens a new SHORT position.
* **Syntax**: `position open short <TICKER> [quantity] [entry_price]`
* **Example**: `position open short TSLA 10 210.00`

### `position close`
* **Description**: Liquidates an active position entirely at the exit price (prompts if price is omitted).
* **Syntax**: `position close <TICKER> [exit_price]`
* **Example**: `position close AAPL 182.00`

### `position reduce`
* **Description**: Partially closes a percentage of an active position.
* **Syntax**: `position reduce <TICKER> <PERCENT> [exit_price]`
* **Example**: `position reduce AAPL 50% 180.00`

### `position increase`
* **Description**: Scales-in to an active position, averaging the entry price.
* **Syntax**: `position increase <TICKER> [quantity] [entry_price]`
* **Example**: `position increase AAPL 25 178.00`

### `position review`
* **Description**: Displays a detailed panel showing the date opened, stop loss, take profit, and realized P/L of a position.
* **Syntax**: `position review <TICKER>`
* **Example**: `position review AAPL`

### `position update-stop`
* **Description**: Updates the stop loss price boundary for an active position (type `none` to remove).
* **Syntax**: `position update-stop <TICKER> <PRICE/none>`
* **Example**: `position update-stop AAPL 165.00`

### `position update-target`
* **Description**: Updates the take profit price boundary for an active position.
* **Syntax**: `position update-target <TICKER> <PRICE/none>`
* **Example**: `position update-target AAPL 195.00`

---

## 3. Portfolio Namespace (`portfolio`)

Commands for tracking balances, exposures, allocations, and realized metrics.

### `portfolio summary`
* **Description**: Renders a panel outlining cash, buying power, equity, and realized/unrealized P/L.
* **Syntax**: `portfolio summary`
* **Example**: `portfolio summary`

### `portfolio exposure`
* **Description**: Outputs dollar totals for Long Exposure, Short Exposure, and Net Exposure.
* **Syntax**: `portfolio exposure`
* **Example**: `portfolio exposure`

### `portfolio sectors`
* **Description**: Renders a table detailing percentage allocations across sectors.
* **Syntax**: `portfolio sectors`
* **Example**: `portfolio sectors`

### `portfolio risk`
* **Description**: Audits risk variables, highlighting missing stop-losses and concentration alerts ($>30\%$ total equity).
* **Syntax**: `portfolio risk`
* **Example**: `portfolio risk`

### `portfolio history`
* **Description**: Lists historical completed trades and their realized P/L.
* **Syntax**: `portfolio history`
* **Example**: `portfolio history`

---

## 4. Watchlist Namespace (`watchlist`)

Commands for managing symbols tracked on watchlist files.

### `watchlist list`
* **Description**: Lists all watchlist categories and their tracked symbols.
* **Syntax**: `watchlist list`
* **Example**: `watchlist list`

### `watchlist create`
* **Description**: Adds a new named watchlist category.
* **Syntax**: `watchlist create <NAME>`
* **Example**: `watchlist create Tech`

### `watchlist delete`
* **Description**: Deletes an entire watchlist category.
* **Syntax**: `watchlist delete <NAME>`
* **Example**: `watchlist delete Tech`

### `watchlist add`
* **Description**: Tracks a symbol in a named watchlist category.
* **Syntax**: `watchlist add <NAME> <TICKER> [notes] [priority] [target_price]`
* **Example**: `watchlist add Tech NVDA "Breakout candidate" 3 120.00`

### `watchlist remove`
* **Description**: Untracks a symbol from a named watchlist category.
* **Syntax**: `watchlist remove <NAME> <TICKER>`
* **Example**: `watchlist remove Tech NVDA`

### `watchlist rename`
* **Description**: Renames an existing watchlist category.
* **Syntax**: `watchlist rename <OLD_NAME> <NEW_NAME>`
* **Example**: `watchlist rename Tech Technology`

---

## 5. Research Namespace (`research`)

Commands for auditing backtesting scores, features, and model weights.

### `research benchmark`
* **Description**: Displays the model leaderboard of the six classical classifiers evaluated in walk-forward sweeps.
* **Syntax**: `research benchmark`
* **Example**: `research benchmark`

### `research thresholds`
* **Description**: Outputs the active decision routing thresholds.
* **Syntax**: `research thresholds`
* **Example**: `research thresholds`

### `research features`
* **Description**: Lists categories and names of the 42 engineered indicators.
* **Syntax**: `research features`
* **Example**: `research features`

### `research models`
* **Description**: Lists registered model wrappers available in the registry.
* **Syntax**: `research models`
* **Example**: `research models`

### `research importance`
* **Description**: Renders top 10 feature importances of the best-performing model.
* **Syntax**: `research importance`
* **Example**: `research importance`

---

## 6. Runtime Namespace (`runtime`)

Commands for controlling and monitoring the background event-driven orchestrator.

### `runtime start`
* **Description**: Spawns the scheduler, heartbeat, and worker pools asynchronously.
* **Syntax**: `runtime start`
* **Example**: `runtime start`

### `runtime stop`
* **Description**: Gracefully stops the background runtime orchestrator thread pool.
* **Syntax**: `runtime stop`
* **Example**: `runtime stop`

### `runtime status`
* **Description**: Displays active runtime status state (uptime, workers count, queue size).
* **Syntax**: `runtime status`
* **Example**: `runtime status`

### `runtime metrics`
* **Description**: Renders a table of completed jobs, failed tasks, queue depth, and speeds.
* **Syntax**: `runtime metrics`
* **Example**: `runtime metrics`

### `runtime jobs`
* **Description**: Lists active tasks in the scheduler queue and their next trigger times.
* **Syntax**: `runtime jobs`
* **Example**: `runtime jobs`

### `runtime events`
* **Description**: Outputs registered event subscriptions and counts.
* **Syntax**: `runtime events`
* **Example**: `runtime events`

### `runtime heartbeat`
* **Description**: Displays system memory RSS statistics and agent health status.
* **Syntax**: `runtime heartbeat`
* **Example**: `runtime heartbeat`

### `runtime config`
* **Description**: Displays task intervals scheduler configurations.
* **Syntax**: `runtime config`
* **Example**: `runtime config`

---

## 7. Session Namespace (`session`)

Commands for session status queries, serialization, and resets.

### `session status`
* **Description**: Summarizes session filepath, schema version, and active counts.
* **Syntax**: `session status`
* **Example**: `session status`

### `session save`
* **Description**: Forces an atomic serialization write of the active session to disk.
* **Syntax**: `session save`
* **Example**: `session save`

### `session reload`
* **Description**: Forces state reload from `session.json`.
* **Syntax**: `session reload`
* **Example**: `session reload`

### `session reset`
* **Description**: Resets the session back to clean initial defaults (prompts for confirmation).
* **Syntax**: `session reset`
* **Example**: `session reset`

---

## 8. Profile Namespace (`profile`)

Commands for editing usernames, default capital limits, and trading preferences.

### `profile show`
* **Description**: Displays name, style, risk, timezone, and default capital.
* **Syntax**: `profile show`
* **Example**: `profile show`

### `profile edit`
* **Description**: Edits profile variables.
* **Syntax**: `profile edit <FIELD> <VALUE>`
* **Arguments**:
  - `FIELD`: Field name (`username`, `style`, `timezone`, `currency`).
  - `VALUE`: Value override.
* **Example**: `profile edit username "John Doe"`

### `profile risk`
* **Description**: Modifies the active risk profile boundary.
* **Syntax**: `profile risk <Conservative/Balanced/Aggressive>`
* **Example**: `profile risk Aggressive`

### `profile capital`
* **Description**: Modifies the default budget allocation bounds.
* **Syntax**: `profile capital <AMOUNT>`
* **Example**: `profile capital 250000`

### `profile preferences`
* **Description**: Renders system preferences or edits a parameter (theme, preferred model, language).
* **Syntax**: `profile preferences [FIELD VALUE]`
* **Example**: `profile preferences theme emerald`

---

## 9. Alerts Namespace (`alerts`)

Commands for listing and clearing stop-loss alerts and target limits.

### `alerts list`
* **Description**: Renders a table of logged trigger events.
* **Syntax**: `alerts list`
* **Example**: `alerts list`

### `alerts acknowledge`
* **Description**: Marks active alerts as acknowledged.
* **Syntax**: `alerts acknowledge`
* **Example**: `alerts acknowledge`

### `alerts clear`
* **Description**: Clears the entire alerts history.
* **Syntax**: `alerts clear`
* **Example**: `alerts clear`

---

## 10. System Namespace

Global commands for shell settings and navigation.

* **`help`**: General syntax overview or `help <namespace>` for subcommand details.
* **`clear`**: Clears the console.
* **`version`**: Displays the active version.
* **`exit` / `quit`**: Gracefully terminates the REPL loop, stopping the background runtime and auto-saving session states.
