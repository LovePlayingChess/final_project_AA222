# Sparse Index Portfolio Construction

This project implements a simple end-to-end pipeline for sparse index portfolio construction
using rolling optimization and a fixed Yahoo Finance universe.

The implementation follows the project proposal, the status update, and Alex's feedback:

- `mu_t` and `Sigma_t` are estimated on each rebalance date using only information
  available up to time `t`.
- `Sigma_t` uses an EWMA estimate and a low-rank eigenfactor projection that keeps the
  top components and preserves diagonal residual risk.
- The cardinality heuristic is Alex's suggested "street-fighting" method: solve the dense
  convex problem, keep the top-`k` weights, fix the rest to zero, and re-solve.
- The backtest sweeps both the support size `k` and the volatility target relative to the
  trailing benchmark volatility.

## Data assumption

To keep the project tractable, the pipeline uses:

- Current S&P 500 constituents scraped from Wikipedia.
- Adjusted daily prices from Yahoo Finance via `yfinance`.
- A fixed universe filtered to names with full-period price coverage.

This deliberately avoids delisting plumbing but introduces survivorship bias, which is
explicitly acknowledged in `run.json`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
source .venv/bin/activate
python main.py \
  --start-date 2005-01-01 \
  --benchmark SPY \
  --dataset-path data/universe_prices.csv \
  --output-dir results/latest \
  --support-sizes 30,50,100,250 \
  --vol-multipliers 0.8,1.0,1.2
```

## Outputs

- `metrics.csv`: annual return, annual volatility, Sharpe, max drawdown, cumulative return.
- `daily_returns.csv`: daily strategy and benchmark returns.
- `cumulative_returns.csv`: cumulative growth curves.
- `data_audit.csv`: retained tickers, start/end dates, observation counts, missing counts.
- `rebalance_summary.csv`: rebalance-level target risk, achieved risk, feasibility, and holdings count.
- `weights.csv`: nonzero holdings at each rebalance date.
- `equity_curves.png`: top strategies against the benchmark.
- `sharpe_heatmap.png`: Sharpe ratio by support size and volatility target.
- `risk_return_scatter.png`: annualized return vs annualized volatility across strategies and SPY.
- `drawdowns.png`: drawdown comparison for selected sparse portfolios vs SPY.
- `average_allocations_heatmap.png`: average portfolio weights across selected strategies.
- `selection_frequency.png`: how often specific stocks were selected.
- `run.json`: configuration and survivorship-bias note.
- `validation_report.md`: feedback mapping, data notes, and feasibility summary.

## Postprocess Saved Runs

Use the saved CSV outputs to generate a compact paper bundle with five figures and summary tables:

```bash
python -m sparse_index_portfolio.postprocess \
  --results-dir tmp/results-live-10-sweep \
  --dataset-path tmp/live_prices_10_2022.csv \
  --output-dir tmp/paper_bundle_live10
```
