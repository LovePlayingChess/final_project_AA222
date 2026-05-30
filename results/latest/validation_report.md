# Validation Report

## Feedback Mapping
- Rolling estimation is used: `mu_t` and `Sigma_t` are estimated on each rebalance date using only trailing data.
- Covariance uses EWMA with half-life `63.0` and keeps the top `15` eigencomponents.
- Cardinality is handled with the suggested heuristic: solve dense, keep top-`k`, re-solve on the reduced support.
- Volatility targets are swept as multiples of trailing benchmark volatility.
- Minimum history filter is enforced before an asset can enter the tradeable universe on a rebalance date.

## Data Audit
- Universe size including benchmark: `61`
- Benchmark ticker: `SPY`
- Non-benchmark assets: `60`
- Max missing observations for any retained ticker: `0`
- Data source is Yahoo Finance adjusted prices on a fixed current-constituent universe; this is valid for prototyping but still carries survivorship bias.

## Optimization Soundness
- Long-only, fully invested, volatility-constrained optimization is solved at each rebalance.
- Dense infeasible rebalance count: `348`
- Sparse infeasible rebalance count: `348`
- If a requested volatility target is below the minimum-variance portfolio, the code returns the minimum-variance portfolio and records the rebalance as infeasible.