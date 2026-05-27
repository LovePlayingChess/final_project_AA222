from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sparse_index_portfolio.estimation import (
    TRADING_DAYS,
    annualized_volatility,
    estimate_ewma_covariance,
    estimate_ewma_mean,
    low_rank_covariance,
)
from sparse_index_portfolio.optimization import solve_sparse_portfolio


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str
    end_date: str | None
    benchmark: str
    rebalance_frequency: str
    lookback_days: int
    min_history_days: int
    ewma_half_life: float
    top_components: int
    support_sizes: list[int]
    vol_multipliers: list[float]
    dense_l2_reg: float = 1e-6


@dataclass
class SweepResults:
    metrics: pd.DataFrame
    daily_returns: pd.DataFrame
    cumulative_returns: pd.DataFrame
    weights: pd.DataFrame
    rebalance_summary: pd.DataFrame
    benchmark_metrics: pd.DataFrame
    metadata: dict[str, object]


@dataclass
class RebalanceInputs:
    rebalance_date: pd.Timestamp
    next_rebalance_date: pd.Timestamp
    mu: pd.Series
    covariance: pd.DataFrame
    benchmark_volatility: float


def compute_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    if frequency == "M":
        periods = index.to_period("M")
    elif frequency == "Q":
        periods = index.to_period("Q")
    else:
        raise ValueError(f"Unsupported rebalance frequency: {frequency}")
    grouped = pd.Series(index=index, data=index)
    return pd.DatetimeIndex(grouped.groupby(periods).last().tolist())


def compute_metrics(returns: pd.Series) -> dict[str, float]:
    compounded = (1.0 + returns).prod()
    n_days = max(len(returns), 1)
    annual_return = compounded ** (TRADING_DAYS / n_days) - 1.0
    annual_vol = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())
    return {
        "annual_return": float(annual_return),
        "annual_volatility": annual_vol,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "cumulative_return": float(compounded - 1.0),
    }


def run_backtest_sweep(prices: pd.DataFrame, config: BacktestConfig) -> SweepResults:
    benchmark = config.benchmark.upper()
    prices = prices.sort_index().copy()
    prices.columns = [col.upper() for col in prices.columns]
    if benchmark not in prices.columns:
        raise RuntimeError(f"Benchmark {benchmark} is missing from the price table.")

    returns = prices.pct_change().dropna(how="all")
    benchmark_returns = returns[benchmark].dropna()
    asset_returns = returns.drop(columns=[benchmark])
    rebalance_dates = compute_rebalance_dates(asset_returns.index, config.rebalance_frequency)

    if len(rebalance_dates) < 3:
        raise RuntimeError("Not enough rebalance dates for a backtest.")

    strategy_returns: dict[str, pd.Series] = {}
    weights_history: list[pd.DataFrame] = []
    rebalance_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    rebalance_inputs: list[RebalanceInputs] = []

    for idx in range(1, len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates[idx]
        next_rebalance = rebalance_dates[idx + 1]
        history = asset_returns.loc[:rebalance_date].tail(config.lookback_days)
        history = history.dropna(axis=1)
        if len(history) < config.min_history_days:
            continue

        eligible = [
            ticker for ticker in history.columns if history[ticker].count() >= config.min_history_days
        ]
        if len(eligible) <= 2:
            continue

        history = history[eligible].dropna()
        if len(history) < config.min_history_days:
            continue

        mu = estimate_ewma_mean(history=history, half_life=config.ewma_half_life)
        covariance = estimate_ewma_covariance(history=history, half_life=config.ewma_half_life)
        covariance = low_rank_covariance(covariance=covariance, top_components=config.top_components)
        benchmark_window = benchmark_returns.loc[:rebalance_date].tail(config.lookback_days)
        benchmark_vol = annualized_volatility(benchmark_window)
        rebalance_inputs.append(
            RebalanceInputs(
                rebalance_date=rebalance_date,
                next_rebalance_date=next_rebalance,
                mu=mu,
                covariance=covariance,
                benchmark_volatility=benchmark_vol,
            )
        )

    if not rebalance_inputs:
        raise RuntimeError("No rebalance dates met the minimum-history requirement.")

    for support_size in config.support_sizes:
        for vol_multiplier in config.vol_multipliers:
            label = f"k={support_size}|vol={vol_multiplier:.2f}"
            period_returns: list[pd.Series] = []
            previous_sparse_weights: pd.Series | None = None

            for inputs in rebalance_inputs:
                target_vol_daily = (
                    inputs.benchmark_volatility * vol_multiplier
                ) / np.sqrt(TRADING_DAYS)
                dense_result, sparse_result = solve_sparse_portfolio(
                    mu=inputs.mu,
                    covariance=inputs.covariance,
                    target_volatility=target_vol_daily,
                    support_size=min(support_size, len(inputs.mu)),
                    l2_reg=config.dense_l2_reg,
                    initial_weights=previous_sparse_weights,
                )

                previous_sparse_weights = sparse_result.weights.copy()
                forward = asset_returns.loc[
                    (asset_returns.index > inputs.rebalance_date)
                    & (asset_returns.index <= inputs.next_rebalance_date),
                    sparse_result.weights.index,
                ]
                if forward.empty:
                    continue

                strategy_path = forward.fillna(0.0).dot(sparse_result.weights)
                strategy_path.name = label
                period_returns.append(strategy_path)

                rebalance_rows.append(
                    {
                        "strategy": label,
                        "support_size": support_size,
                        "vol_multiplier": vol_multiplier,
                        "rebalance_date": inputs.rebalance_date,
                        "next_rebalance_date": inputs.next_rebalance_date,
                        "target_vol_daily": target_vol_daily,
                        "target_vol_annualized": target_vol_daily * np.sqrt(TRADING_DAYS),
                        "benchmark_vol_annualized": inputs.benchmark_volatility,
                        "dense_realized_vol_daily": dense_result.realized_volatility,
                        "dense_realized_vol_annualized": dense_result.realized_volatility
                        * np.sqrt(TRADING_DAYS),
                        "sparse_realized_vol_daily": sparse_result.realized_volatility,
                        "sparse_realized_vol_annualized": sparse_result.realized_volatility
                        * np.sqrt(TRADING_DAYS),
                        "dense_expected_return_daily": dense_result.objective_value,
                        "sparse_expected_return_daily": sparse_result.objective_value,
                        "dense_feasible": dense_result.feasible,
                        "sparse_feasible": sparse_result.feasible,
                        "num_holdings": int((sparse_result.weights > 1e-8).sum()),
                    }
                )

                snapshot = sparse_result.weights[sparse_result.weights > 1e-8].sort_values(
                    ascending=False
                )
                if not snapshot.empty:
                    weights_frame = snapshot.to_frame("weight")
                    weights_frame["strategy"] = label
                    weights_frame["rebalance_date"] = inputs.rebalance_date
                    weights_frame["dense_expected_return"] = dense_result.objective_value
                    weights_frame["sparse_expected_return"] = sparse_result.objective_value
                    weights_history.append(weights_frame.reset_index(names="ticker"))

            if not period_returns:
                continue

            stitched = pd.concat(period_returns).sort_index()
            stitched = stitched[~stitched.index.duplicated(keep="last")]
            stitched.name = label
            strategy_returns[label] = stitched
            metrics = compute_metrics(stitched)
            metrics["support_size"] = support_size
            metrics["vol_multiplier"] = vol_multiplier
            metrics["strategy"] = label
            summary_rows.append(metrics)

    if not strategy_returns:
        raise RuntimeError("No strategies were produced. Try relaxing the filters.")

    daily_returns = pd.DataFrame(strategy_returns).sort_index()
    daily_returns[benchmark] = benchmark_returns.reindex(daily_returns.index).fillna(0.0)
    cumulative_returns = (1.0 + daily_returns).cumprod()
    weights = pd.concat(weights_history, ignore_index=True) if weights_history else pd.DataFrame()
    rebalance_summary = (
        pd.DataFrame(rebalance_rows).sort_values(by=["strategy", "rebalance_date"])
        if rebalance_rows
        else pd.DataFrame()
    )
    metrics_df = pd.DataFrame(summary_rows).sort_values(
        by=["sharpe", "annual_return"], ascending=[False, False]
    )
    benchmark_metrics = pd.DataFrame(
        [compute_metrics(daily_returns[benchmark])], index=[benchmark]
    )

    return SweepResults(
        metrics=metrics_df,
        daily_returns=daily_returns,
        cumulative_returns=cumulative_returns,
        weights=weights,
        rebalance_summary=rebalance_summary,
        benchmark_metrics=benchmark_metrics,
        metadata={
            "benchmark": benchmark,
            "n_assets": int(asset_returns.shape[1]),
            "n_strategies": int(len(strategy_returns)),
            "data_start": str(prices.index.min().date()),
            "data_end": str(prices.index.max().date()),
            "returns_start": str(daily_returns.index.min().date()),
            "returns_end": str(daily_returns.index.max().date()),
            "survivorship_bias_note": (
                "Uses current S&P 500 constituents from Wikipedia filtered to names with "
                "full-period price coverage in Yahoo Finance. This avoids delisting "
                "plumbing but introduces survivorship bias."
            ),
        },
    )
