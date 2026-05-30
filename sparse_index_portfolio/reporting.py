from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sparse_index_portfolio.backtest import BacktestConfig, SweepResults

def get_benchmark_name_and_sharpe(benchmark_metrics: pd.DataFrame) -> tuple[str, float]:
    benchmark_name = str(benchmark_metrics.index[0])
    benchmark_sharpe = float(benchmark_metrics.iloc[0]["sharpe"])
    return benchmark_name, benchmark_sharpe

def select_comparison_strategies(
    metrics: pd.DataFrame,
    preferred_vol_multiplier: float = 1.0,
    max_count: int = 5,
) -> list[str]:
    if metrics.empty:
        return []
    rows: list[str] = []
    for support_size, group in metrics.groupby("support_size"):
        distances = (group["vol_multiplier"] - preferred_vol_multiplier).abs()
        pick = group.loc[distances.sort_values().index].iloc[0]
        rows.append(str(pick["strategy"]))
    return rows[:max_count]


def write_equity_curve_plot(
    cumulative_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    benchmark: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 7))
    comparison_strategies = select_comparison_strategies(metrics=metrics, max_count=5)
    if not comparison_strategies:
        comparison_strategies = metrics.head(5)["strategy"].tolist()
    for column in comparison_strategies + [benchmark]:
        if column in cumulative_returns.columns:
            linewidth = 2.8 if column == benchmark else 1.8
            alpha = 0.95 if column == benchmark else 0.85
            plt.plot(
                cumulative_returns.index,
                cumulative_returns[column],
                label=column,
                linewidth=linewidth,
                alpha=alpha,
            )
    plt.title("Sparse Portfolio vs Benchmark")
    plt.ylabel("Growth of $1")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_sharpe_heatmap(
    metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    pivot = metrics.pivot(
        index="support_size", columns="vol_multiplier", values="sharpe"
    ).sort_index()
    if pivot.empty:
        return

    plt.figure(figsize=(8, 5))
    image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    plt.colorbar(image, label="Sharpe ratio")
    plt.xticks(range(len(pivot.columns)), [f"{col:.2f}" for col in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(idx) for idx in pivot.index])
    plt.xlabel("Volatility multiplier")
    plt.ylabel("Support size k")
    benchmark_name, benchmark_sharpe = get_benchmark_name_and_sharpe(benchmark_metrics)
    plt.title(f"Sharpe Ratio Sweep\n{benchmark_name} Sharpe = {benchmark_sharpe:.3f}")
    for row_idx, support in enumerate(pivot.index):
        for col_idx, vol in enumerate(pivot.columns):
            value = pivot.loc[support, vol]
            plt.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

def write_sparsity_performance_plot(
    metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    if metrics.empty:
        return

    plt.figure(figsize=(8.5, 5.5))

    for vol_multiplier, group in metrics.groupby("vol_multiplier"):
        group = group.sort_values("support_size")
        plt.plot(
            group["support_size"],
            group["sharpe"],
            marker="o",
            linewidth=2,
            label=f"vol={vol_multiplier:.2f}",
        )
    benchmark_name, benchmark_sharpe = get_benchmark_name_and_sharpe(benchmark_metrics)
    plt.axhline(
        benchmark_sharpe,
        linestyle="--",
        linewidth=1.8,
        label=f"{benchmark_name} Sharpe = {benchmark_sharpe:.3f}",
    )
    plt.xlabel("Support size k")
    plt.ylabel("Sharpe ratio")
    plt.title("Sparsity vs Performance Relative to Benchmark")
    plt.legend(title="Volatility target")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
def write_risk_return_scatter(
    metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8.5, 6))
    scatter = plt.scatter(
        metrics["annual_volatility"],
        metrics["annual_return"],
        c=metrics["support_size"],
        cmap="viridis",
        s=100,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.4,
    )
    plt.colorbar(scatter, label="Support size k")
    for _, row in metrics.iterrows():
        plt.annotate(
            row["strategy"],
            (row["annual_volatility"], row["annual_return"]),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    benchmark = benchmark_metrics.iloc[0]
    plt.scatter(
        [benchmark["annual_volatility"]],
        [benchmark["annual_return"]],
        marker="*",
        s=320,
        color="crimson",
        edgecolor="black",
        linewidth=0.6,
        label=str(benchmark_metrics.index[0]),
    )
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized return")
    plt.title("Risk-Return Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_drawdown_plot(
    cumulative_returns: pd.DataFrame,
    metrics: pd.DataFrame,
    benchmark: str,
    output_path: Path,
) -> None:
    comparison_strategies = select_comparison_strategies(metrics=metrics, max_count=4)
    plt.figure(figsize=(12, 7))
    for column in comparison_strategies + [benchmark]:
        if column not in cumulative_returns.columns:
            continue
        wealth = cumulative_returns[column]
        drawdown = wealth / wealth.cummax() - 1.0
        linewidth = 2.8 if column == benchmark else 1.6
        plt.plot(drawdown.index, drawdown, label=column, linewidth=linewidth)
    plt.axhline(0.0, color="black", linewidth=0.7)
    plt.title("Drawdown Comparison")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_average_weight_heatmap(
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    if weights.empty:
        return
    strategies = select_comparison_strategies(metrics=metrics, max_count=6)
    if not strategies:
        return
    selected = weights[weights["strategy"].isin(strategies)].copy()
    by_date = selected.pivot_table(
        index=["strategy", "rebalance_date"],
        columns="ticker",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )
    pivot = by_date.groupby(level=0).mean()
    top_columns = pivot.mean(axis=0).sort_values(ascending=False).head(12).index
    pivot = pivot.reindex(strategies).loc[:, top_columns]

    plt.figure(figsize=(11, 6.5))
    image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
    colorbar = plt.colorbar(image, label="Average portfolio weight")
    colorbar.ax.text(
        0.5,
        -0.08,
        "Average across rebalances;\n0 when not held",
        ha="center",
        va="top",
        fontsize=8,
        transform=colorbar.ax.transAxes,
    )
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Average Allocations by Strategy")
    plt.tight_layout(rect=(0, 0, 0.96, 1))
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_selection_frequency_plot(
    weights: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    if weights.empty:
        return
    strategies = select_comparison_strategies(metrics=metrics, max_count=4)
    if not strategies:
        return
    selected = weights[weights["strategy"].isin(strategies)].copy()
    rebalance_counts = selected.groupby("strategy")["rebalance_date"].nunique()
    freq = (
        selected.groupby(["strategy", "ticker"])["rebalance_date"]
        .nunique()
        .div(rebalance_counts, level=0)
        .reset_index(name="selection_frequency")
    )
    top = (
        freq.groupby("ticker")["selection_frequency"]
        .max()
        .sort_values(ascending=False)
        .head(10)
        .index
    )
    pivot = freq[freq["ticker"].isin(top)].pivot(
        index="strategy", columns="ticker", values="selection_frequency"
    ).fillna(0.0)
    pivot = pivot.loc[strategies]

    plt.figure(figsize=(11, 6))
    image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="Blues")
    plt.colorbar(image, label="Fraction of rebalances selected")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("Stock Selection Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_metadata(metadata: dict[str, object], config: BacktestConfig, output_path: Path) -> None:
    payload = {
        "config": {
            "start_date": config.start_date,
            "end_date": config.end_date,
            "benchmark": config.benchmark,
            "rebalance_frequency": config.rebalance_frequency,
            "lookback_days": config.lookback_days,
            "min_history_days": config.min_history_days,
            "ewma_half_life": config.ewma_half_life,
            "top_components": config.top_components,
            "support_sizes": config.support_sizes,
            "vol_multipliers": config.vol_multipliers,
            "max_weight": config.max_weight,
            "dense_l2_reg": config.dense_l2_reg,
        },
        "notes": metadata,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def write_validation_report(
    results: SweepResults,
    config: BacktestConfig,
    data_audit: pd.DataFrame,
    output_path: Path,
) -> None:
    infeasible_dense = 0
    infeasible_sparse = 0
    if not results.rebalance_summary.empty:
        infeasible_dense = int((~results.rebalance_summary["dense_feasible"]).sum())
        infeasible_sparse = int((~results.rebalance_summary["sparse_feasible"]).sum())

    lines = [
        "# Validation Report",
        "",
        "## Feedback Mapping",
        "- Rolling estimation is used: `mu_t` and `Sigma_t` are estimated on each rebalance date using only trailing data.",
        f"- Covariance uses EWMA with half-life `{config.ewma_half_life}` and keeps the top `{config.top_components}` eigencomponents.",
        "- Cardinality is handled with the suggested heuristic: solve dense, keep top-`k`, re-solve on the reduced support.",
        "- Volatility targets are swept as multiples of trailing benchmark volatility.",
        "- Minimum history filter is enforced before an asset can enter the tradeable universe on a rebalance date.",
        "",
        "## Data Audit",
        f"- Universe size including benchmark: `{len(data_audit)}`",
        f"- Benchmark ticker: `{config.benchmark}`",
        f"- Non-benchmark assets: `{int((~data_audit['is_benchmark']).sum())}`",
        f"- Max missing observations for any retained ticker: `{int(data_audit['missing_count'].max())}`",
        "- Data source is Yahoo Finance adjusted prices on a fixed current-constituent universe; this is valid for prototyping but still carries survivorship bias.",
        "",
        "## Optimization Soundness",
        "- Long-only, fully invested, volatility-constrained optimization is solved at each rebalance.",
        f"- Dense infeasible rebalance count: `{infeasible_dense}`",
        f"- Sparse infeasible rebalance count: `{infeasible_sparse}`",
        "- If a requested volatility target is below the minimum-variance portfolio, the code returns the minimum-variance portfolio and records the rebalance as infeasible.",
    ]
    output_path.write_text("\n".join(lines))

def write_benchmark_comparison_table(
    metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    best = metrics.sort_values("sharpe", ascending=False).iloc[0]
    benchmark = benchmark_metrics.iloc[0]
    benchmark_name = str(benchmark_metrics.index[0])

    table = pd.DataFrame(
        [
            {
                "portfolio": benchmark_name,
                "annual_return": benchmark["annual_return"],
                "annual_volatility": benchmark["annual_volatility"],
                "sharpe": benchmark["sharpe"],
                "max_drawdown": benchmark["max_drawdown"],
            },
            {
                "portfolio": f"Best optimized ({best['strategy']})",
                "annual_return": best["annual_return"],
                "annual_volatility": best["annual_volatility"],
                "sharpe": best["sharpe"],
                "max_drawdown": best["max_drawdown"],
            },
        ]
    )

    table.to_csv(output_path, index=False)
    return table

def write_report(
    results: SweepResults,
    output_dir: Path,
    config: BacktestConfig,
    data_audit: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_with_benchmark = pd.concat([results.metrics, results.benchmark_metrics], axis=0)
    metrics_with_benchmark.to_csv(output_dir / "metrics.csv", index=True)
    results.daily_returns.to_csv(output_dir / "daily_returns.csv", index=True)
    results.cumulative_returns.to_csv(output_dir / "cumulative_returns.csv", index=True)
    data_audit.to_csv(output_dir / "data_audit.csv", index=False)
    if not results.weights.empty:
        results.weights.to_csv(output_dir / "weights.csv", index=False)
    if not results.rebalance_summary.empty:
        results.rebalance_summary.to_csv(output_dir / "rebalance_summary.csv", index=False)

    benchmark = str(results.metadata["benchmark"])
    write_equity_curve_plot(
        cumulative_returns=results.cumulative_returns,
        metrics=results.metrics,
        benchmark=benchmark,
        output_path=output_dir / "equity_curves.png",
    )
    write_sharpe_heatmap(
        metrics=results.metrics,
        benchmark_metrics=results.benchmark_metrics,
        output_path=output_dir / "sharpe_heatmap.png",
    )
    write_sparsity_performance_plot(
        metrics=results.metrics,
        benchmark_metrics=results.benchmark_metrics,
        output_path=output_dir / "sparsity_vs_performance.png",
    )
    write_risk_return_scatter(
        metrics=results.metrics,
        benchmark_metrics=results.benchmark_metrics,
        output_path=output_dir / "risk_return_scatter.png",
    )
    write_drawdown_plot(
        cumulative_returns=results.cumulative_returns,
        metrics=results.metrics,
        benchmark=benchmark,
        output_path=output_dir / "drawdowns.png",
    )
    write_average_weight_heatmap(
        weights=results.weights,
        metrics=results.metrics,
        output_path=output_dir / "average_allocations_heatmap.png",
    )
    write_selection_frequency_plot(
        weights=results.weights,
        metrics=results.metrics,
        output_path=output_dir / "selection_frequency.png",
    )
    write_metadata(metadata=results.metadata, config=config, output_path=output_dir / "run.json")
    write_validation_report(
        results=results,
        config=config,
        data_audit=data_audit,
        output_path=output_dir / "validation_report.md",
    )

    comparison_table = write_benchmark_comparison_table(
        metrics=results.metrics,
        benchmark_metrics=results.benchmark_metrics,
        output_path=output_dir / "benchmark_comparison.csv",
    )

    print("\nBenchmark comparison:")
    print(comparison_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
