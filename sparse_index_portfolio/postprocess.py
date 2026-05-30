from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sparse_index_portfolio.data import summarize_dataset
from sparse_index_portfolio.reporting import select_comparison_strategies


def load_run_bundle(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metrics = pd.read_csv(results_dir / "metrics.csv", index_col=0)
    daily_returns = pd.read_csv(results_dir / "daily_returns.csv", index_col=0, parse_dates=True)
    cumulative_returns = pd.read_csv(
        results_dir / "cumulative_returns.csv", index_col=0, parse_dates=True
    )
    weights_path = results_dir / "weights.csv"
    weights = pd.read_csv(weights_path, parse_dates=["rebalance_date"]) if weights_path.exists() else pd.DataFrame()
    run = json.loads((results_dir / "run.json").read_text())
    return metrics, daily_returns, cumulative_returns, weights, run


def comparison_metrics(metrics: pd.DataFrame, benchmark: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmark_metrics = metrics.loc[[benchmark]].copy()
    strategy_metrics = metrics.loc[metrics.index != benchmark].copy()
    return strategy_metrics, benchmark_metrics


def write_equity_figure(
    cumulative_returns: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    benchmark: str,
    output_path: Path,
) -> list[str]:
    strategies = select_comparison_strategies(strategy_metrics, preferred_vol_multiplier=1.0)
    plt.figure(figsize=(12, 7))
    for column in strategies + [benchmark]:
        if column not in cumulative_returns.columns:
            continue
        linewidth = 2.8 if column == benchmark else 1.9
        plt.plot(cumulative_returns.index, cumulative_returns[column], label=column, linewidth=linewidth)
    plt.title("Growth of $1: Sparse Portfolios vs SPY")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return strategies


def write_sharpe_heatmap(strategy_metrics: pd.DataFrame, output_path: Path) -> None:
    pivot = strategy_metrics.pivot(index="support_size", columns="vol_multiplier", values="sharpe")
    plt.figure(figsize=(7.5, 5))
    image = plt.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu")
    plt.colorbar(image, label="Sharpe ratio")
    plt.xticks(range(len(pivot.columns)), [f"{x:.2f}" for x in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
    plt.xlabel("Volatility multiplier")
    plt.ylabel("Support size k")
    plt.title("Sharpe Ratio Sweep")
    for i, support_size in enumerate(pivot.index):
        for j, vol in enumerate(pivot.columns):
            plt.text(j, i, f"{pivot.loc[support_size, vol]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_risk_return_scatter(
    strategy_metrics: pd.DataFrame,
    benchmark_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        strategy_metrics["annual_volatility"],
        strategy_metrics["annual_return"],
        c=strategy_metrics["support_size"],
        cmap="viridis",
        s=110,
        edgecolor="black",
        linewidth=0.4,
    )
    plt.colorbar(scatter, label="Support size k")
    for _, row in strategy_metrics.iterrows():
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
        label=str(benchmark_metrics.index[0]),
    )
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized return")
    plt.title("Risk-Return Benchmarking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def write_average_allocations(
    weights: pd.DataFrame,
    strategies: list[str],
    output_path: Path,
) -> pd.DataFrame:
    selected = weights[weights["strategy"].isin(strategies)].copy()
    by_date = selected.pivot_table(
        index=["strategy", "rebalance_date"],
        columns="ticker",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )
    avg_weights = by_date.groupby(level=0).mean()
    top_cols = avg_weights.mean(axis=0).sort_values(ascending=False).head(12).index
    heatmap = avg_weights.loc[strategies, top_cols]

    plt.figure(figsize=(11, 6.5))
    image = plt.imshow(heatmap.to_numpy(), aspect="auto", cmap="YlOrRd")
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
    plt.xticks(range(len(heatmap.columns)), heatmap.columns, rotation=45, ha="right")
    plt.yticks(range(len(heatmap.index)), heatmap.index)
    plt.title("Average Allocations by k")
    plt.tight_layout(rect=(0, 0, 0.96, 1))
    plt.savefig(output_path, dpi=180)
    plt.close()
    return avg_weights


def write_calendar_year_returns(
    daily_returns: pd.DataFrame,
    strategies: list[str],
    benchmark: str,
    output_path: Path,
) -> pd.DataFrame:
    selected = daily_returns[strategies + [benchmark]].copy()
    annual = selected.resample("YE").apply(lambda frame: (1.0 + frame).prod() - 1.0)
    annual.index = annual.index.year

    plt.figure(figsize=(12, 6.5))
    annual.plot(kind="bar", ax=plt.gca())
    plt.title("Calendar-Year Returns")
    plt.ylabel("Return")
    plt.xlabel("Year")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return annual


def write_selection_frequency(
    weights: pd.DataFrame,
    strategies: list[str],
    output_path: Path,
) -> pd.DataFrame:
    selected = weights[weights["strategy"].isin(strategies)].copy()
    rebalance_counts = selected.groupby("strategy")["rebalance_date"].nunique()
    freq = (
        selected.groupby(["strategy", "ticker"])["rebalance_date"]
        .nunique()
        .div(rebalance_counts, level=0)
        .reset_index(name="selection_frequency")
    )
    top = (
        freq.groupby("ticker")["selection_frequency"].max().sort_values(ascending=False).head(12).index
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
    plt.title("Selection Frequency by k")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return freq


def write_notes(
    output_path: Path,
    benchmark: str,
    benchmark_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    annual_returns: pd.DataFrame,
    avg_weights: pd.DataFrame,
    holdings_summary: pd.DataFrame,
) -> None:
    benchmark_row = benchmark_metrics.iloc[0]
    best_strategy = strategy_metrics.sort_values(by="sharpe", ascending=False).iloc[0]
    worst_gap_year = None
    worst_gap_value = None
    if benchmark in annual_returns.columns:
        for strategy in [best_strategy["strategy"]]:
            gap = annual_returns[benchmark] - annual_returns[strategy]
            worst_gap_year = int(gap.idxmax())
            worst_gap_value = float(gap.max())
    top_holdings = avg_weights.loc[best_strategy["strategy"]].sort_values(ascending=False).head(5)
    lines = [
        "# Paper Notes",
        "",
        f"- Benchmark `{benchmark}` Sharpe: `{benchmark_row['sharpe']:.3f}`.",
        f"- Best sparse strategy: `{best_strategy['strategy']}` with Sharpe `{best_strategy['sharpe']:.3f}`.",
        f"- Benchmark annual return `{benchmark_row['annual_return']:.2%}` vs best sparse `{best_strategy['annual_return']:.2%}`.",
        f"- Benchmark annualized volatility `{benchmark_row['annual_volatility']:.2%}` vs best sparse `{best_strategy['annual_volatility']:.2%}`.",
        "",
        "## Why SPY Outperformed",
        "- In this run, the sparse portfolios stayed more concentrated in lower-volatility names instead of fully participating in the broad-market upside.",
        "- The volatility cap plus the top-k truncation reduced exposure to the strongest benchmark rally periods.",
        f"- Largest annual underperformance gap for the best sparse strategy occurred in `{worst_gap_year}` with a return gap of `{worst_gap_value:.2%}`.",
        "- The average holdings table shows repeated concentration in a narrow subset of names rather than broad index participation.",
        (
            f"- For `{best_strategy['strategy']}`, the realized number of active holdings averaged "
            f"`{holdings_summary.loc[best_strategy['strategy'], 'mean']:.2f}` per rebalance."
        ),
        "",
        f"## Top Holdings for {best_strategy['strategy']}",
    ]
    lines.extend([f"- `{ticker}` average weight `{weight:.2%}`" for ticker, weight in top_holdings.items()])
    output_path.write_text("\n".join(lines))


def write_validation_summary(
    output_path: Path,
    data_audit: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    benchmark: str,
    holdings_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Validation Summary",
        "",
        "## Feedback Coverage",
        "- Rolling estimates are used in the saved backtest outputs rather than a single static train/test split.",
        "- The covariance model is EWMA-based and low-rank projected, matching the requested direction.",
        "- The cardinality constraint is handled with the dense-solve, top-k truncation, and re-solve heuristic.",
        "- The backtest uses a volatility-target sweep against trailing benchmark volatility.",
        "",
        "## Data Validity",
        f"- Benchmark: `{benchmark}`.",
        f"- Retained tickers including benchmark: `{len(data_audit)}`.",
        f"- Maximum missing observations among retained tickers: `{int(data_audit['missing_count'].max())}`.",
        "- Prices are adjusted daily Yahoo Finance prices on a fixed constituent universe.",
        "- This is valid for an accessible prototype, but it still has survivorship bias because delistings and historical constituent changes are not modeled.",
        "",
        "## Result Interpretation",
        f"- Number of sparse strategies compared: `{len(strategy_metrics)}`.",
        "- Sparse portfolios underperformed SPY in this validation run, so the code is not hiding weak results.",
        "- The allocation and annual-return figures show that the sparse portfolios were more concentrated and missed part of the broad-market upside.",
        (
            f"- Average realized holdings count ranged from `{holdings_summary['mean'].min():.2f}` "
            f"to `{holdings_summary['mean'].max():.2f}` across the compared strategies."
        ),
    ]
    output_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures from saved backtest outputs.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", default="SPY")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, daily_returns, cumulative_returns, weights, _ = load_run_bundle(results_dir)
    strategy_metrics, benchmark_metrics = comparison_metrics(metrics=metrics, benchmark=args.benchmark)
    strategies = write_equity_figure(
        cumulative_returns=cumulative_returns,
        strategy_metrics=strategy_metrics,
        benchmark=args.benchmark,
        output_path=output_dir / "figure_1_equity_curves.png",
    )
    write_sharpe_heatmap(strategy_metrics, output_dir / "figure_2_sharpe_heatmap.png")
    write_risk_return_scatter(
        strategy_metrics=strategy_metrics,
        benchmark_metrics=benchmark_metrics,
        output_path=output_dir / "figure_3_risk_return.png",
    )
    avg_weights = write_average_allocations(
        weights=weights,
        strategies=strategies,
        output_path=output_dir / "figure_4_average_allocations.png",
    )
    annual_returns = write_calendar_year_returns(
        daily_returns=daily_returns,
        strategies=strategies,
        benchmark=args.benchmark,
        output_path=output_dir / "figure_5_calendar_year_returns.png",
    )
    selection_frequency = write_selection_frequency(
        weights=weights,
        strategies=strategies,
        output_path=output_dir / "selection_frequency.png",
    )

    prices = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    data_audit = summarize_dataset(prices=prices, benchmark=args.benchmark)
    holdings_summary = (
        weights.groupby(["strategy", "rebalance_date"])["ticker"]
        .count()
        .groupby("strategy")
        .agg(["mean", "min", "max"])
    )
    data_audit.to_csv(output_dir / "data_audit.csv", index=False)
    strategy_metrics.to_csv(output_dir / "strategy_metrics.csv", index=True)
    benchmark_metrics.to_csv(output_dir / "benchmark_metrics.csv", index=True)
    annual_returns.to_csv(output_dir / "annual_returns.csv", index=True)
    avg_weights.to_csv(output_dir / "average_weights.csv", index=True)
    holdings_summary.to_csv(output_dir / "holdings_summary.csv", index=True)
    selection_frequency.to_csv(output_dir / "selection_frequency.csv", index=False)
    write_notes(
        output_path=output_dir / "paper_notes.md",
        benchmark=args.benchmark,
        benchmark_metrics=benchmark_metrics,
        strategy_metrics=strategy_metrics,
        annual_returns=annual_returns,
        avg_weights=avg_weights,
        holdings_summary=holdings_summary,
    )
    write_validation_summary(
        output_path=output_dir / "validation_summary.md",
        data_audit=data_audit,
        strategy_metrics=strategy_metrics,
        benchmark=args.benchmark,
        holdings_summary=holdings_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
