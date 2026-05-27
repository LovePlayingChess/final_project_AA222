from __future__ import annotations

import argparse
from pathlib import Path

from sparse_index_portfolio.backtest import BacktestConfig, run_backtest_sweep
from sparse_index_portfolio.data import DataConfig, build_universe_dataset, summarize_dataset
from sparse_index_portfolio.reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sparse index portfolio construction with rolling optimization."
    )
    parser.add_argument("--start-date", default="2005-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument(
        "--dataset-path",
        default="data/universe_prices.csv",
        help="CSV cache used for downloaded adjusted close data.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/latest",
        help="Directory for metrics, weights, returns, and plots.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        default="M",
        choices=["M", "Q"],
        help="Monthly or quarterly rebalancing.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=252 * 2,
        help="Historical returns used to estimate mu_t and Sigma_t.",
    )
    parser.add_argument(
        "--min-history-days",
        type=int,
        default=126,
        help="An asset must have at least this much history to be tradeable.",
    )
    parser.add_argument(
        "--ewma-half-life",
        type=float,
        default=63.0,
        help="EWMA half-life used for the mean and covariance estimators.",
    )
    parser.add_argument(
        "--top-components",
        type=int,
        default=15,
        help="Number of covariance eigencomponents retained.",
    )
    parser.add_argument(
        "--support-sizes",
        default="30,50,100,250",
        help="Comma-separated cardinalities k.",
    )
    parser.add_argument(
        "--vol-multipliers",
        default="0.8,1.0,1.2",
        help="Comma-separated multipliers on trailing benchmark volatility.",
    )
    parser.add_argument(
        "--dense-l2-reg",
        type=float,
        default=1e-6,
        help="Small ridge term to stabilize optimization.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only fetch the fixed-universe dataset and exit.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Ignore the cached CSV and re-download the dataset.",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=150,
        help=(
            "Maximum universe size after coverage filtering. "
            "The oldest assets by start-date coverage are kept."
        ),
    )
    parser.add_argument(
        "--universe-source",
        default="wikipedia",
        choices=["wikipedia"],
        help="Source used to fetch the current S&P 500 constituent list.",
    )
    return parser


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    data_config = DataConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        dataset_path=dataset_path,
        force_download=args.force_download,
        max_assets=args.max_assets,
        universe_source=args.universe_source,
    )
    prices = build_universe_dataset(data_config)
    data_audit = summarize_dataset(prices=prices, benchmark=args.benchmark)

    if args.download_only:
        print(f"Downloaded dataset to {dataset_path}")
        return 0

    backtest_config = BacktestConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        benchmark=args.benchmark,
        rebalance_frequency=args.rebalance_frequency,
        lookback_days=args.lookback_days,
        min_history_days=args.min_history_days,
        ewma_half_life=args.ewma_half_life,
        top_components=args.top_components,
        support_sizes=parse_int_list(args.support_sizes),
        vol_multipliers=parse_float_list(args.vol_multipliers),
        dense_l2_reg=args.dense_l2_reg,
    )
    results = run_backtest_sweep(prices=prices, config=backtest_config)
    write_report(
        results=results,
        output_dir=output_dir,
        config=backtest_config,
        data_audit=data_audit,
    )
    print(f"Wrote results to {output_dir}")
    return 0
