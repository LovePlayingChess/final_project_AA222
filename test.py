from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print the number of active holdings at the latest rebalance for each strategy."
    )
    parser.add_argument(
        "weights_path",
        nargs="?",
        default="results/latest/weights.csv",
        help="Path to a weights.csv file. Defaults to results/latest/weights.csv.",
    )
    return parser


def resolve_weights_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    candidates = sorted(Path("tmp").glob("**/weights.csv"))
    candidate_lines = "\n".join(f"  - {candidate}" for candidate in candidates[:10])
    if len(candidates) > 10:
        candidate_lines += f"\n  - ... and {len(candidates) - 10} more"
    if not candidate_lines:
        candidate_lines = "  - No weights.csv files were found under tmp/"

    raise FileNotFoundError(
        f"Could not find weights file at {path}.\n"
        "Run the backtest first with `python main.py --output-dir results/latest`,\n"
        "or point this script at an existing file, for example:\n"
        "  python test.py tmp/results-live/weights.csv\n"
        "Available candidates:\n"
        f"{candidate_lines}"
    )


def main() -> int:
    args = build_parser().parse_args()
    weights_path = resolve_weights_path(args.weights_path)
    weights = pd.read_csv(weights_path)

    for strategy in weights["strategy"].unique():
        subset = weights[weights["strategy"] == strategy]
        last_rebalance = subset["rebalance_date"].max()
        current = subset[subset["rebalance_date"] == last_rebalance]
        print(strategy, (current["weight"] > 1e-6).sum())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
