from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


@dataclass(frozen=True)
class DataConfig:
    start_date: str
    end_date: str | None
    benchmark: str
    dataset_path: Path
    force_download: bool = False
    max_assets: int = 150
    universe_source: str = "wikipedia"


def load_prices_from_cache(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index.name = "date"
    return prices.sort_index()


def normalize_ticker(ticker: str) -> str:
    return ticker.replace(".", "-").strip().upper()


def fetch_sp500_constituents() -> list[str]:
    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text), match="Symbol")[0]
    tickers = [normalize_ticker(value) for value in table["Symbol"].astype(str)]
    return sorted(set(tickers))


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    import yfinance as yf

    all_tickers = sorted(set(tickers))
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(
        tickers=all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    print("Download complete.")
    if data.empty:
        raise RuntimeError("Yahoo Finance returned an empty dataset.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        elif "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            raise RuntimeError("Expected Close or Adj Close prices in yfinance output.")
    else:
        prices = data.to_frame(name=all_tickers[0])

    prices = prices.sort_index()
    prices.index.name = "date"
    prices.columns = [normalize_ticker(str(col)) for col in prices.columns]
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices


def filter_full_period_assets(
    prices: pd.DataFrame,
    benchmark: str,
    max_assets: int,
) -> pd.DataFrame:
    non_benchmark = [col for col in prices.columns if col != benchmark]
    valid_assets: list[tuple[str, pd.Timestamp]] = []
    for ticker in non_benchmark:
        series = prices[ticker].dropna()
        if series.empty:
            continue
        if series.index[0] > prices.index[0] or series.index[-1] < prices.index[-1]:
            continue
        valid_assets.append((ticker, series.index[0]))

    valid_assets.sort(key=lambda item: (item[1], item[0]))
    selected = [ticker for ticker, _ in valid_assets[:max_assets]]
    if benchmark not in prices.columns:
        raise RuntimeError(f"Benchmark ticker {benchmark} is missing from the dataset.")
    selected_columns = selected + [benchmark]
    filtered = prices[selected_columns].copy()
    filtered = filtered.dropna(axis=0, how="all")
    filtered = filtered.ffill()
    return filtered


def build_universe_dataset(config: DataConfig) -> pd.DataFrame:
    if config.dataset_path.exists() and not config.force_download:
        return load_prices_from_cache(config.dataset_path)

    config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    tickers = fetch_sp500_constituents()
    all_tickers = tickers + [normalize_ticker(config.benchmark)]
    prices = download_prices(
        tickers=all_tickers,
        start_date=config.start_date,
        end_date=config.end_date,
    )
    prices = filter_full_period_assets(
        prices=prices,
        benchmark=normalize_ticker(config.benchmark),
        max_assets=config.max_assets,
    )
    prices.to_csv(config.dataset_path)
    return prices


def summarize_dataset(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    benchmark = normalize_ticker(benchmark)
    for ticker in prices.columns:
        series = prices[ticker]
        non_null = series.dropna()
        if non_null.empty:
            first_date = None
            last_date = None
        else:
            first_date = str(non_null.index.min().date())
            last_date = str(non_null.index.max().date())
        rows.append(
            {
                "ticker": ticker,
                "is_benchmark": ticker == benchmark,
                "first_date": first_date,
                "last_date": last_date,
                "observations": int(non_null.shape[0]),
                "missing_count": int(series.isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["is_benchmark", "ticker"], ascending=[False, True])
