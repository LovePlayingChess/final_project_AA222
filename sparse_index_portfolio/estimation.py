from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def ewma_weights(length: int, half_life: float) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    decay = math.exp(math.log(0.5) / half_life)
    powers = np.arange(length - 1, -1, -1, dtype=float)
    weights = np.power(decay, powers)
    weights /= weights.sum()
    return weights


def estimate_ewma_mean(history: pd.DataFrame, half_life: float) -> pd.Series:
    weights = ewma_weights(len(history), half_life)
    values = history.to_numpy()
    mean = values.T @ weights
    return pd.Series(mean, index=history.columns)


def estimate_ewma_covariance(history: pd.DataFrame, half_life: float) -> pd.DataFrame:
    weights = ewma_weights(len(history), half_life)
    values = history.to_numpy()
    mean = values.T @ weights
    centered = values - mean
    covariance = centered.T @ (centered * weights[:, None])
    covariance = 0.5 * (covariance + covariance.T)
    return pd.DataFrame(covariance, index=history.columns, columns=history.columns)


def low_rank_covariance(covariance: pd.DataFrame, top_components: int) -> pd.DataFrame:
    matrix = covariance.to_numpy()
    eigvals, eigvecs = np.linalg.eigh(matrix)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = min(top_components, len(eigvals))
    kept_vals = np.clip(eigvals[:keep], 0.0, None)
    kept_vecs = eigvecs[:, :keep]
    factor = kept_vecs @ np.diag(kept_vals) @ kept_vecs.T
    residual_diag = np.clip(np.diag(matrix - factor), 1e-10, None)
    shrunk = factor + np.diag(residual_diag)
    shrunk = 0.5 * (shrunk + shrunk.T)
    return pd.DataFrame(shrunk, index=covariance.index, columns=covariance.columns)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS))

