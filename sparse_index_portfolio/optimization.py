from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    weights: pd.Series
    realized_volatility: float
    objective_value: float
    status: str
    feasible: bool


def portfolio_variance(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(weights @ covariance @ weights)


def solve_min_variance(
    covariance: pd.DataFrame,
    l2_reg: float = 1e-8,
    max_weight: float = 0.05,
) -> OptimizationResult:
    tickers = list(covariance.columns)
    n_assets = len(tickers)
    cov = covariance.to_numpy()

    def objective(weights: np.ndarray) -> float:
        return portfolio_variance(weights, cov) + l2_reg * float(weights @ weights)

    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]
    bounds = [(0.0, max_weight)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)
    solution = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not solution.success:
        raise RuntimeError(f"Min-variance optimization failed: {solution.message}")

    weights = np.clip(solution.x, 0.0, max_weight)
    weights /= weights.sum()
    variance = portfolio_variance(weights, cov)
    return OptimizationResult(
        weights=pd.Series(weights, index=tickers),
        realized_volatility=float(np.sqrt(max(variance, 0.0))),
        objective_value=-variance,
        status=solution.message,
        feasible=True,
    )


def solve_max_return_under_risk(
    mu: pd.Series,
    covariance: pd.DataFrame,
    target_volatility: float,
    l2_reg: float = 1e-6,
    initial_weights: pd.Series | None = None,
    max_weight: float = 0.05,
) -> OptimizationResult:
    tickers = list(mu.index)
    n_assets = len(tickers)
    mu_vec = mu.to_numpy()
    cov = covariance.loc[tickers, tickers].to_numpy()

    min_var_result = solve_min_variance(
        covariance=covariance,
        l2_reg=l2_reg,
        max_weight=max_weight,
    )
    if min_var_result.realized_volatility > target_volatility + 1e-10:
        min_var_result.objective_value = float(mu_vec @ min_var_result.weights.to_numpy())
        min_var_result.status = "Infeasible target volatility; returned minimum-variance portfolio."
        min_var_result.feasible = False
        return min_var_result

    def objective(weights: np.ndarray) -> float:
        return -float(mu_vec @ weights) + l2_reg * float(weights @ weights)

    constraints = [
        {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        {
            "type": "ineq",
            "fun": lambda weights: target_volatility**2 - portfolio_variance(weights, cov),
        },
    ]
    bounds = [(0.0, max_weight)] * n_assets
    x0 = np.full(n_assets, 1.0 / n_assets)
    if initial_weights is not None:
        aligned = initial_weights.reindex(tickers).fillna(0.0).to_numpy()
        if aligned.sum() > 0:
            x0 = aligned / aligned.sum()

    solution = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not solution.success:
        raise RuntimeError(f"Risk-constrained optimization failed: {solution.message}")

    weights = np.clip(solution.x, 0.0, max_weight)
    weights /= weights.sum()
    variance = portfolio_variance(weights, cov)
    return OptimizationResult(
        weights=pd.Series(weights, index=tickers),
        realized_volatility=float(np.sqrt(max(variance, 0.0))),
        objective_value=float(mu_vec @ weights),
        status=solution.message,
        feasible=True,
    )


def solve_sparse_portfolio(
    mu: pd.Series,
    covariance: pd.DataFrame,
    target_volatility: float,
    support_size: int,
    l2_reg: float = 1e-6,
    initial_weights: pd.Series | None = None,
    max_weight: float = 0.05,
) -> tuple[OptimizationResult, OptimizationResult]:
    dense_result = solve_max_return_under_risk(
        mu=mu,
        covariance=covariance,
        target_volatility=target_volatility,
        l2_reg=l2_reg,
        initial_weights=initial_weights,
        max_weight=max_weight,
    )
    if support_size >= len(mu):
        return dense_result, dense_result

    support = dense_result.weights.sort_values(ascending=False).head(support_size).index
    sparse_result = solve_max_return_under_risk(
        mu=mu.loc[support],
        covariance=covariance.loc[support, support],
        target_volatility=target_volatility,
        l2_reg=l2_reg,
        initial_weights=dense_result.weights.loc[support],
        max_weight=max_weight,
    )
    full_sparse = pd.Series(0.0, index=mu.index)
    full_sparse.loc[support] = sparse_result.weights
    full_sparse /= full_sparse.sum()
    sparse_variance = portfolio_variance(
        full_sparse.to_numpy(), covariance.loc[mu.index, mu.index].to_numpy()
    )
    finalized_sparse = OptimizationResult(
        weights=full_sparse,
        realized_volatility=float(np.sqrt(max(sparse_variance, 0.0))),
        objective_value=float(mu.to_numpy() @ full_sparse.to_numpy()),
        status=sparse_result.status,
        feasible=sparse_result.feasible,
    )
    return dense_result, finalized_sparse
