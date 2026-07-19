"""
Compass optimizer API
optimizer/api/main.py

Thin FastAPI wrapper around the portfolio_optimization pipeline. Every route
below calls the SAME public, pure functions the CLI stages use (see
correlation_matrix.py / expected_returns.py / optimal_portfolio.py /
backtest.py) — no logic is duplicated here, this file only shapes their
outputs into JSON.

No auth, no database, no rate limiting: this mirrors the pipeline's current
single-user, personal-tool scope. Add auth before exposing this beyond
localhost / a trusted server-side caller.

Run from optimizer/ with the venv active:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from codelib.portfolio_optimization.mean_variance import portfolio_mean, portfolio_std
from codelib.statistics.moments import cov_to_corr_matrix

import portfolio_optimization.correlation_matrix as correlation_matrix
from portfolio_optimization.correlation_matrix import (
    ASSET_NAMES,
    COV_METHOD,
    compute_cov_preferred,
    get_default_universe,
    tickers,
)
from portfolio_optimization.expected_returns import (
    RF_ANNUAL,
    VIEWS,
    WEEKS_PER_YEAR,
    build_expected_returns,
)
from portfolio_optimization.optimal_portfolio import (
    MAX_WEIGHT,
    TARGET_RETURN_ANNUAL,
    compute_constrained_frontier,
    constrained_gmv,
    constrained_max_sharpe,
    optimal_portfolio,
)
from portfolio_optimization.backtest import build_strategies, performance_table, run_backtest

app = FastAPI(title="Compass Optimizer API", version="0.1.0")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _cov_and_mu(rf_annual: float = RF_ANNUAL) -> tuple[np.ndarray, np.ndarray]:
    """Fresh (cov_annual, mu_weekly) off the current default universe.

    Args:
        rf_annual: Annual risk-free rate — feeds the BL excess/total
            conversion, so it shifts where mu (and everything downstream)
            sits, not just the Sharpe ratios computed from it.

    Raises:
        HTTPException: 502, if the universe can't be loaded (e.g. the price
            download failed).
    """
    try:
        universe = get_default_universe()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    cov_df = compute_cov_preferred(universe.returns, universe.market_returns, method=COV_METHOD)
    mu_weekly, _bl = build_expected_returns(cov_df, ASSET_NAMES, views=VIEWS, rf_annual=rf_annual)

    cov = np.asarray(cov_df, dtype=float)
    cov = (cov + cov.T) / 2.0
    return cov, np.asarray(mu_weekly, dtype=float)


def _weight_dict(weights: np.ndarray) -> dict[str, float]:
    return {name: float(w) for name, w in zip(ASSET_NAMES, weights)}


def _portfolio_summary(
    weights: np.ndarray, mu_weekly: np.ndarray, cov: np.ndarray, rf_annual: float = RF_ANNUAL,
) -> dict:
    mu_annual = mu_weekly * WEEKS_PER_YEAR
    ret = float(portfolio_mean(weights, mu_annual))
    vol = float(portfolio_std(weights, cov))
    sharpe = (ret - rf_annual) / vol if vol > 1e-10 else None
    return {
        "weights": _weight_dict(weights),
        "return_annual": ret,
        "vol_annual": vol,
        "sharpe": sharpe,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/universe")
def universe():
    """Static universe config — no network call, safe to poll freely."""
    return {"asset_names": ASSET_NAMES, "tickers": tickers, "max_weight": MAX_WEIGHT}


@app.post("/refresh")
def refresh():
    """Force a fresh price download, clearing the memoised universe."""
    correlation_matrix._default_universe = None
    try:
        loaded = get_default_universe()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "asset_names": ASSET_NAMES,
        "weeks": len(loaded.returns),
        "start": str(loaded.returns.index[0].date()),
        "end": str(loaded.returns.index[-1].date()),
    }


@app.get("/portfolio")
def portfolio(
    target_return: float = Query(
        TARGET_RETURN_ANNUAL,
        description="Annualised target return for the target-return portfolio",
    ),
    rf_annual: float = Query(
        RF_ANNUAL,
        description="Annualised risk-free rate — affects mu (BL excess/total "
        "conversion) and every Sharpe ratio, not just the Max Sharpe portfolio",
    ),
):
    cov, mu_weekly = _cov_and_mu(rf_annual)

    w_gmv = constrained_gmv(mu_weekly, cov)
    w_ms = constrained_max_sharpe(mu_weekly, cov, rf_annual)

    target_summary, target_error = None, None
    try:
        w_target = optimal_portfolio(mu_weekly, cov, target_return)
        target_summary = _portfolio_summary(w_target, mu_weekly, cov, rf_annual)
    except (ValueError, RuntimeError) as exc:
        target_error = str(exc)

    return {
        "rf_annual": rf_annual,
        "target_return_annual": target_return,
        "mu_weekly": _weight_dict(mu_weekly),
        "mu_annual": _weight_dict(mu_weekly * WEEKS_PER_YEAR),
        "covariance_condition_number": float(np.linalg.cond(cov)),
        "portfolios": {
            "gmv": _portfolio_summary(w_gmv, mu_weekly, cov, rf_annual),
            "max_sharpe": _portfolio_summary(w_ms, mu_weekly, cov, rf_annual),
            "target": target_summary,
        },
        "target_error": target_error,
    }


@app.get("/frontier")
def frontier(
    n_points: int = Query(50, ge=5, le=500),
    rf_annual: float = Query(RF_ANNUAL, description="Annualised risk-free rate"),
):
    cov, mu_weekly = _cov_and_mu(rf_annual)
    mu_f, std_f, w_f = compute_constrained_frontier(mu_weekly, cov, n_points=n_points)
    points = [
        {"return_annual": float(r), "vol_annual": float(s), "weights": _weight_dict(w)}
        for r, s, w in zip(mu_f, std_f, w_f)
    ]
    return {"points": points}


@app.get("/correlation")
def correlation():
    try:
        loaded = get_default_universe()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    cov_df = compute_cov_preferred(loaded.returns, loaded.market_returns, method=COV_METHOD)
    corr = cov_to_corr_matrix(cov_df.values)
    return {
        "assets": ASSET_NAMES,
        "matrix": [[float(x) for x in row] for row in corr],
        "method": f"ledoit_wolf_{COV_METHOD}",
    }


@app.get("/backtest")
def backtest(
    target_return: float = Query(
        TARGET_RETURN_ANNUAL,
        description="Annualised target return for the backtest's 'Target X%' strategy",
    ),
    rf_annual: float = Query(RF_ANNUAL, description="Annualised risk-free rate"),
    neutral_prior: bool = Query(
        False,
        description="If true, each rebalance re-estimates mu with an equal-weight "
        "prior and no views — tests the shrinkage + optimisation machinery in "
        "isolation. If false (default), uses the live STRATEGIC_WEIGHTS + VIEWS "
        "at every historical rebalance, which is partly a backtest of beliefs "
        "formed with full-sample hindsight, not just the machinery.",
    ),
):
    try:
        weekly_returns, equity, infeasible = run_backtest(
            strategies=build_strategies(target_return_annual=target_return, rf_annual=rf_annual),
            rf_annual=rf_annual,
            neutral_prior=neutral_prior,
        )
    except SystemExit as exc:
        # run_backtest() raises SystemExit when there isn't enough history for
        # a single rebalance — a config problem, not a server error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    table = performance_table(weekly_returns, rf_annual=rf_annual)
    return {
        "performance": table.to_dict(orient="index"),
        "equity_curves": {
            col: {str(idx.date()): float(v) for idx, v in equity[col].items()}
            for col in equity.columns
        },
        "infeasible": infeasible,
        "oos_weeks": len(weekly_returns),
        "rf_annual": rf_annual,
        "neutral_prior": neutral_prior,
    }
