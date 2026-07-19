"""
Compass optimizer API
optimizer/api/main.py

Thin FastAPI wrapper around the portfolio_optimization pipeline. Every route
below calls the SAME public, pure functions the CLI stages use (see
correlation_matrix.py / expected_returns.py / optimal_portfolio.py /
backtest.py) — no logic is duplicated here, this file only shapes their
outputs into JSON.

/portfolio, /frontier, /backtest take a JSON body (not query params) because
they accept strategic_weights, a per-asset dict — the same shape VIEWS will
need when that's wired in next, so this is the contract both will share.

No auth, no database, no rate limiting: this mirrors the pipeline's current
single-user, personal-tool scope. Add auth before exposing this beyond
localhost / a trusted server-side caller.

Run from optimizer/ with the venv active:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
    STRATEGIC_WEIGHTS,
    VIEWS,
    WEEKS_PER_YEAR,
    build_expected_returns,
    build_reference_weights,
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
# Request bodies
# ---------------------------------------------------------------------------

class PortfolioRequest(BaseModel):
    target_return: float = TARGET_RETURN_ANNUAL
    rf_annual: float = RF_ANNUAL
    strategic_weights: dict[str, float] | None = None  # None = pipeline default


class FrontierRequest(BaseModel):
    n_points: int = Field(50, ge=5, le=500)
    rf_annual: float = RF_ANNUAL
    strategic_weights: dict[str, float] | None = None


class BacktestRequest(BaseModel):
    target_return: float = TARGET_RETURN_ANNUAL
    rf_annual: float = RF_ANNUAL
    neutral_prior: bool = False
    strategic_weights: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_strategic_weights(sw: dict[str, float] | None) -> dict[str, float]:
    """Fall back to the pipeline default, and validate against ASSET_NAMES.

    Centralised here so every route validates the same way, once, up front —
    letting an invalid dict reach `run_backtest()` would otherwise fail
    silently per-rebalance (caught as "infeasible" and papered over with
    carried-forward weights) instead of failing loudly as a bad request.

    Raises:
        HTTPException: 400, if weights are missing an asset or non-positive.
    """
    resolved = STRATEGIC_WEIGHTS if sw is None else sw
    try:
        build_reference_weights(ASSET_NAMES, resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return resolved


def _cov_and_mu(rf_annual: float, strategic_weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Fresh (cov_annual, mu_weekly) off the current default universe.

    Callers must resolve/validate `strategic_weights` first (see
    `_resolve_strategic_weights`) — this function assumes it's already valid.

    Raises:
        HTTPException: 502, if the universe can't be loaded (e.g. the price
            download failed).
    """
    try:
        universe = get_default_universe()
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    cov_df = compute_cov_preferred(universe.returns, universe.market_returns, method=COV_METHOD)
    mu_weekly, _bl = build_expected_returns(
        cov_df, ASSET_NAMES, views=VIEWS, rf_annual=rf_annual, strategic_weights=strategic_weights,
    )

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
    """Static universe config — no network call, safe to poll freely.

    strategic_weights here is the pipeline's DEFAULT — the value to
    pre-populate a weights editor with, not necessarily what the last
    /portfolio call used (that's echoed back on each response instead).
    """
    return {
        "asset_names": ASSET_NAMES,
        "tickers": tickers,
        "max_weight": MAX_WEIGHT,
        "strategic_weights": STRATEGIC_WEIGHTS,
    }


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


@app.post("/portfolio")
def portfolio(req: PortfolioRequest):
    sw = _resolve_strategic_weights(req.strategic_weights)
    cov, mu_weekly = _cov_and_mu(req.rf_annual, sw)

    w_gmv = constrained_gmv(mu_weekly, cov)
    w_ms = constrained_max_sharpe(mu_weekly, cov, req.rf_annual)

    target_summary, target_error = None, None
    try:
        w_target = optimal_portfolio(mu_weekly, cov, req.target_return)
        target_summary = _portfolio_summary(w_target, mu_weekly, cov, req.rf_annual)
    except (ValueError, RuntimeError) as exc:
        target_error = str(exc)

    return {
        "rf_annual": req.rf_annual,
        "target_return_annual": req.target_return,
        "strategic_weights": sw,
        "mu_weekly": _weight_dict(mu_weekly),
        "mu_annual": _weight_dict(mu_weekly * WEEKS_PER_YEAR),
        "covariance_condition_number": float(np.linalg.cond(cov)),
        "portfolios": {
            "gmv": _portfolio_summary(w_gmv, mu_weekly, cov, req.rf_annual),
            "max_sharpe": _portfolio_summary(w_ms, mu_weekly, cov, req.rf_annual),
            "target": target_summary,
        },
        "target_error": target_error,
    }


@app.post("/frontier")
def frontier(req: FrontierRequest):
    sw = _resolve_strategic_weights(req.strategic_weights)
    cov, mu_weekly = _cov_and_mu(req.rf_annual, sw)
    mu_f, std_f, w_f = compute_constrained_frontier(mu_weekly, cov, n_points=req.n_points)
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


@app.post("/backtest")
def backtest(req: BacktestRequest):
    sw = _resolve_strategic_weights(req.strategic_weights)
    try:
        weekly_returns, equity, infeasible = run_backtest(
            strategies=build_strategies(
                target_return_annual=req.target_return, rf_annual=req.rf_annual, strategic_weights=sw,
            ),
            rf_annual=req.rf_annual,
            neutral_prior=req.neutral_prior,
            strategic_weights=sw,
        )
    except SystemExit as exc:
        # run_backtest() raises SystemExit when there isn't enough history for
        # a single rebalance — a config problem, not a server error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    table = performance_table(weekly_returns, rf_annual=req.rf_annual)
    return {
        "performance": table.to_dict(orient="index"),
        "equity_curves": {
            col: {str(idx.date()): float(v) for idx, v in equity[col].items()}
            for col in equity.columns
        },
        "infeasible": infeasible,
        "oos_weeks": len(weekly_returns),
        "rf_annual": req.rf_annual,
        "neutral_prior": req.neutral_prior,
        "strategic_weights": sw,
    }
