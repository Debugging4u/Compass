"""
Compass optimizer API
optimizer/api/main.py

Thin FastAPI wrapper around the portfolio_optimization pipeline. Every route
below calls the SAME public, pure functions the CLI stages use (see
correlation_matrix.py / expected_returns.py / optimal_portfolio.py /
backtest.py) — no logic is duplicated here, this file only shapes their
outputs into JSON.

/portfolio, /frontier, /backtest take a JSON body (not query params) because
they accept strategic_weights (a per-asset dict) and views (a list of typed
view objects) — neither fits cleanly in a query string.

No auth, no database, no rate limiting: this mirrors the pipeline's current
single-user, personal-tool scope. Add auth before exposing this beyond
localhost / a trusted server-side caller.

Run from optimizer/ with the venv active:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

from typing import Literal

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
    VIEW_CONFIDENCE,
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

class ViewRequest(BaseModel):
    """One investor view. Mirrors expected_returns.VIEWS's tuple format:

        ("absolute", asset, annual_total_return [, confidence])
        ("relative", long_asset, short_asset, annual_outperformance [, confidence])

    `target` is always the annual figure (e.g. 0.052 for "5.2% p.a."), in
    fraction form, matching the pipeline's own convention. `confidence` is
    the Meucci confidence multiplier (omit for VIEW_CONFIDENCE, the global
    default — >1 tightens conviction, <1 loosens it).
    """

    type: Literal["absolute", "relative"]
    asset: str | None = None            # absolute views
    long_asset: str | None = None       # relative views
    short_asset: str | None = None      # relative views
    target: float
    confidence: float | None = None


class PortfolioRequest(BaseModel):
    target_return: float = TARGET_RETURN_ANNUAL
    rf_annual: float = RF_ANNUAL
    strategic_weights: dict[str, float] | None = None  # None = pipeline default
    views: list[ViewRequest] | None = None              # None = pipeline default


class FrontierRequest(BaseModel):
    n_points: int = Field(50, ge=5, le=500)
    rf_annual: float = RF_ANNUAL
    strategic_weights: dict[str, float] | None = None
    views: list[ViewRequest] | None = None


class BacktestRequest(BaseModel):
    target_return: float = TARGET_RETURN_ANNUAL
    rf_annual: float = RF_ANNUAL
    neutral_prior: bool = False
    strategic_weights: dict[str, float] | None = None
    views: list[ViewRequest] | None = None


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


def _resolve_views(views: list[ViewRequest] | None) -> list[tuple]:
    """Fall back to the pipeline default (VIEWS), or convert + validate a
    custom list against ASSET_NAMES.

    Validated here rather than inside apply_views() (which needs a live
    BlackLitterman instance to mutate, so it can't be called standalone) —
    same reasoning as `_resolve_strategic_weights`: fail with a clean 400 now
    rather than an uncaught exception mid-backtest.

    Raises:
        HTTPException: 400, on an unknown asset, a missing required field, or
            a non-positive confidence.
    """
    if views is None:
        return VIEWS

    tuples: list[tuple] = []
    for i, v in enumerate(views, start=1):
        conf = VIEW_CONFIDENCE if v.confidence is None else v.confidence
        if conf <= 0:
            raise HTTPException(status_code=400, detail=f"View {i}: confidence must be positive; got {conf}.")

        if v.type == "absolute":
            if not v.asset:
                raise HTTPException(status_code=400, detail=f"View {i}: absolute view requires 'asset'.")
            if v.asset not in ASSET_NAMES:
                raise HTTPException(
                    status_code=400,
                    detail=f"View {i}: unknown asset '{v.asset}'. Known: {ASSET_NAMES}.",
                )
            tuples.append(("absolute", v.asset, v.target, conf))
        else:  # "relative"
            if not v.long_asset or not v.short_asset:
                raise HTTPException(
                    status_code=400, detail=f"View {i}: relative view requires 'long_asset' and 'short_asset'.",
                )
            if v.long_asset == v.short_asset:
                raise HTTPException(
                    status_code=400, detail=f"View {i}: 'long_asset' and 'short_asset' must differ.",
                )
            for name in (v.long_asset, v.short_asset):
                if name not in ASSET_NAMES:
                    raise HTTPException(
                        status_code=400,
                        detail=f"View {i}: unknown asset '{name}'. Known: {ASSET_NAMES}.",
                    )
            tuples.append(("relative", v.long_asset, v.short_asset, v.target, conf))
    return tuples


def _views_to_dicts(views_tuples: list[tuple]) -> list[dict]:
    """Convert the internal tuple form back to ViewRequest-shaped dicts, for
    echoing in responses (and for /universe's default-views payload)."""
    out = []
    for spec in views_tuples:
        kind = spec[0]
        if kind == "absolute":
            asset, target = spec[1], spec[2]
            conf = spec[3] if len(spec) >= 4 else VIEW_CONFIDENCE
            out.append({"type": "absolute", "asset": asset, "target": target, "confidence": conf})
        else:
            long_asset, short_asset, target = spec[1], spec[2], spec[3]
            conf = spec[4] if len(spec) >= 5 else VIEW_CONFIDENCE
            out.append({
                "type": "relative", "long_asset": long_asset, "short_asset": short_asset,
                "target": target, "confidence": conf,
            })
    return out


def _cov_and_mu(
    rf_annual: float, strategic_weights: dict[str, float], views: list[tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """Fresh (cov_annual, mu_weekly) off the current default universe.

    Callers must resolve/validate `strategic_weights`/`views` first (see
    `_resolve_strategic_weights`/`_resolve_views`) — this function assumes
    they're already valid.

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
        cov_df, ASSET_NAMES, views=views, rf_annual=rf_annual, strategic_weights=strategic_weights,
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

    strategic_weights/views here are the pipeline's DEFAULTS — the values to
    pre-populate the editors with, not necessarily what the last /portfolio
    call used (that's echoed back on each response instead).
    """
    return {
        "asset_names": ASSET_NAMES,
        "tickers": tickers,
        "max_weight": MAX_WEIGHT,
        "strategic_weights": STRATEGIC_WEIGHTS,
        "views": _views_to_dicts(VIEWS),
        "default_confidence": VIEW_CONFIDENCE,
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
    v = _resolve_views(req.views)
    cov, mu_weekly = _cov_and_mu(req.rf_annual, sw, v)

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
        "views": _views_to_dicts(v),
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
    v = _resolve_views(req.views)
    cov, mu_weekly = _cov_and_mu(req.rf_annual, sw, v)
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
    v = _resolve_views(req.views)
    try:
        weekly_returns, equity, infeasible = run_backtest(
            strategies=build_strategies(
                target_return_annual=req.target_return, rf_annual=req.rf_annual, strategic_weights=sw,
            ),
            rf_annual=req.rf_annual,
            neutral_prior=req.neutral_prior,
            strategic_weights=sw,
            views=v,
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
        "views": _views_to_dicts(v),
    }
