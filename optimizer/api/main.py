"""
Compass optimizer API
optimizer/api/main.py

Thin FastAPI wrapper around the portfolio_optimization pipeline. Every route
below calls the SAME public, pure functions the CLI stages use (see
correlation_matrix.py / expected_returns.py / optimal_portfolio.py /
backtest.py) — no logic is duplicated here, this file only shapes their
outputs into JSON.

/portfolio, /frontier, /backtest take a JSON body (not query params) because
they accept strategic_weights (a per-asset dict), views (a list of typed view
objects), and assets (a list of asset names) — none of which fit cleanly in a
query string.

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
    CANDIDATE_TICKERS,
    COV_METHOD,
    UniverseData,
    compute_cov_preferred,
    get_default_universe,
    load_universe,
    tickers,
)
from portfolio_optimization.expected_returns import (
    CANDIDATE_STRATEGIC_WEIGHTS,
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
    assets: list[str] | None = None                     # None = pipeline default universe


class FrontierRequest(BaseModel):
    n_points: int = Field(50, ge=5, le=500)
    rf_annual: float = RF_ANNUAL
    strategic_weights: dict[str, float] | None = None
    views: list[ViewRequest] | None = None
    assets: list[str] | None = None


class BacktestRequest(BaseModel):
    target_return: float = TARGET_RETURN_ANNUAL
    rf_annual: float = RF_ANNUAL
    neutral_prior: bool = False
    strategic_weights: dict[str, float] | None = None
    views: list[ViewRequest] | None = None
    assets: list[str] | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_universe(assets: list[str] | None) -> tuple[list[str], UniverseData]:
    """Fall back to the pipeline default universe, or validate + load a
    custom asset subset from CANDIDATE_TICKERS.

    A custom subset always downloads fresh (not memoised like the default) —
    this is a deliberate, on-demand action from the universe editor, not
    something hit on every request.

    Raises:
        HTTPException: 400, on an unknown asset name or fewer than 2 assets.
        HTTPException: 502, if the price download fails.
    """
    if assets is None:
        try:
            return ASSET_NAMES, get_default_universe()
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    unknown = [a for a in assets if a not in CANDIDATE_TICKERS]
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown asset(s) {unknown}. Known: {list(CANDIDATE_TICKERS)}.",
        )
    # Dedupe, preserving order.
    names = list(dict.fromkeys(assets))
    if len(names) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 2 assets to optimise over; got {names}.",
        )

    subset = {name: CANDIDATE_TICKERS[name] for name in names}
    try:
        return names, load_universe(tickers=subset)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _resolve_strategic_weights(sw: dict[str, float] | None, asset_names: list[str]) -> dict[str, float]:
    """Fall back to the full candidate policy weights, and validate against
    `asset_names`.

    The fallback is CANDIDATE_STRATEGIC_WEIGHTS (the full 14-asset superset),
    not the active-universe-only STRATEGIC_WEIGHTS — build_reference_weights
    ignores extra keys, so this is behaviourally identical for the default
    universe while also giving any newly-selected asset a sensible starting
    weight automatically, with no frontend guessing required.

    Centralised here so every route validates the same way, once, up front —
    letting an invalid dict reach `run_backtest()` would otherwise fail
    silently per-rebalance (caught as "infeasible" and papered over with
    carried-forward weights) instead of failing loudly as a bad request.

    Raises:
        HTTPException: 400, if weights are missing an asset or non-positive.
    """
    resolved = CANDIDATE_STRATEGIC_WEIGHTS if sw is None else sw
    try:
        build_reference_weights(asset_names, resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return resolved


def _resolve_views(views: list[ViewRequest] | None, asset_names: list[str]) -> list[tuple]:
    """Fall back to the pipeline default (VIEWS), or convert + validate a
    custom list against `asset_names`.

    Validated here rather than inside apply_views() (which needs a live
    BlackLitterman instance to mutate, so it can't be called standalone) —
    same reasoning as `_resolve_strategic_weights`: fail with a clean 400 now
    rather than an uncaught exception mid-backtest.

    NOTE: the pipeline default VIEWS references assets from the default
    7-asset universe (HBONDS, Technology, Global, GRID). If a custom `assets`
    list drops one of those, pass `views=[]` explicitly (not None) — leaving
    it None here would otherwise try to validate the default VIEWS against
    the smaller universe and 400.

    Raises:
        HTTPException: 400, on an unknown asset, a missing required field, or
            a non-positive confidence.
    """
    if views is None:
        views = VIEWS
        # Silently drop default views that don't apply to a custom universe —
        # this only fires for the *pipeline* default views against a custom
        # asset list, not for anything the user explicitly typed in.
        views = [
            spec for spec in views
            if all(name in asset_names for name in _view_asset_names(spec))
        ]

    tuples: list[tuple] = []
    for i, v in enumerate(views, start=1):
        # v may already be a raw tuple (from the VIEWS fallback above) or a
        # ViewRequest (from the request body) — normalise to one shape.
        if isinstance(v, tuple):
            v = _tuple_to_view_request(v)

        conf = VIEW_CONFIDENCE if v.confidence is None else v.confidence
        if conf <= 0:
            raise HTTPException(status_code=400, detail=f"View {i}: confidence must be positive; got {conf}.")

        if v.type == "absolute":
            if not v.asset:
                raise HTTPException(status_code=400, detail=f"View {i}: absolute view requires 'asset'.")
            if v.asset not in asset_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"View {i}: unknown asset '{v.asset}'. Known: {asset_names}.",
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
                if name not in asset_names:
                    raise HTTPException(
                        status_code=400,
                        detail=f"View {i}: unknown asset '{name}'. Known: {asset_names}.",
                    )
            tuples.append(("relative", v.long_asset, v.short_asset, v.target, conf))
    return tuples


def _view_asset_names(spec: tuple) -> list[str]:
    """Asset name(s) a raw VIEWS tuple references, for the default-universe filter above."""
    return [spec[1]] if spec[0] == "absolute" else [spec[1], spec[2]]


def _tuple_to_view_request(spec: tuple) -> ViewRequest:
    kind = spec[0]
    if kind == "absolute":
        conf = spec[3] if len(spec) >= 4 else None
        return ViewRequest(type="absolute", asset=spec[1], target=spec[2], confidence=conf)
    conf = spec[4] if len(spec) >= 5 else None
    return ViewRequest(type="relative", long_asset=spec[1], short_asset=spec[2], target=spec[3], confidence=conf)


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
    universe: UniverseData, asset_names: list[str],
    rf_annual: float, strategic_weights: dict[str, float], views: list[tuple],
) -> tuple[np.ndarray, np.ndarray]:
    """(cov_annual, mu_weekly) for the given universe/assumptions.

    Callers must resolve/validate `strategic_weights`/`views` first (see
    `_resolve_strategic_weights`/`_resolve_views`) — this function assumes
    they're already valid.
    """
    cov_df = compute_cov_preferred(universe.returns, universe.market_returns, method=COV_METHOD)
    mu_weekly, _bl = build_expected_returns(
        cov_df, asset_names, views=views, rf_annual=rf_annual, strategic_weights=strategic_weights,
    )

    cov = np.asarray(cov_df, dtype=float)
    cov = (cov + cov.T) / 2.0
    return cov, np.asarray(mu_weekly, dtype=float)


def _weight_dict(weights: np.ndarray, asset_names: list[str]) -> dict[str, float]:
    return {name: float(w) for name, w in zip(asset_names, weights)}


def _portfolio_summary(
    weights: np.ndarray, mu_weekly: np.ndarray, cov: np.ndarray, asset_names: list[str],
    rf_annual: float = RF_ANNUAL,
) -> dict:
    mu_annual = mu_weekly * WEEKS_PER_YEAR
    ret = float(portfolio_mean(weights, mu_annual))
    vol = float(portfolio_std(weights, cov))
    sharpe = (ret - rf_annual) / vol if vol > 1e-10 else None
    return {
        "weights": _weight_dict(weights, asset_names),
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

    asset_names/tickers/strategic_weights/views are the pipeline's DEFAULTS —
    the values to pre-populate the editors with, not necessarily what the
    last /portfolio call used (that's echoed back on each response instead).
    candidate_tickers/candidate_strategic_weights are the FULL selectable
    universe, for the asset picker.
    """
    return {
        "asset_names": ASSET_NAMES,
        "tickers": tickers,
        "candidate_tickers": CANDIDATE_TICKERS,
        "candidate_strategic_weights": CANDIDATE_STRATEGIC_WEIGHTS,
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
    asset_names, univ = _resolve_universe(req.assets)
    sw = _resolve_strategic_weights(req.strategic_weights, asset_names)
    v = _resolve_views(req.views, asset_names)
    cov, mu_weekly = _cov_and_mu(univ, asset_names, req.rf_annual, sw, v)

    w_gmv = constrained_gmv(mu_weekly, cov)
    w_ms = constrained_max_sharpe(mu_weekly, cov, req.rf_annual)

    target_summary, target_error = None, None
    try:
        w_target = optimal_portfolio(mu_weekly, cov, req.target_return)
        target_summary = _portfolio_summary(w_target, mu_weekly, cov, asset_names, req.rf_annual)
    except (ValueError, RuntimeError) as exc:
        target_error = str(exc)

    return {
        "asset_names": asset_names,
        "rf_annual": req.rf_annual,
        "target_return_annual": req.target_return,
        "strategic_weights": sw,
        "views": _views_to_dicts(v),
        "mu_weekly": _weight_dict(mu_weekly, asset_names),
        "mu_annual": _weight_dict(mu_weekly * WEEKS_PER_YEAR, asset_names),
        "covariance_condition_number": float(np.linalg.cond(cov)),
        "portfolios": {
            "gmv": _portfolio_summary(w_gmv, mu_weekly, cov, asset_names, req.rf_annual),
            "max_sharpe": _portfolio_summary(w_ms, mu_weekly, cov, asset_names, req.rf_annual),
            "target": target_summary,
        },
        "target_error": target_error,
    }


@app.post("/frontier")
def frontier(req: FrontierRequest):
    asset_names, univ = _resolve_universe(req.assets)
    sw = _resolve_strategic_weights(req.strategic_weights, asset_names)
    v = _resolve_views(req.views, asset_names)
    cov, mu_weekly = _cov_and_mu(univ, asset_names, req.rf_annual, sw, v)
    mu_f, std_f, w_f = compute_constrained_frontier(mu_weekly, cov, n_points=req.n_points)
    points = [
        {"return_annual": float(r), "vol_annual": float(s), "weights": _weight_dict(w, asset_names)}
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
    asset_names, univ = _resolve_universe(req.assets)
    sw = _resolve_strategic_weights(req.strategic_weights, asset_names)
    v = _resolve_views(req.views, asset_names)
    try:
        weekly_returns, equity, infeasible = run_backtest(
            returns_df=univ.returns,
            strategies=build_strategies(
                target_return_annual=req.target_return, rf_annual=req.rf_annual,
                strategic_weights=sw, asset_names=asset_names,
            ),
            rf_annual=req.rf_annual,
            neutral_prior=req.neutral_prior,
            strategic_weights=sw,
            views=v,
            asset_names=asset_names,
        )
    except SystemExit as exc:
        # run_backtest() raises SystemExit when there isn't enough history for
        # a single rebalance — a config problem, not a server error.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    table = performance_table(weekly_returns, rf_annual=req.rf_annual)
    return {
        "asset_names": asset_names,
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
