"""
Stage 4 — Expected return forecasting (Black-Litterman)
portfolio_optimization/expected_returns.py

Produces the expected-return vector consumed by Stage 5. Replaces the
placeholder ``mu`` in ``optimal_portfolio.py`` with Black-Litterman posterior
means, built on the Meucci (2008) reformulation in
``codelib.portfolio_optimization.black_litterman.BlackLitterman``.

Why Black-Litterman (and the Meucci variant):
    Mean-variance optimisation is pathologically sensitive to the return
    vector; raw historical means produce extreme, unstable weights. BL anchors
    on the market-implied equilibrium return (a well-behaved neutral prior) and
    tilts it only where the investor holds an explicit view. The Meucci variant
    folds the classic ``tau`` into a single ``confidence`` handle (one fewer
    arbitrary knob) and exposes a consistency diagnostic that flags views
    statistically incompatible with equilibrium.

    Literature-faithful division of labour (Black-Litterman 1992; He & Litterman
    1999; Idzorek 2005): the PRIOR is the neutral reference an opinion-free
    investor would hold; active opinion enters ONLY through the views. For this
    core-satellite, overlapping universe the neutral reference is Global + the
    strategic bond allocation, with the regional/sector satellites held at small
    near-neutral weights — every satellite tilt is expressed as a VIEW, not baked
    into the reference weights. This keeps ``pi`` an honest reference, keeps the
    posterior doing real work, and keeps structural policy separable from
    time-varying conviction.

Per-view confidence:
    The Meucci scalar ``confidence`` scales the whole view-covariance Omega
    uniformly (Omega = (P Sigma P^T) / c). To mix a high-conviction bond view
    with softer equity views, each view carries its OWN optional confidence; the
    global ``VIEW_CONFIDENCE`` is the default for views that omit one. Omega is
    built as ``M @ (P Sigma P^T) @ M`` with ``M = diag(1/sqrt(c_i))``, which
    reduces EXACTLY to the scalar Meucci formula when all confidences are equal,
    so behaviour is unchanged for the uniform-confidence case.

Output contract (what Stage 5 imports):
    ``mu_preferred`` — a 1-D ``np.ndarray`` of **weekly, total (not excess),
    arithmetic** expected returns, ordered to match ``returns.columns`` from
    Stage 3. Stage 5 annualises it linearly (``mu * 52``), exactly as it does
    the placeholder. To wire it in, change one line in ``optimal_portfolio.py``:

        from portfolio_optimization.expected_returns import mu_preferred
        ...
        mu_weekly = mu_preferred          # was the placeholder block

Unit conventions (kept consistent with Stage 5):
    - Covariance: ``cov_preferred`` from Stage 3 is **annualised**. BL is run on
      the **weekly** covariance (``cov_annual / 52``) so the resulting weekly
      ``mu`` matches Stage 5's weekly input and its ``* 52`` annualisation.
      Linear scaling assumes iid weekly returns (same assumption Stage 3/5 make).
    - Equilibrium ``pi`` and the BL posterior are **excess** returns (over rf).
      ``rf_weekly = rf_annual / 52`` is added back at the end to produce the
      **total** returns Stage 5's placeholder also reported.
    - Views are specified in intuitive **annual, total-return** units and
      converted internally to the weekly-excess units BL requires.
    - Arithmetic vs log: BL is an arithmetic-return (normal) model, while the
      covariance is estimated from log returns. For weekly magnitudes the two
      coincide to first order; the linear ``* 52`` annualisation in Stage 5 is
      exact for log returns and a close approximation for arithmetic. Flagged,
      consistent with the existing Stage 5 annualisation note.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# codelib — used directly, no wrapper (lean design).
from codelib.portfolio_optimization.black_litterman import BlackLitterman

# Stage 3: asset order (no I/O), the universe loader, and the covariance
# recipe — cov_preferred itself is no longer a module-level value (see
# correlation_matrix.py's import-time contract), so it's built below from
# get_default_universe() instead of imported directly.
from portfolio_optimization.correlation_matrix import (
    ASSET_NAMES,
    COV_METHOD,
    compute_cov_preferred,
    get_default_universe,
)


# ---------------------------------------------------------------------------
# Configuration (no magic numbers — every parameter lives here)
# ---------------------------------------------------------------------------

WEEKS_PER_YEAR = 52

# Risk-free rate. Kept identical to Stage 5 (10y government bond) so the
# excess/total conversion is consistent across stages.
RF_ANNUAL = 0.04

# Risk-aversion (lambda). Leave as None to CALIBRATE it from a target market
# Sharpe (recommended) — a fixed lambda is not tied to this universe's risk
# level and tends to imply an unrealistically low market Sharpe. Set a float to
# pin lambda directly instead.
LAMBDA: float | None = None

# Target annual Sharpe of the neutral market portfolio, used only when
# LAMBDA is None. ~0.4 is the long-run world-equity figure. lambda is then
# backed out as SR / sigma_mkt, so the implied premium is anchored to a Sharpe
# you believe rather than to an arbitrary risk-aversion constant.
TARGET_MARKET_SHARPE = 0.45

# DEFAULT view confidence (Meucci). Applied to any view that does not specify
# its own confidence as a trailing tuple element (see VIEWS). 1.0 weights a view
# roughly on par with the prior; > 1.0 tightens conviction (posterior moves
# closer to the view), < 1.0 loosens it. This replaces the classic ``tau`` as
# the uncertainty handle.
VIEW_CONFIDENCE = 1.0

# Strategic (policy) reference weights used to derive the equilibrium prior
# ``pi``. Renamed from MARKET_CAPS: these are deliberate POLICY weights, not
# market caps — see the literature note in the module docstring.
# IMPORTANT:
#   * The neutral, opinion-free portfolio
#   * These are *asset-class* size proxies, NOT ETF AUM. AUM measures wrapper
#     popularity, not the size of the underlying market, and would distort pi.
#   * Units are arbitrary (relative proportions only); they are normalised to
#     sum to 1. Keep everything in one currency (USD) so the NORW/Norway FX
#     distortion stays confined to that sleeve.
#   * The function rejects non-positive weights, so a satellite cannot be set to
#     literal zero — give it a small de-minimis weight and let its VIEW do the
#     work.
#   * Keys must cover every asset in ``returns.columns``. Extra keys not present
#     in the universe are simply ignored.
# Full candidate set, including assets not currently active in
# correlation_matrix.tickers — so re-activating one via the API's universe
# selector has a sensible starting weight instead of an arbitrary placeholder.
# build_reference_weights() ignores keys not in asset_names, so this superset
# is safe to hand it directly regardless of which subset is actually active.
CANDIDATE_STRATEGIC_WEIGHTS: dict[str, float] = {
    "Global": 77.0,
    "Asia": 5.0,
    "Europe": 5.0,
    "Norway": 1.0,
    "Technology": 2.0,
    "HBONDS": 30.0,
    "GRID": 1.0,
    "Industrials": 2.0,
    "Atea": 11.0,
    "Austevoll": 11.0,
    "DNB": 11.0,
    "Equinor": 11.0,
    "SalMar": 11.0,
    "Veidekke": 11.0,
}

STRATEGIC_WEIGHTS: dict[str, float] = {
    name: CANDIDATE_STRATEGIC_WEIGHTS[name] for name in [
        "Global", "Asia", "Europe", "Norway", "Technology", "HBONDS", "GRID",
    ]
}

# Investor views. Empty list => pure equilibrium returns. All targets are in ANNUAL TOTAL-return units. Each
# view may carry an OPTIONAL trailing per-view confidence; if omitted, the
# global VIEW_CONFIDENCE is used.
#
#   ("absolute", asset, annual_total_return [, confidence])
#       e.g. ("absolute", "Technology", 0.12)        -> "Tech returns 12% p.a."
#       e.g. ("absolute", "HBONDS", 0.052, 2.0)      -> same, high conviction
#   ("relative", long_asset, short_asset, annual_outperformance [, confidence])
#       e.g. ("relative", "Asia", "Europe", 0.03)         -> "Asia beats Europe 3% p.a."
#       e.g. ("relative", "Technology", "Global", 0.03, 0.5) -> softer conviction
#
# NOTE: the confidence magnitudes below encode YOUR conviction, not a
# recommendation — the bond yield is closer to observable, hence tighter; the
# equity tilt is a softer opinion, hence looser. Adjust to taste and watch the
# consistency index in the diagnostics.
VIEWS = [
    ("absolute", "HBONDS", 0.052, 2.0),
    ("relative", "Technology", "Global", 0.03, 0.5),
    ("relative", "GRID", "Global", 0.02, 0.4),       # grid-upgrade names beat the global market
    #("relative", "Industrials", "Europe", 0.02, 0.4),  # European industrials beat the European market
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_reference_weights(
    asset_names: list[str],
    strategic_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Build the normalised reference-weight vector aligned to ``asset_names``.

    Alignment is by name, never by position, to avoid the column-ordering class
    of bug.

    If ``strategic_weights`` is ``None``, returns an EQUAL-WEIGHT vector — a
    zero-config neutral prior that lets an arbitrary universe run with no
    hand-tuning. Supply a mapping only when you hold a deliberate policy/
    strategic allocation; in that case every asset must have an entry.

    Args:
        asset_names: Asset order to align to (``list(returns.columns)``).
        strategic_weights: Mapping of asset name -> relative weight proxy, or
            ``None`` for an equal-weight default.

    Returns:
        Weights of shape (n,), summing to 1, in ``asset_names`` order.

    Raises:
        ValueError: If ``strategic_weights`` is supplied but missing an asset,
            or any supplied weight is non-positive.
    """
    if strategic_weights is None:
        n = len(asset_names)
        return np.full(n, 1.0 / n)

    missing = [name for name in asset_names if name not in strategic_weights]
    if missing:
        raise ValueError(
            f"STRATEGIC_WEIGHTS is missing entries for {missing}. "
            f"Every asset in returns.columns ({asset_names}) needs a weight."
        )

    weights = np.array([strategic_weights[name] for name in asset_names], dtype=float)
    if np.any(weights <= 0) or weights.sum() <= 0:
        raise ValueError(
            f"Strategic weights must be positive; got "
            f"{dict(zip(asset_names, (float(w) for w in weights)))}."
        )

    return weights / weights.sum()


def _view_confidence(spec: tuple, default: float) -> float:
    """Return the per-view confidence for a view spec, or ``default`` if omitted.

    Absolute views carry confidence in position 3 (4-tuple); relative views in
    position 4 (5-tuple).
    """
    kind = spec[0]
    if kind == "absolute":
        return float(spec[3]) if len(spec) >= 4 else float(default)
    if kind == "relative":
        return float(spec[4]) if len(spec) >= 5 else float(default)
    raise ValueError(f"Unknown view type '{kind}'. Use 'absolute' or 'relative'.")


def apply_views(
    bl: BlackLitterman,
    views: list[tuple],
    asset_names: list[str],
    rf_annual: float,
    default_confidence: float,
) -> np.ndarray:
    """Translate annual, total-return views into BL's weekly-excess units.

    Absolute view target (annual total) -> weekly excess via
    ``(target - rf) / 52``. Relative views are spreads, so the risk-free rate
    cancels and only ``target / 52`` is applied. Views are added in list order,
    so the returned confidence array is row-aligned to ``bl.view_mat``.

    Args:
        bl: A ``BlackLitterman`` instance to attach views to (mutated in place).
        views: View specifications (see the ``VIEWS`` config for the format).
        asset_names: Asset order, for resolving names to indices.
        rf_annual: Annual risk-free rate, for the excess conversion.
        default_confidence: Confidence applied to views with no explicit value.

    Returns:
        Per-view confidences, shape (k,), in the same row order as the views
        were added (i.e. aligned to ``bl.view_mat``).

    Raises:
        ValueError: On an unknown view type, unknown asset name, or a
            non-positive confidence.
    """
    index_of = {name: i for i, name in enumerate(asset_names)}

    def resolve(name: str) -> int:
        if name not in index_of:
            raise ValueError(f"View references unknown asset '{name}'. Known: {asset_names}.")
        return index_of[name]

    confidences: list[float] = []
    for spec in views:
        kind = spec[0]
        conf = _view_confidence(spec, default_confidence)
        if kind == "absolute":
            asset, annual_total = spec[1], spec[2]
            target_weekly_excess = (annual_total - rf_annual) / WEEKS_PER_YEAR
            bl.add_equality_view(resolve(asset), target_weekly_excess)
        elif kind == "relative":
            long_asset, short_asset, annual_diff = spec[1], spec[2], spec[3]
            target_weekly = annual_diff / WEEKS_PER_YEAR
            bl.add_diff_view(resolve(long_asset), resolve(short_asset), target_weekly)
        else:
            raise ValueError(f"Unknown view type '{kind}'. Use 'absolute' or 'relative'.")
        confidences.append(conf)

    conf_arr = np.asarray(confidences, dtype=float)
    if np.any(conf_arr <= 0):
        raise ValueError(
            f"View confidences must be positive; got {dict(enumerate(conf_arr))}."
        )
    return conf_arr


def calibrate_lambda(
    cov_annual_np: np.ndarray,
    w_mkt: np.ndarray,
    target_sharpe: float,
) -> float:
    """Back out the risk-aversion lambda so the neutral portfolio carries a
    target Sharpe ratio.

    lambda = SR / sigma_mkt, evaluated on the ANNUAL covariance. Sharpe ratios
    scale with sqrt-time, so the annualised volatility is the correct
    denominator; the resulting lambda is itself scale-invariant and can be
    passed straight to a BlackLitterman built on weekly covariance.

    Args:
        cov_annual_np: Annualised covariance matrix, shape (n, n).
        w_mkt: Normalised market weights, shape (n,).
        target_sharpe: Desired annual Sharpe ratio of the neutral portfolio
            (~0.4 for long-run world equity).

    Returns:
        The calibrated lambda.

    Raises:
        ValueError: If the neutral-portfolio variance is non-positive.
    """
    sigma2_mkt = float(w_mkt @ cov_annual_np @ w_mkt)
    if sigma2_mkt <= 0:
        raise ValueError(
            f"Neutral-portfolio variance must be positive; got {sigma2_mkt}."
        )
    return target_sharpe / np.sqrt(sigma2_mkt)


def build_view_cov_mat(
    cov_weekly: np.ndarray,
    view_mat: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    """Build the Meucci view-covariance Omega with PER-VIEW confidence.

    The scalar Meucci form is ``Omega = (P Sigma P^T) / c``. Generalising to a
    per-view confidence vector ``c_i``, set ``M = diag(1/sqrt(c_i))`` and

        Omega = M @ (P Sigma P^T) @ M

    This scales view i's variance by ``1/c_i`` and the (i, j) cross term by
    ``1/sqrt(c_i c_j)``, preserving the equilibrium-induced correlation between
    views. When all ``c_i`` are equal to ``c`` it collapses to ``(P Sigma P^T)/c``
    EXACTLY, i.e. ``BlackLitterman.calculate_view_cov_mat`` — so the
    uniform-confidence case is unchanged.

    Args:
        cov_weekly: Weekly covariance Sigma the BL model runs on, shape (n, n).
        view_mat: Pick matrix P, shape (k, n).
        confidences: Per-view confidences, shape (k,), row-aligned to ``view_mat``.

    Returns:
        Omega, shape (k, k).
    """
    base_omega = view_mat @ cov_weekly @ view_mat.T          # P Sigma P^T, (k, k)
    scale = np.diag(1.0 / np.sqrt(confidences))               # M, (k, k)
    return scale @ base_omega @ scale


def build_expected_returns(
    cov_annual: pd.DataFrame,
    asset_names: list[str],
    *,
    strategic_weights: dict[str, float] | None = STRATEGIC_WEIGHTS,
    rf_annual: float = RF_ANNUAL,
    lam: float | None = LAMBDA,
    views: list[tuple] | None = None,
    default_confidence: float = VIEW_CONFIDENCE,
    target_market_sharpe: float = TARGET_MARKET_SHARPE,
) -> tuple[np.ndarray, BlackLitterman]:
    """Compute weekly, total-return Black-Litterman expected returns.

    Views are folded in with per-view confidence (see ``build_view_cov_mat``).
    Because the codelib ``calculate_posterior_distribution`` only accepts a
    single scalar confidence, this function builds Omega itself and calls the
    static posterior methods directly, then writes the results back onto the
    ``BlackLitterman`` instance so its consistency / sensitivity diagnostics
    remain usable.

    Args:
        cov_annual: Annualised covariance (``cov_preferred`` from Stage 3),
            indexed/columned by asset name.
        asset_names: Output order (``list(returns.columns)``).
        strategic_weights: Policy weights for the equilibrium prior.
        rf_annual: Annual risk-free rate (excess <-> total conversion).
        lam: Risk-aversion coefficient. If ``None``, it is calibrated from
            ``target_market_sharpe``; pass a float to pin it directly.
        views: View specifications; ``None`` or ``[]`` gives pure equilibrium.
        default_confidence: Meucci confidence for views with no explicit value
            (> 1 tightens, < 1 loosens).
        target_market_sharpe: Target annual Sharpe of the neutral portfolio,
            used only when ``lam is None``.

    Returns:
        A tuple of:
          - ``mu_weekly_total``: weekly total expected returns, shape (n,),
            ordered to match ``asset_names``.
          - ``bl``: the fitted ``BlackLitterman`` instance (for diagnostics).

    Raises:
        ValueError: If the covariance cannot be aligned to ``asset_names``.
    """
    # Reindex defensively so the matrix is guaranteed to match asset order,
    # then symmetrise away any tiny reconstruction asymmetry before inversion.
    try:
        cov_aligned = cov_annual.reindex(index=asset_names, columns=asset_names)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Could not align covariance to {asset_names}: {exc}") from exc
    if cov_aligned.isnull().to_numpy().any():
        raise ValueError(
            "Covariance contains NaNs after aligning to asset_names — a name "
            "mismatch between cov_preferred and returns.columns is likely."
        )

    cov_annual_np = np.asarray(cov_aligned, dtype=float)
    cov_annual_np = (cov_annual_np + cov_annual_np.T) / 2.0

    # BL runs on the WEEKLY covariance so its weekly mu feeds Stage 5's *52.
    cov_weekly = cov_annual_np / WEEKS_PER_YEAR

    w_mkt = build_reference_weights(asset_names, strategic_weights)

    # Calibrate lambda to the target market Sharpe unless pinned explicitly.
    lam_eff = (
        calibrate_lambda(cov_annual_np, w_mkt, target_market_sharpe)
        if lam is None else lam
    )
    bl = BlackLitterman(cov_mat=cov_weekly, market_weights=w_mkt, lam=lam_eff)

    if views:
        confidences = apply_views(bl, views, asset_names, rf_annual, default_confidence)
        # Per-view Omega; bypass calculate_posterior_distribution (scalar-only)
        # and call the static posterior methods with our own view covariance.
        omega = build_view_cov_mat(bl.cov_mat, bl.view_mat, confidences)
        bl.view_cov_mat = omega
        bl.mean_posterior = BlackLitterman.calculate_posterior_mean(
            bl.pi, bl.cov_mat, bl.view_mat, bl.view_vec, omega
        )
        bl.cov_mat_posterior = BlackLitterman.calculate_posterior_cov_mat(
            bl.cov_mat, bl.view_mat, omega
        )
    else:
        # No views: posterior is the pure equilibrium prior.
        bl.calculate_posterior_distribution(confidence=default_confidence)

    # mean_posterior is weekly EXCESS; add rf_weekly for the total return Stage 5
    # expects (its placeholder was rf_weekly + premium).
    rf_weekly = rf_annual / WEEKS_PER_YEAR
    mu_weekly_total = np.asarray(bl.mean_posterior, dtype=float) + rf_weekly

    return mu_weekly_total, bl


# ---------------------------------------------------------------------------
# Diagnostics (run only when executed directly, mirroring Stage 3/5 style)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    universe = get_default_universe()
    cov_preferred = compute_cov_preferred(
        universe.returns, universe.market_returns, method=COV_METHOD,
    )

    mu_preferred, _bl = build_expected_returns(
        cov_preferred, ASSET_NAMES, views=VIEWS,
    )
    w_mkt = build_reference_weights(ASSET_NAMES, STRATEGIC_WEIGHTS)

    print("=== Strategic (neutral reference) weights ===")
    for name, w in zip(ASSET_NAMES, w_mkt):
        print(f"  {name:<14} {w:>7.2%}")

    # Effective lambda and the market Sharpe it implies (the calibration target).
    sigma_mkt = float(np.sqrt(w_mkt @ np.asarray(
        cov_preferred.reindex(index=ASSET_NAMES, columns=ASSET_NAMES), float) @ w_mkt))
    lam_eff = calibrate_lambda(np.asarray(
        cov_preferred.reindex(index=ASSET_NAMES, columns=ASSET_NAMES), float),
        w_mkt, TARGET_MARKET_SHARPE) if LAMBDA is None else LAMBDA
    print("\n=== Risk-aversion calibration ===")
    print(f"  lambda (effective)     : {lam_eff:>8.3f}"
          f"{'  (calibrated)' if LAMBDA is None else '  (pinned)'}")
    print(f"  Neutral-portfolio vol  : {sigma_mkt:>8.2%}  (annual)")
    print(f"  Implied market Sharpe  : {lam_eff * sigma_mkt:>8.3f}")

    # Covariance conditioning. cond = largest/smallest eigenvalue ratio; it is
    # scale-invariant, so the annual matrix gives the same number as the weekly
    # one BL uses. High values flag near-collinear assets (e.g. the Global/
    # Europe/Tech overlap) that will destabilise the Stage 5 optimiser, even
    # though the shrunk covariance keeps the BL step itself well-behaved.
    cond = np.linalg.cond(np.asarray(
        cov_preferred.reindex(index=ASSET_NAMES, columns=ASSET_NAMES), float))
    flag = ("OK" if cond < 1e3 else
            "moderate — watch optimiser weights" if cond < 1e6 else
            "ILL-CONDITIONED — de-overlap the universe")
    print("\n=== Covariance conditioning ===")
    print(f"  Condition number : {cond:>12.1f}  ({flag})")
    print("\n=== Equilibrium prior (annual, excess) ===")
    for name, p in zip(ASSET_NAMES, _bl.pi * WEEKS_PER_YEAR):
        print(f"  {name:<14} {p:>8.4%}")

    print("\n=== mu_preferred (Stage 4 output) ===")
    print(f"  {'asset':<14} {'weekly':>10}  {'annual (x52)':>14}")
    for name, m in zip(ASSET_NAMES, mu_preferred):
        print(f"  {name:<14} {m:>10.6f}  {m * WEEKS_PER_YEAR:>13.4%}")
    print(f"  rf (annual)    {RF_ANNUAL:>33.4%}")

    if VIEWS:
        ci = _bl.calculate_consistency_index()
        print("\n=== View diagnostics ===")
        print(f"  {len(VIEWS)} view(s) applied "
              f"(default confidence={VIEW_CONFIDENCE} where unspecified)")
        for spec in VIEWS:
            conf = _view_confidence(spec, VIEW_CONFIDENCE)
            if spec[0] == "absolute":
                label = f"{spec[1]} = {spec[2]:.2%} p.a."
            else:
                label = f"{spec[1]} - {spec[2]} = {spec[3]:+.2%} p.a."
            print(f"    [{spec[0]:<8}] {label:<28} confidence={conf:.2f}")
        print(f"  Consistency index: {ci:.4f}  (near 1 = views compatible with "
              f"equilibrium; near 0 = in tension)")
    else:
        print("\n(no views configured — mu_preferred is the pure equilibrium return)")
