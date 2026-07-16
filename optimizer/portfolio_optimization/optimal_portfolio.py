"""
Stage 5 — Optimisation (Sharpe max / MVO)
portfolio_optimization/optimal_portfolio.py

Computes the long-only constrained efficient frontier, key portfolios,
and prints a statistics table so you can decide which portfolio to construct.

Covariance input:
  Consumes `cov_preferred` from Stage 3 (currently the Ledoit-Wolf
  constant-variance shrinkage estimate, annualised). The matrix is converted
  to a plain ndarray at import so every solver/quad_form call receives a
  constant numpy array (no pandas/cvxpy version fragility, no index-alignment
  surprises). To compare estimators, change the single alias in Stage 3
  (`cov_preferred = ...`) — nothing in this file needs to change.

Key portfolios computed:
  - GMV (constrained):        minimum variance, long-only
  - Max Sharpe (constrained): highest Sharpe ratio, long-only
  - Target-return portfolio:  minimum variance for a user-specified return

Plots produced:
  1. Efficient frontier (long-only) + CML + key portfolio scatter + asset scatter
  2. Weight stackplot across the frontier (shows how allocations shift with return target)

Statistics printed:
  - Annualised return, annualised vol, Sharpe ratio, and weights for each key portfolio

Unit conventions (kept consistent throughout):
  - Returns: weekly LOG returns (from Stage 3).
  - Mean annualisation: linear, mu_annual = mu_weekly * 52 (log-return convention).
  - Variance annualisation: linear, handled in Stage 3 (cov * 52); vol = sqrt(diag).
  - This linear scaling assumes weekly returns are serially uncorrelated (iid).
    If weekly autocorrelation is material, sqrt-time scaling is biased — flagged
    for review.
"""
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# codelib — use directly
from codelib.portfolio_optimization.mean_variance import (
    portfolio_mean,
    portfolio_std,
)
from codelib.visualization.layout import DefaultStyle

# Stage 3 output: annualised preferred covariance (LW constant-variance) +
# weekly returns DataFrame with named columns.
# NOTE: requires Stage 3 to expose `cov_preferred = cov_lw_constant`.
from portfolio_optimization.correlation_matrix import cov_preferred, returns

from portfolio_optimization.expected_returns import (
    build_expected_returns,
    RF_ANNUAL,
    WEEKS_PER_YEAR,
    VIEWS,
)

DefaultStyle()

ASSET_NAMES = list(returns.columns)

# --- Convert the covariance to a plain ndarray once, here. ---------------
# Everything downstream (cvxpy.quad_form, portfolio_std/mean) expects a
# constant matrix. Symmetrise to clean up any tiny numerical asymmetry from
# the shrinkage reconstruction before it reaches the solver.
COV = np.asarray(cov_preferred, dtype=float)
COV = (COV + COV.T) / 2.0

# ---------------------------------------------------------------------------
# Constraints (Stage 5 box constraint — acts as a regulariser, per
# Jagannathan & Ma 2003). Long-only is enforced separately via w >= 0.
# Set MAX_WEIGHT = None to disable the cap (reproduces the old uncapped
# behaviour, e.g. for diagnostics).
# ---------------------------------------------------------------------------

MAX_WEIGHT: float | None = None

# Sanity guard: with a per-asset cap, the weights can only sum to 1 if there
# is enough headroom across the universe. Caught here at import rather than as
# an opaque solver failure later.
if MAX_WEIGHT is not None and len(ASSET_NAMES) * MAX_WEIGHT < 1.0 - 1e-12:
    raise ValueError(
        f"MAX_WEIGHT={MAX_WEIGHT} is infeasible for {len(ASSET_NAMES)} assets: "
        f"{len(ASSET_NAMES)} * {MAX_WEIGHT} = {len(ASSET_NAMES) * MAX_WEIGHT:.2f} < 1. "
        f"Raise MAX_WEIGHT or add assets."
    )

# ---------------------------------------------------------------------------
# Core solver — every portfolio in this file goes through here
# ---------------------------------------------------------------------------
def _attainable_return_range(
    mu_annual: np.ndarray,
    max_weight: float | None,
) -> tuple[float, float]:
    """Return the (min, max) annualised return achievable long-only under a cap.

    Without a cap the range is simply [min asset return, max asset return],
    since a long-only portfolio return is a convex combination of asset
    returns. With a per-asset cap the extremes are no longer attainable: the
    maximum is found by greedily filling the highest-return assets at the cap
    until the budget is exhausted (and symmetrically for the minimum).

    Args:
        mu_annual: Annualised expected return vector, shape (n,).
        max_weight: Per-asset upper bound, or None for no cap.

    Returns:
        Tuple (mu_min_attainable, mu_max_attainable), annualised.
    """
    if max_weight is None:
        return float(np.min(mu_annual)), float(np.max(mu_annual))

    def _greedy_fill(order: np.ndarray) -> float:
        budget, ret = 1.0, 0.0
        for i in order:
            w_i = min(max_weight, budget)
            ret += w_i * mu_annual[i]
            budget -= w_i
            if budget <= 1e-15:
                break
        return ret

    order_desc = np.argsort(mu_annual)[::-1]
    order_asc = np.argsort(mu_annual)
    return _greedy_fill(order_asc), _greedy_fill(order_desc)
def _solve_min_variance(
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
    target_return: float | None = None,
    no_short_selling: bool = True,
    max_weight: float | None = MAX_WEIGHT,
) -> np.ndarray | None:
    """Solve a single minimum-variance QP with optional return target and cap.

    Args:
        mu_est: Weekly expected return vector, shape (n,).
        cov_mat: Annualised covariance matrix, shape (n, n).
        target_return: Annualised target return. If None, minimises variance
            without a return constraint (gives constrained GMV).
        no_short_selling: If True enforces w >= 0.
        max_weight: Per-asset upper bound (box constraint). Defaults to the
            module-level MAX_WEIGHT. Pass None to disable the cap.

    Returns:
        Weight vector of shape (n,), or None if infeasible.
    """
    n = len(mu_est)
    mu_annual = mu_est * WEEKS_PER_YEAR  # annualise for constraint consistency

    w = cp.Variable(n)
    # assume_PSD=True is safe here: the Ledoit-Wolf constant-variance
    # estimator is positive-definite by construction. Re-check this flag if
    # you ever swap in a different (possibly non-PSD) covariance estimate.
    objective = cp.Minimize(cp.quad_form(w, cov_mat, assume_PSD=True))

    constraints = [cp.sum(w) == 1.0]
    if target_return is not None:
        constraints.append(mu_annual @ w == target_return)
    if no_short_selling:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    return w.value

# ---------------------------------------------------------------------------
# Constrained GMV (long-only minimum variance — no return target)
# ---------------------------------------------------------------------------

def constrained_gmv(
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
) -> np.ndarray:
    """Compute the long-only Global Minimum Variance portfolio.

    NOTE: codelib's minimum_variance_portfolio() is analytical and allows
    short selling. This function uses cvxpy to enforce w >= 0. mu_est is
    accepted for a uniform signature but is unused (GMV is independent of mu).

    Args:
        mu_est: Weekly expected return vector, shape (n,).
        cov_mat: Annualised covariance matrix, shape (n, n).

    Returns:
        Weight vector of shape (n,).

    Raises:
        RuntimeError: If the QP is infeasible.
    """
    w = _solve_min_variance(mu_est, cov_mat, target_return=None, no_short_selling=True)
    if w is None:
        raise RuntimeError("Constrained GMV optimisation failed.")
    return w


# ---------------------------------------------------------------------------
# Constrained Max Sharpe (long-only tangency via frontier scan)
# ---------------------------------------------------------------------------

def constrained_max_sharpe(
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
    rf_annual: float,
    n_scan: int = 200,
) -> np.ndarray:
    """Find the long-only Max Sharpe portfolio by scanning the frontier.

    NOTE: codelib's tangency_portfolio() is analytical and allows short
    selling. The constrained tangency has no closed form, so we solve the
    frontier at n_scan return levels and pick the portfolio with the highest
    Sharpe ratio.

    Args:
        mu_est: Weekly expected return vector, shape (n,).
        cov_mat: Annualised covariance matrix, shape (n, n).
        rf_annual: Annual risk-free rate.
        n_scan: Number of return levels to scan. Default 200.

    Returns:
        Weight vector of shape (n,).
    """
    mu_annual = mu_est * WEEKS_PER_YEAR

    # Range: from constrained GMV return up to max individual asset return
    w_gmv = constrained_gmv(mu_est, cov_mat)
    mu_lo = float(portfolio_mean(w_gmv, mu_annual))
    mu_hi = float(np.max(mu_annual))

    targets = np.linspace(mu_lo, mu_hi, n_scan)
    best_w, best_sharpe = None, -np.inf

    for t in targets:
        w = _solve_min_variance(mu_est, cov_mat, target_return=t, no_short_selling=True)
        if w is None:
            continue
        port_ret = portfolio_mean(w, mu_annual)
        port_std = portfolio_std(w, cov_mat)
        sharpe = (port_ret - rf_annual) / port_std if port_std > 1e-10 else -np.inf
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_w = w.copy()

    if best_w is None:
        raise RuntimeError("Could not find a feasible Max Sharpe portfolio.")
    return best_w


# ---------------------------------------------------------------------------
# Constrained target-return portfolio
# ---------------------------------------------------------------------------
def optimal_portfolio(
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
    target_return_annual: float,
    max_weight: float | None = MAX_WEIGHT,
) -> np.ndarray:
    """Compute the long-only minimum-variance portfolio for a return target.

    Args:
        mu_est: Weekly expected return vector, shape (n,).
        cov_mat: Annualised covariance matrix, shape (n, n).
        target_return_annual: Desired annualised portfolio return.
        max_weight: Per-asset upper bound. Defaults to module-level MAX_WEIGHT.

    Returns:
        Weight vector of shape (n,).

    Raises:
        ValueError: If target_return_annual is outside the cap-attainable range.
        RuntimeError: If the solver fails.
    """
    mu_annual = mu_est * WEEKS_PER_YEAR
    mu_min, mu_max = _attainable_return_range(mu_annual, max_weight)

    if not (mu_min <= target_return_annual <= mu_max):
        cap_note = "" if max_weight is None else f" (with max_weight={max_weight})"
        raise ValueError(
            f"target_return_annual={target_return_annual:.4f} is outside the "
            f"attainable range [{mu_min:.4f}, {mu_max:.4f}]{cap_note}."
        )

    w = _solve_min_variance(
        mu_est, cov_mat,
        target_return=target_return_annual,
        no_short_selling=True,
        max_weight=max_weight,
    )
    if w is None:
        raise RuntimeError(
            f"Optimisation failed for target_return={target_return_annual:.4f}. "
            "Check that the covariance matrix is positive semi-definite."
        )
    return w


# ---------------------------------------------------------------------------
# Efficient frontier (constrained, for plotting)
# ---------------------------------------------------------------------------

def compute_constrained_frontier(
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
    n_points: int = 100,
    max_weight: float | None = MAX_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep target returns across the long-only, capped efficient frontier.

    Sweeps from the GMV return up to the cap-attainable maximum, so it traces
    only the efficient (upper) branch. Uses a single parametric cvxpy problem
    for efficiency.

    Args:
        mu_est: Weekly expected return vector, shape (n,).
        cov_mat: Annualised covariance matrix, shape (n, n).
        n_points: Number of points on the frontier. Default 100.
        max_weight: Per-asset upper bound. Defaults to module-level MAX_WEIGHT.

    Returns:
        Tuple of:
          - mu_frontier: annualised returns, shape (n_points,)
          - std_frontier: annualised vols, shape (n_points,)
          - w_frontier: weights, shape (n_points, n_assets)
    """
    n = len(mu_est)
    mu_annual = mu_est * WEEKS_PER_YEAR

    w_gmv = constrained_gmv(mu_est, cov_mat)
    mu_lo = float(portfolio_mean(w_gmv, mu_annual))
    _, mu_hi = _attainable_return_range(mu_annual, max_weight)
    targets = np.linspace(mu_lo, mu_hi, n_points)

    # parametric cvxpy problem — build once, solve repeatedly
    mu_target_param = cp.Parameter()
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1.0, mu_annual @ w == mu_target_param, w >= 0]
    if max_weight is not None:
        constraints.append(w <= max_weight)
    prob = cp.Problem(
        cp.Minimize(cp.quad_form(w, cov_mat, assume_PSD=True)),
        constraints,
    )

    mu_frontier, std_frontier, w_frontier = [], [], []

    for t in targets:
        mu_target_param.value = t
        prob.solve()
        if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
            continue
        weights = w.value
        mu_frontier.append(portfolio_mean(weights, mu_annual))
        std_frontier.append(portfolio_std(weights, cov_mat))
        w_frontier.append(weights.copy())

    return (
        np.array(mu_frontier),
        np.array(std_frontier),
        np.array(w_frontier),
    )

# ---------------------------------------------------------------------------
# Portfolio statistics helper
# ---------------------------------------------------------------------------

def portfolio_stats(
    weights: np.ndarray,
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
    rf_annual: float,
    asset_names: list[str],
    label: str,
) -> None:
    """Print annualised return, vol, Sharpe, and per-asset weights.

    Args:
        weights: Portfolio weight vector, shape (n,).
        mu_est: Weekly expected return vector.
        cov_mat: Annualised covariance matrix.
        rf_annual: Annual risk-free rate.
        asset_names: List of asset names.
        label: Portfolio name for the printed header.
    """
    mu_annual = mu_est * WEEKS_PER_YEAR
    ret = portfolio_mean(weights, mu_annual)
    vol = portfolio_std(weights, cov_mat)
    sharpe = (ret - rf_annual) / vol if vol > 1e-10 else float("nan")

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Ann. return  : {ret:>8.2%}")
    print(f"  Ann. vol     : {vol:>8.2%}")
    print(f"  Sharpe ratio : {sharpe:>8.4f}  (rf={rf_annual:.2%})")
    print(f"  Weights:")
    for name, w in zip(asset_names, weights):
        bar = "█" * int(w * 40)
        print(f"    {name:<14} {w:>6.2%}  {bar}")


# ---------------------------------------------------------------------------
# Plot 1: Efficient frontier + CML + key portfolios + individual assets
# ---------------------------------------------------------------------------

def plot_efficient_frontier(
    mu_frontier: np.ndarray,
    std_frontier: np.ndarray,
    mu_est: np.ndarray,
    cov_mat: np.ndarray,
    rf_annual: float,
    w_gmv: np.ndarray,
    w_max_sr: np.ndarray,
    asset_names: list[str],
    w_target: np.ndarray | None = None,
    target_label: str = "Target",
) -> None:
    """Plot the long-only efficient frontier with key reference points.

    Follows the plotting style of week_9/modern_portfolio_theory.ipynb.

    NOTE: asset_vols are sqrt(diag(cov_mat)). Because cov_mat is the
    Ledoit-Wolf constant-variance estimate, these are *shrunk* (regularised)
    volatilities, pulled toward the cross-sectional average — not the raw
    sample SDs. The asset scatter therefore reflects model-adjusted risk.

    Args:
        mu_frontier: Annualised returns along the frontier.
        std_frontier: Annualised vols along the frontier.
        mu_est: Weekly expected returns.
        cov_mat: Annualised covariance matrix.
        rf_annual: Annual risk-free rate.
        w_gmv: Long-only GMV weights.
        w_max_sr: Long-only Max Sharpe weights.
        asset_names: Asset names for annotation.
        w_target: Optional target-return weights; plotted as a diamond if given.
        target_label: Legend label for the target-return marker.
    """
    mu_annual = mu_est * WEEKS_PER_YEAR
    asset_vols = np.sqrt(np.diag(cov_mat))

    mu_gmv = portfolio_mean(w_gmv, mu_annual)
    std_gmv = portfolio_std(w_gmv, cov_mat)
    mu_tan = portfolio_mean(w_max_sr, mu_annual)
    std_tan = portfolio_std(w_max_sr, cov_mat)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Efficient frontier (long-only)
    ax.plot(std_frontier, mu_frontier, color="blue",
            label="Efficient frontier (long-only)", linewidth=2)

    # Capital Market Line: from rf to tangency, extended
    cml_x = np.array([0.0, std_tan * 1.8])
    cml_slope = (mu_tan - rf_annual) / std_tan
    cml_y = rf_annual + cml_slope * cml_x
    ax.plot(cml_x, cml_y, color="gray", linestyle="--", label="Capital Market Line")

    # Key portfolio scatter
    ax.scatter(std_gmv, mu_gmv, color="red", s=80, zorder=10, label="GMV portfolio")
    ax.scatter(std_tan, mu_tan, color="green", s=80, zorder=10, label="Max Sharpe portfolio")
    ax.scatter(0, rf_annual, color="cyan", s=80, zorder=10, label="Risk-free rate")

    # Target-return portfolio (optional)
    if w_target is not None:
        ax.scatter(
            portfolio_std(w_target, cov_mat),
            portfolio_mean(w_target, mu_annual),
            color="crimson", s=80, zorder=11, marker="D", label=target_label,
        )

    # Individual assets
    ax.scatter(asset_vols, mu_annual, color="black", s=50, zorder=8, label="Assets")
    for name, vol, ret in zip(asset_names, asset_vols, mu_annual):
        ax.annotate(name, (vol, ret), textcoords="offset points", xytext=(5, 3), fontsize=9)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_title("Efficient Frontier — Long-Only, No Short Selling")
    ax.set_xlabel(r"$\sigma_p$ (annualised std)")
    ax.set_ylabel(r"$\mu_p$ (annualised expected return)")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Plot 2: Weight stackplot across the frontier
# ---------------------------------------------------------------------------

def plot_weight_stackplot(
    mu_frontier: np.ndarray,
    w_frontier: np.ndarray,
    asset_names: list[str],
) -> None:
    """Show how asset allocations shift as the return target increases.

    Follows codelib/week6/visualization.py (create_stackplot).

    Args:
        mu_frontier: Annualised return targets along the frontier.
        w_frontier: Weight matrix, shape (n_points, n_assets).
        asset_names: Asset names for the legend.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.stackplot(mu_frontier, w_frontier.T, labels=asset_names)

    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Target Return (annualised)")
    ax.set_ylabel("Portfolio Weights")
    ax.set_title("Optimal Portfolio Weights for Different Target Returns (Long-Only)")
    ax.legend(loc="center", bbox_to_anchor=(0.5, -0.25),
              fancybox=True, shadow=True, ncol=len(asset_names))
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

TARGET_RETURN_ANNUAL = 0.105

if __name__ == "__main__":

    # vols are taken from COV — the SAME matrix the optimiser uses — so mu and
    # the risk model stay coherent.
    mu_preferred, _bl = build_expected_returns(
        cov_preferred, ASSET_NAMES, views=VIEWS,
    )
    rf_annual = RF_ANNUAL
    mu_weekly = mu_preferred
    mu_annual_arr = mu_weekly * WEEKS_PER_YEAR

    # --- mu diagnostics ---
    print("=== mu diagnostics (weekly) ===")
    for name, m in zip(ASSET_NAMES, mu_weekly):
        print(f"  {name:<14} weekly={m:.6f}  annual={m*WEEKS_PER_YEAR:.4%}")
    print(f"  rf (annual)  : {rf_annual:.4%}")

    # --- reference bookends (frame the frontier; not the deliverable) ---
    w_gmv = constrained_gmv(mu_weekly, COV)
    w_max_sr = constrained_max_sharpe(mu_weekly, COV, rf_annual)
    mu_gmv_annual = portfolio_mean(w_gmv, mu_annual_arr)
    mu_tan_annual = portfolio_mean(w_max_sr, mu_annual_arr)
    _, mu_max_attain = _attainable_return_range(mu_annual_arr, MAX_WEIGHT)

    portfolio_stats(w_gmv,    mu_weekly, COV, rf_annual, ASSET_NAMES, "GMV (long-only) — reference")
    portfolio_stats(w_max_sr, mu_weekly, COV, rf_annual, ASSET_NAMES, "Max Sharpe (long-only) — reference")

    # --- target selection context: where TARGET_RETURN_ANNUAL sits ---
    band = mu_max_attain - mu_gmv_annual
    frac = (TARGET_RETURN_ANNUAL - mu_gmv_annual) / band if band > 1e-12 else float("nan")
    cap_state = "None (no cap)" if MAX_WEIGHT is None else f"{MAX_WEIGHT:.0%}"
    print(f"\n{'='*55}")
    print("  TARGET SELECTION CONTEXT")
    print(f"{'='*55}")
    print(f"  weight cap            : {cap_state}")
    print(f"  GMV return (floor)    : {mu_gmv_annual:>8.2%}")
    print(f"  Max-Sharpe return     : {mu_tan_annual:>8.2%}")
    print(f"  attainable ceiling    : {mu_max_attain:>8.2%}")
    print(f"  your target           : {TARGET_RETURN_ANNUAL:>8.2%}")
    print(f"  headroom to ceiling   : {mu_max_attain - TARGET_RETURN_ANNUAL:>8.2%}")
    if np.isfinite(frac):
        print(f"  position in band      : {frac:>8.1%}  (0% = GMV, 100% = ceiling)")
        if frac > 0.85:
            print("    ! target is in the top 15% of the attainable band — expect")
            print("      elevated infeasible draws when mu is jittered.")

    # --- headline: the target-return portfolio ---
    try:
        w_target = optimal_portfolio(mu_weekly, COV, target_return_annual=TARGET_RETURN_ANNUAL)
    except ValueError as exc:
        raise SystemExit(
            f"\nTarget {TARGET_RETURN_ANNUAL:.2%} is not attainable: {exc}\n"
            f"Pick a target inside [{mu_gmv_annual:.2%}, {mu_max_attain:.2%}]."
        )
    portfolio_stats(w_target, mu_weekly, COV, rf_annual, ASSET_NAMES,
                    f">>> TARGET-RETURN PORTFOLIO  (target={TARGET_RETURN_ANNUAL:.2%} p.a.) <<<")

    # --- efficient frontier ---
    mu_f, std_f, w_f = compute_constrained_frontier(mu_weekly, COV, n_points=100)

    # --- plots ---
    plot_efficient_frontier(
        mu_frontier=mu_f, std_frontier=std_f,
        mu_est=mu_weekly, cov_mat=COV,
        rf_annual=rf_annual,
        w_gmv=w_gmv, w_max_sr=w_max_sr,
        asset_names=ASSET_NAMES,
        w_target=w_target,
        target_label=f"Target {TARGET_RETURN_ANNUAL:.1%}",
    )

    plot_weight_stackplot(mu_f, w_f, ASSET_NAMES)

    # --- signature tripwire: confirm the frontier honours the live cap ---
    import inspect
    print("\nglobal MAX_WEIGHT      :", MAX_WEIGHT)
    print("frontier default bound :",
          inspect.signature(compute_constrained_frontier).parameters["max_weight"].default)