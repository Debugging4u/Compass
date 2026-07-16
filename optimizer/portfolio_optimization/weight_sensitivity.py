"""
Robustness diagnostic — weight sensitivity under mu jitter
portfolio_optimization/weight_sensitivity.py

Perturbs the expected-return vector with noise drawn from the sampling
distribution of the mean estimator, re-solves the SAME target-return
portfolio many times, and reports how much each asset's weight moves across
draws. Large dispersion flags an allocation that is reading estimation noise
rather than signal — the warning we expect for near-collinear sleeves
(GRID / Industrials / Norway, which talk to each other through Schneider,
ABB and Siemens).

Why this jitter, not an arbitrary percentage:
  The standard error of a sample mean over T observations is sqrt(diag(Σ)/T).
  Perturbing mu by draws from N(0, Σ_weekly / T) therefore matches the actual
  estimation uncertainty in the input, and uses the FULL covariance so that
  cross-asset estimation errors are correlated through Σ — which is precisely
  the channel that destabilises collinear sleeves. This is the same object
  Michaud resampling perturbs, so this diagnostic and the planned Stage-10
  resampling step share one definition of "noise".

Design:
  - Calls the PUBLIC Stage 5 entry point `optimal_portfolio(...)`, so the
    jittered solves inherit WHATEVER constraints that function currently
    enforces. As of now that is long-only (w >= 0) and sum-to-one only; the
    max-weight / asset-class caps in context-doc Section 13 are still [DECIDE]
    and are NOT yet in the target-return solver path (consistent with the base
    portfolio showing GRID at ~30%+). Once Stage 6 wires caps in, this
    diagnostic picks them up automatically — no change needed here.
  - The return target is fixed (your chosen deliverable). Because jittered mu
    shifts the attainable range, some draws push the target outside the
    feasible band and raise ValueError; these are caught, skipped, and the
    infeasible fraction is reported. A high fraction means the target sits
    near the edge of the frontier — itself a finding.

Unit conventions (consistent with Stages 3–5):
  - mu_preferred is WEEKLY, total, arithmetic.
  - COV is ANNUALISED; the weekly covariance is COV / WEEKS_PER_YEAR.
  - optimal_portfolio() takes weekly mu and annualises internally, so the
    jittered mu is passed in weekly space.

Run from repo root:
    python -m portfolio_optimization.weight_sensitivity
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Stage 3 — universe loader + covariance recipe. Weekly return history is
# only used for T, the effective sample size that scales the jitter.
#
# NOTE: this module previously imported `mu_preferred` from expected_returns
# and `COV` from optimal_portfolio at module level — neither name actually
# existed there (both are built only inside those modules' `__main__` blocks),
# so this import was broken before this refactor too. Fixed here by building
# mu/cov locally, the same way optimal_portfolio.py's own __main__ does.
from portfolio_optimization.correlation_matrix import (
    ASSET_NAMES,
    COV_METHOD,
    compute_cov_preferred,
    get_default_universe,
)

from portfolio_optimization.expected_returns import (
    build_expected_returns,
    RF_ANNUAL,
    VIEWS,
    WEEKS_PER_YEAR,
)

from portfolio_optimization.optimal_portfolio import (
    optimal_portfolio,
    portfolio_stats,
    TARGET_RETURN_ANNUAL,
)
# ---------------------------------------------------------------------------
# Configuration (diagnostic-only knobs — none of these are defined upstream,
# so this module owns them. The target return is NOT here: it is imported from
# Stage 5 above, per the single-source-of-truth rule.)
# ---------------------------------------------------------------------------

# Number of Monte-Carlo draws. 2000–5000 gives stable summary statistics;
# raise it if the reported std values still wobble between runs.
N_DRAWS = 3000

# Jitter magnitude. 1.0 = one-standard-error perturbation (the honest default,
# matching the true sampling uncertainty of the mean). Use >1 to stress harder,
# <1 to probe local stability.
NOISE_SCALE = 1.0

# Reproducibility.
SEED = 42


# ---------------------------------------------------------------------------
# Core diagnostic
# ---------------------------------------------------------------------------

def weight_sensitivity(
    mu_weekly: np.ndarray,
    cov_annual: np.ndarray,
    target_return_annual: float,
    n_obs: int,
    n_draws: int = N_DRAWS,
    noise_scale: float = NOISE_SCALE,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Re-solve the target-return portfolio under mu jitter.

    Perturbs mu with noise ~ N(0, noise_scale**2 * Σ_weekly / n_obs), where
    Σ_weekly = cov_annual / WEEKS_PER_YEAR, then re-solves the fixed-target
    long-only portfolio for each draw.

    Args:
        mu_weekly: Base weekly expected-return vector, shape (n,).
        cov_annual: Annualised covariance matrix, shape (n, n).
        target_return_annual: The fixed annualised return target whose
            allocation is being stress-tested.
        n_obs: Number of weekly observations behind the mean estimate (T).
            Scales the jitter — fewer observations, noisier mu, wider weights.
        n_draws: Number of Monte-Carlo draws.
        noise_scale: Multiplier on the one-standard-error jitter.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of:
          - w_samples: feasible weight draws, shape (n_feasible, n).
          - base_weights: unperturbed target-return weights, shape (n,).
          - n_infeasible: number of draws skipped because the fixed target
            fell outside the (shifted) attainable range.
    """
    rng = np.random.default_rng(seed)

    # Sampling covariance of the WEEKLY mean estimate. Full matrix, so the
    # jitter inherits the cross-asset correlation structure of Σ.
    cov_weekly = cov_annual / WEEKS_PER_YEAR
    mean_sampling_cov = (noise_scale ** 2) * cov_weekly / n_obs

    # Symmetrise defensively before the multivariate-normal Cholesky.
    mean_sampling_cov = (mean_sampling_cov + mean_sampling_cov.T) / 2.0

    n = len(mu_weekly)
    deltas = rng.multivariate_normal(np.zeros(n), mean_sampling_cov, size=n_draws)

    # Unperturbed reference allocation.
    base_weights = optimal_portfolio(mu_weekly, cov_annual, target_return_annual)

    w_samples = []
    n_infeasible = 0
    for d in deltas:
        mu_draw = mu_weekly + d
        try:
            w = optimal_portfolio(mu_draw, cov_annual, target_return_annual)
        except ValueError:
            # Fixed target fell outside [min, max] of the jittered mu.
            n_infeasible += 1
            continue
        except RuntimeError:
            # Solver failure on this draw — treat as infeasible but keep going.
            n_infeasible += 1
            continue
        w_samples.append(w)

    return np.array(w_samples), base_weights, n_infeasible


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(
    w_samples: np.ndarray,
    base_weights: np.ndarray,
    asset_names: list[str],
    n_infeasible: int,
    n_draws: int,
    watch: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Print and return a per-asset weight-dispersion table.

    Columns:
        base     unperturbed target-return weight
        mean     mean weight across feasible draws
        std      standard deviation of weight across draws (the key number)
        min/max  range across draws
        spread   max - min
        cv       coefficient of variation (std / mean), dimensionless;
                 lets you compare stability across assets of different size.

    Args:
        w_samples: Feasible weight draws, shape (n_feasible, n).
        base_weights: Unperturbed weights, shape (n,).
        asset_names: Asset names aligned to the weight columns.
        n_infeasible: Draws skipped (target out of range / solver failure).
        n_draws: Total draws attempted.
        watch: Optional assets to flag with a marker in the printout.
            Defaults to none.

    Returns:
        The summary DataFrame, sorted by std descending (most unstable first).
    """
    n_feasible = len(w_samples)
    mean_w = w_samples.mean(axis=0)
    std_w = w_samples.std(axis=0, ddof=1)
    min_w = w_samples.min(axis=0)
    max_w = w_samples.max(axis=0)
    spread = max_w - min_w
    # Guard against divide-by-zero for assets the optimiser never funds.
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean_w > 1e-6, std_w / mean_w, np.nan)

    table = pd.DataFrame(
        {
            "base": base_weights,
            "mean": mean_w,
            "std": std_w,
            "min": min_w,
            "max": max_w,
            "spread": spread,
            "cv": cv,
        },
        index=asset_names,
    ).sort_values("std", ascending=False)

    infeasible_pct = 100.0 * n_infeasible / n_draws

    print(f"\n{'='*72}")
    print("  WEIGHT SENSITIVITY  —  mu jittered at the mean's sampling SE")
    print(f"{'='*72}")
    print(f"  draws attempted     : {n_draws}")
    print(f"  feasible draws      : {n_feasible}")
    print(f"  infeasible (skipped): {n_infeasible}  ({infeasible_pct:.1f}%)")
    if infeasible_pct > 10.0:
        print("    ! >10% infeasible — the fixed target sits near the edge of")
        print("      the attainable frontier. Read the weights with caution.")
    print(f"\n  Per-asset weight dispersion (sorted by std, most unstable first):\n")

    fmt = "  {name:<14}{base:>8}{mean:>8}{std:>8}{min:>8}{max:>8}{spread:>9}{cv:>8}"
    print(fmt.format(name="asset", base="base", mean="mean", std="std",
                     min="min", max="max", spread="spread", cv="cv"))
    print("  " + "-" * 70)
    for name, row in table.iterrows():
        marker = "  <--" if name in watch else ""
        cv_str = f"{row['cv']:.2f}" if np.isfinite(row["cv"]) else "  n/a"
        print(
            f"  {name:<14}"
            f"{row['base']:>7.2%}"
            f"{row['mean']:>8.2%}"
            f"{row['std']:>8.2%}"
            f"{row['min']:>8.2%}"
            f"{row['max']:>8.2%}"
            f"{row['spread']:>9.2%}"
            f"{cv_str:>8}"
            f"{marker}"
        )

    print(f"\n{'-'*72}")
    print("  How to read this:")
    print("   - std is the headline number: how much the weight moves under")
    print("     one-SE noise in mu. A few % is fine; a sleeve whose std is a")
    print("     large fraction of its own weight is noise-driven.")
    print("   - cv normalises std by mean, so small-but-volatile sleeves stand")
    print("     out. High cv plus a large spread on a sleeve means the")
    print("     collinearity is leaking into its allocation, and points to an")
    print("     asset-class cap (Stage 6) or Michaud resampling as the fix —")
    print("     not a mu re-forecast.")
    print(f"{'='*72}\n")

    return table

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    universe = get_default_universe()
    cov_preferred = compute_cov_preferred(
        universe.returns, universe.market_returns, method=COV_METHOD,
    )
    mu_preferred, _bl = build_expected_returns(
        cov_preferred, ASSET_NAMES, views=VIEWS,
    )

    mu_weekly = np.asarray(mu_preferred, dtype=float)
    cov_annual = np.asarray(cov_preferred, dtype=float)
    cov_annual = (cov_annual + cov_annual.T) / 2.0
    n_obs = len(universe.returns)  # T: weeks of history behind the mean estimate

    print(f"\nTarget return (annualised): {TARGET_RETURN_ANNUAL:.2%}")
    print(f"Sample size T (weeks)     : {n_obs}")
    print(f"Noise scale               : {NOISE_SCALE}  "
          f"({'one-SE jitter' if NOISE_SCALE == 1.0 else 'scaled jitter'})")

    w_samples, base_weights, n_infeasible = weight_sensitivity(
        mu_weekly,
        cov_annual,
        TARGET_RETURN_ANNUAL,
        n_obs=n_obs,
        n_draws=N_DRAWS,
        noise_scale=NOISE_SCALE,
        seed=SEED,
    )

    if len(w_samples) == 0:
        raise SystemExit(
            "All draws were infeasible — the target return is unattainable for "
            "essentially every jittered mu. Lower TARGET_RETURN_ANNUAL or check "
            "that it lies inside the base attainable range."
        )

    # Show the unperturbed deliverable for reference.
    portfolio_stats(
        base_weights, mu_weekly, cov_annual, RF_ANNUAL, ASSET_NAMES,
        f"Base target-return portfolio (target={TARGET_RETURN_ANNUAL:.2%} p.a.)",
    )

    summarise(w_samples, base_weights, ASSET_NAMES, n_infeasible, N_DRAWS)