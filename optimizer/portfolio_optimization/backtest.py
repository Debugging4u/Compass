"""
Stage 9 — Out-of-sample backtest
portfolio_optimization/backtest.py

Walk-forward test of every portfolio construction against naive benchmarks,
on this universe's own data. Answers the question the whole pipeline has been
circling: does any of the optimisation machinery beat equal-weight (and your
own policy weights) OUT of sample?

Method (strict, no look-ahead):
  At each rebalance date the estimators see ONLY the trailing window. We
  re-estimate covariance (Stage 3 recipe) and Black-Litterman mu (Stage 4) on
  that window, build each portfolio through the PUBLIC Stage 5 solvers, then
  hold those weights forward over the next out-of-sample block and record the
  realised returns. The estimation window and the realised block never overlap.

Single source of truth:
  - Covariance via `compute_cov_preferred` (Stage 3) — the SAME recipe the live
    pipeline uses; no duplicated Ledoit-Wolf call here.
  - mu via `build_expected_returns(..., views=VIEWS)` (Stage 4) — the SAME BL
    construction, prior, and view as the deployed model.
  - Weights via `constrained_gmv` / `constrained_max_sharpe` /
    `optimal_portfolio` (Stage 5) — the SAME solvers, so they inherit whatever
    constraints (and MAX_WEIGHT) those functions currently enforce.

Realised-return convention:
  Returns are weekly LOG returns (Stage 3). For realised P&L they are converted
  to simple returns (expm1). Within a holding block the portfolio is rebalanced
  weekly back to the block's target weights, so the realised weekly portfolio
  return is w . r_simple. This is the standard backtest simplification; it keeps
  weight-drift bookkeeping out and ignores turnover cost (which ties to the
  deferred turnover constraint — add it here when Stage 8 exists).

Caveat — sample size:
  History is short (~4 years of weekly data), so the walk-forward produces only
  a handful of rebalances. Realised-Sharpe differences on this few observations
  are NOISY. Read the table as "is one clearly better / clearly worse", not as a
  precise ranking. Because BL anchors on the fixed STRATEGIC_WEIGHTS prior, the
  optimised portfolios will tend to track that policy allocation — so this is in
  part a test of "policy weights + one view + shrinkage" versus 1/N.

Run from repo root:
    python -m portfolio_optimization.backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Stage 3 — weekly log returns + the covariance recipe (single source of truth).
from portfolio_optimization.correlation_matrix import returns, compute_cov_preferred

# Stage 4 — Black-Litterman mu, the policy prior, and the live views.
from portfolio_optimization.expected_returns import (
    build_expected_returns,
    build_reference_weights,
    STRATEGIC_WEIGHTS,
    VIEWS,
)

# Stage 5 — public solvers and shared constants.
from portfolio_optimization.optimal_portfolio import (
    constrained_gmv,
    constrained_max_sharpe,
    optimal_portfolio,
    ASSET_NAMES,
    RF_ANNUAL,
    WEEKS_PER_YEAR,
    TARGET_RETURN_ANNUAL,
)

# ---------------------------------------------------------------------------
# Configuration (no magic numbers — tune here)
# ---------------------------------------------------------------------------
WINDOW_MODE = "rolling"      # "rolling" (trailing WINDOW_WEEKS) or "expanding"
WINDOW_WEEKS = 26           # trailing estimation window in rolling mode (~2y)
MIN_TRAIN_WEEKS = 26        # first trade only after this much history
REBAL_WEEKS = 13             # re-optimise every N weeks (~quarterly)

# ---------------------------------------------------------------------------
# Experiment toggle: neutral prior vs. live policy prior
# ---------------------------------------------------------------------------
# True  -> BL uses an EQUAL-WEIGHT, opinion-free prior and NO views. This is a
#          clean test of the shrinkage + optimisation MACHINERY: any win over
#          1/N here is repeatable, not an artefact of hindsight beliefs.
# False -> reproduces the live deployed model (STRATEGIC_WEIGHTS prior + the
#          configured VIEWS). Informative, but partly a backtest of beliefs you
#          authored with full-sample hindsight.
NEUTRAL_PRIOR = False

# ---------------------------------------------------------------------------
# Strategy builders: each maps (mu_weekly, cov_annual_np) -> weight vector.
# The target-return builder can raise ValueError when the fixed target is
# unattainable for a given window; the walk-forward loop handles that.
# ---------------------------------------------------------------------------
def _w_equal(mu_weekly: np.ndarray, cov_np: np.ndarray) -> np.ndarray:
    n = len(ASSET_NAMES)
    return np.full(n, 1.0 / n)


def _w_strategic(mu_weekly: np.ndarray, cov_np: np.ndarray) -> np.ndarray:
    # The BL prior itself, held as a static policy allocation.
    return build_reference_weights(ASSET_NAMES, STRATEGIC_WEIGHTS)


def _w_gmv(mu_weekly: np.ndarray, cov_np: np.ndarray) -> np.ndarray:
    return constrained_gmv(mu_weekly, cov_np)


def _w_max_sharpe(mu_weekly: np.ndarray, cov_np: np.ndarray) -> np.ndarray:
    return constrained_max_sharpe(mu_weekly, cov_np, RF_ANNUAL)


def _w_target(mu_weekly: np.ndarray, cov_np: np.ndarray) -> np.ndarray:
    return optimal_portfolio(mu_weekly, cov_np, TARGET_RETURN_ANNUAL)


STRATEGIES: dict[str, callable] = {
    "1/N (equal)": _w_equal,
    "Strategic (prior)": _w_strategic,
    "GMV": _w_gmv,
    "Max Sharpe": _w_max_sharpe,
    f"Target {TARGET_RETURN_ANNUAL:.0%}": _w_target,
}


# ---------------------------------------------------------------------------
# Per-window estimation: reproduce the live pipeline on a returns slice.
# ---------------------------------------------------------------------------
def estimate_window(window_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Re-estimate (mu_weekly, cov_annual_np) on a trailing returns window.

    Uses the exact Stage 3 covariance recipe and the exact Stage 4 BL build, so
    the walk-forward sees the same inputs the live model would have seen had it
    been run on that date.
    """
    cov_df = compute_cov_preferred(window_returns)                  # Stage 3

    if NEUTRAL_PRIOR:
        sw, views = None, []                       # equal-weight prior, no views
    else:
        sw, views = STRATEGIC_WEIGHTS, VIEWS       # live policy prior + views

    mu_weekly, _ = build_expected_returns(                          # Stage 4
        cov_df, ASSET_NAMES,
        strategic_weights=sw,
        views=views,
    )
    cov_np = np.asarray(cov_df.reindex(index=ASSET_NAMES, columns=ASSET_NAMES), float)
    cov_np = (cov_np + cov_np.T) / 2.0
    return np.asarray(mu_weekly, dtype=float), cov_np


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run the walk-forward backtest.

    Returns:
        weekly_returns: realised weekly SIMPLE returns per strategy (OOS only).
        equity: cumulative growth-of-1 curves per strategy.
        infeasible: per-strategy count of rebalances where the target was
            unattainable and the previous weights were carried forward.
    """
    simple = np.expm1(returns)                       # weekly log -> simple
    dates = returns.index
    n_weeks = len(returns)
    names = list(STRATEGIES.keys())

    # Rebalance dates: first at MIN_TRAIN_WEEKS, then every REBAL_WEEKS, leaving
    # at least one forward week to realise.
    start = max(MIN_TRAIN_WEEKS, WINDOW_WEEKS if WINDOW_MODE == "rolling" else MIN_TRAIN_WEEKS)
    rebal_idx = list(range(start, n_weeks, REBAL_WEEKS))
    if not rebal_idx:
        raise SystemExit(
            f"No rebalance dates: need > {start} weeks of history, have {n_weeks}. "
            f"Lower WINDOW_WEEKS / MIN_TRAIN_WEEKS or extend the data start in Stage 3."
        )

    weekly_rows: dict[str, list[float]] = {nm: [] for nm in names}
    oos_dates: list[pd.Timestamp] = []
    infeasible: dict[str, int] = {nm: 0 for nm in names}
    last_w: dict[str, np.ndarray] = {nm: np.full(len(ASSET_NAMES), 1.0 / len(ASSET_NAMES)) for nm in names}

    for k, t in enumerate(rebal_idx):
        lo = 0 if WINDOW_MODE == "expanding" else t - WINDOW_WEEKS
        window = returns.iloc[lo:t]                  # strictly before t
        mu_weekly, cov_np = estimate_window(window)

        # Build target weights for each strategy at this rebalance.
        weights_now: dict[str, np.ndarray] = {}
        for nm, builder in STRATEGIES.items():
            try:
                w = builder(mu_weekly, cov_np)
            except ValueError:                       # target unattainable this window
                infeasible[nm] += 1
                w = last_w[nm]                       # carry previous weights forward
            except RuntimeError:                     # solver failure
                infeasible[nm] += 1
                w = last_w[nm]
            weights_now[nm] = np.asarray(w, dtype=float)
            last_w[nm] = weights_now[nm]

        # Realise forward over [t, t_next), no overlap with the window.
        t_next = rebal_idx[k + 1] if k + 1 < len(rebal_idx) else n_weeks
        block = simple.iloc[t:t_next]
        for ts, row in block.iterrows():
            r = row.to_numpy()
            oos_dates.append(ts)
            for nm in names:
                weekly_rows[nm].append(float(weights_now[nm] @ r))

    weekly_returns = pd.DataFrame(weekly_rows, index=pd.Index(oos_dates, name="date"))
    equity = (1.0 + weekly_returns).cumprod()
    return weekly_returns, equity, infeasible


# ---------------------------------------------------------------------------
# Realised performance statistics
# ---------------------------------------------------------------------------
def performance_table(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised realised stats per strategy from weekly SIMPLE returns."""
    out = {}
    for nm in weekly_returns.columns:
        r = weekly_returns[nm].to_numpy()
        n = len(r)
        growth = float(np.prod(1.0 + r))
        ann_ret = growth ** (WEEKS_PER_YEAR / n) - 1.0 if n > 0 else np.nan
        ann_vol = float(np.std(r, ddof=1)) * np.sqrt(WEEKS_PER_YEAR) if n > 1 else np.nan
        sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol and ann_vol > 1e-12 else np.nan
        curve = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(curve)
        max_dd = float(np.min(curve / peak - 1.0)) if n > 0 else np.nan
        out[nm] = {
            "Ann.return": ann_ret,
            "Ann.vol": ann_vol,
            "Sharpe": sharpe,
            "MaxDD": max_dd,
            "TotalRet": growth - 1.0,
        }
    table = pd.DataFrame(out).T
    return table.sort_values("Sharpe", ascending=False)


def plot_equity(equity: pd.DataFrame) -> None:
    """Plot growth-of-1 equity curves for every strategy."""
    # High-contrast palette + distinct line styles, so the curves stay
    # separable even where they overlap (and in greyscale / for colour-blind
    # readers). Styles cycle independently of colour for extra separation.
    palette = {
        "1/N (equal)":       ("#000000", "-"),    # black, solid  — the benchmark
        "Strategic (prior)": ("#E69F00", "--"),   # orange, dashed
        "GMV":               ("#0072B2", "-."),   # blue, dash-dot
        "Max Sharpe":        ("#009E73", ":"),    # green, dotted
    }
    fallback_color = "#D55E00"   # vermillion — for the Target curve (label varies)
    cycle = ["#000000", "#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, nm in enumerate(equity.columns):
        color, ls = palette.get(nm, (fallback_color, (0, (3, 1, 1, 1))))
        # If a name isn't in the palette (e.g. the "Target 10%" label), fall
        # back to the cycle by position so two unknowns never collide.
        if nm not in palette:
            color = cycle[i % len(cycle)]
        ax.plot(equity.index, equity[nm], label=nm,
                color=color, linestyle=ls, linewidth=1.8)

    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title(
        f"Out-of-sample equity curves — {WINDOW_MODE} window, "
        f"rebalanced every {REBAL_WEEKS}w"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of 1 (OOS)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    weekly_returns, equity, infeasible = run_backtest()

    n_oos = len(weekly_returns)
    print(f"\n{'='*64}")
    print("  OUT-OF-SAMPLE BACKTEST")
    print(f"{'='*64}")
    print(f"  window mode        : {WINDOW_MODE}"
          + (f" ({WINDOW_WEEKS}w)" if WINDOW_MODE == 'rolling' else ''))
    print(f"  rebalance every    : {REBAL_WEEKS} weeks")
    print(f"  OOS observations   : {n_oos} weeks "
          f"({n_oos / WEEKS_PER_YEAR:.1f} years)")
    print(f"  target return      : {TARGET_RETURN_ANNUAL:.2%}")
    if any(infeasible.values()):
        flagged = {k: v for k, v in infeasible.items() if v}
        print(f"  carried-forward    : {flagged}  "
              f"(rebalances where target/solver was infeasible)")
    if n_oos < 52:
        print("  ! fewer than ~1 year of OOS weeks — treat all rankings as")
        print("    indicative only; differences are within noise.")

    table = performance_table(weekly_returns)
    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    print(f"\n  Realised performance (sorted by Sharpe):\n")
    print(table.to_string())
    print(f"\n{'='*64}\n")

    plot_equity(equity)