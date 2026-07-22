"""
Stage 3 — Universe loading & covariance estimation
portfolio_optimization/correlation_matrix.py

Downloads the investable universe + market proxy, builds weekly log returns,
and estimates the preferred (Ledoit-Wolf) covariance.

Import-time contract:
    Importing this module does NOTHING besides defining constants and
    functions — no network call, no printing, no plotting. All of that used
    to run at module level (a look-ahead-adjacent anti-pattern already fixed
    once in expected_returns.py's mu_preferred; the fix here removes the
    remaining, larger instance, since every downstream stage imports from
    this module). Call `load_universe()` (or the memoised `get_default_universe()`)
    to actually fetch data.

Output contract (what downstream stages use):
    ASSET_NAMES         — asset order, derived from `tickers` alone (no
                           network needed to know it).
    get_default_universe() -> UniverseData(prices, returns, market_price,
                           market_returns) — memoised per process.
    compute_cov_preferred(returns_window, market_window=None, method=...)
                        — pure function, annualised covariance DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from codelib.statistics.robust_covariance import (
    ledoit_wolf_constant_variance,
    ledoit_wolf_single_index,
)
from codelib.statistics.moments import cov_to_corr_matrix

# ============================================================
# INVESTABLE UNIVERSE
# The full set of assets you can choose to hold and optimise over. `tickers`
# below picks the currently ACTIVE subset — comment/uncomment freely there,
# or (since the API/dashboard added a universe selector) pick a different
# subset per-request instead of editing this file at all.
# ============================================================

CANDIDATE_TICKERS = {
    "Global": "IWDA.AS",
    "Asia": "AAXJ",
    "Europe": "IMAE.AS",
    "Norway": "OSEBX.OL",
    "Technology": "XDWT.DE",
    "HBONDS": "0P0001ILS7.IR",
    "GRID": "GRID.DE",
    "Industrials": "ESIN.DE",
    "Atea": "ATEA.OL",
    "Austevoll": "AUSS.OL",
    "DNB": "DNB.OL",
    "Equinor": "EQNR.OL",
    "SalMar": "SALM.OL",
    "Veidekke": "VEI.OL",
}

# Currently active subset. Comment/uncomment names here to change the default
# CLI/import-time universe; extra names in CANDIDATE_TICKERS above are simply
# unused until picked, either here or via a request's `assets` list.
tickers = {name: CANDIDATE_TICKERS[name] for name in [
    "Global",
    "Asia",
    "Europe",
    "Norway",
    "Technology",
    "HBONDS",
    "GRID",
 #   "Industrials",
 #   "Atea",
 #   "Austevoll",
 #   "DNB",
 #   "Equinor",
 #   "SalMar",
 #   "Veidekke",
]}

# Asset order for every downstream stage. Derived from `tickers` directly (not
# from a downloaded frame's columns), so it is known without hitting the
# network and can't silently reorder if a ticker fails to download —
# `load_universe()` validates the downloaded frame against this list instead.
ASSET_NAMES: list[str] = list(tickers.keys())

# ============================================================
# MARKET PROXY (for the single-index shrinkage target only)
# Downloaded independently of the universe, so the single-index
# estimator always has a valid market series whether or not the
# global fund is one of the holdings above. This is a benchmark/
# factor input — never an entry in the covariance matrix.
# ============================================================

MARKET_PROXY_SYMBOL = "IWDA.AS"   # MSCI World — broad world-market proxy
MARKET_PROXY_NAME = "Market"      # internal column name, never a holding

DEFAULT_START = "2018-01-01"
COV_METHOD = "constant"           # the estimator Stage 4/5 build on — see below


@dataclass(frozen=True)
class UniverseData:
    """Downloaded prices + weekly log returns for the universe and market proxy."""

    prices: pd.DataFrame          # weekly close, columns = asset names
    returns: pd.DataFrame         # weekly log returns, columns = asset names
    market_price: pd.Series       # weekly close, market proxy
    market_returns: pd.Series     # weekly log returns, market proxy


# ============================================================
# LOAD UNIVERSE — the only function that touches the network.
# ============================================================

def load_universe(
    tickers: dict[str, str] = tickers,
    market_proxy_symbol: str = MARKET_PROXY_SYMBOL,
    market_proxy_name: str = MARKET_PROXY_NAME,
    start: str = DEFAULT_START,
) -> UniverseData:
    """Download prices and build weekly log returns for the universe + proxy.

    Universe + proxy are fetched in one call so they share one date grid,
    keyed by SYMBOL and sliced afterwards (avoids a duplicate-column
    collision if the proxy symbol is also a universe ticker).

    Args:
        tickers: Mapping of asset name -> ticker symbol.
        market_proxy_symbol: Ticker for the single-index shrinkage target.
        market_proxy_name: Internal column name for the proxy series.
        start: Download start date (YYYY-MM-DD).

    Returns:
        A `UniverseData` with prices/returns aligned to `list(tickers.keys())`.

    Raises:
        RuntimeError: If the market proxy failed to download.
        ValueError: If any universe ticker is missing from the downloaded
            frame (e.g. dropped for having no overlapping history) — this
            would otherwise silently desync from `ASSET_NAMES`.
    """
    download_symbols = list(dict.fromkeys(list(tickers.values()) + [market_proxy_symbol]))

    raw = yf.download(
        download_symbols,
        start=start,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if market_proxy_symbol not in raw.columns or not raw[market_proxy_symbol].notna().any():
        raise RuntimeError(
            f"Market proxy '{market_proxy_symbol}' failed to download — "
            f"the single-index shrinkage target cannot be built."
        )

    missing = [name for name, sym in tickers.items() if sym not in raw.columns]
    if missing:
        raise ValueError(
            f"These tickers did not download at all: {missing}. "
            f"ASSET_NAMES would desync from the returns frame — fix the "
            f"symbols or drop them from `tickers` instead of letting them "
            f"disappear silently."
        )

    prices = raw[list(tickers.values())].rename(
        columns={sym: name for name, sym in tickers.items()}
    )
    market_price = raw[market_proxy_symbol].rename(market_proxy_name)

    # Align universe + proxy on a common, fully-populated date grid. Doing the
    # ffill/dropna jointly guarantees the proxy is non-NaN on every retained
    # date, so it lines up perfectly with the universe.
    combined = pd.concat([prices, market_price], axis=1)
    combined = combined.dropna(axis=1, how="all").ffill().dropna()

    prices = combined[list(prices.columns)]
    market_price = combined[market_proxy_name]

    weekly = combined.resample("W-FRI").last()
    weekly_returns_all = np.log(weekly / weekly.shift(1)).dropna()

    returns = weekly_returns_all[list(prices.columns)]
    market_returns = weekly_returns_all[market_proxy_name]

    return UniverseData(
        prices=prices,
        returns=returns,
        market_price=market_price,
        market_returns=market_returns,
    )


_default_universe: UniverseData | None = None


def get_default_universe() -> UniverseData:
    """Return `load_universe()` with default args, memoised for this process.

    Convenience for stages/scripts that just want "the" universe without
    re-downloading on every call. Callers who need a different date range or
    ticker set should call `load_universe(...)` directly instead.
    """
    global _default_universe
    if _default_universe is None:
        _default_universe = load_universe()
    return _default_universe


#=============================================================
# LEDOIT-WOLF COVARIANCE FUNCTIONS (pure — no I/O)
#=============================================================

def compute_cov_lw_constant(returns_window: pd.DataFrame) -> pd.DataFrame:
    """LW constant-variance covariance, annualised (Ledoit-Wolf 2004).

    No factor assumptions; always PSD and well-conditioned. Valid when mixing
    in fixed income. Single source of truth for this estimator.
    """
    cov = ledoit_wolf_constant_variance(data=returns_window.values, demean=True) * 52
    return pd.DataFrame(cov, index=returns_window.columns, columns=returns_window.columns)


def compute_cov_lw_single(
    returns_window: pd.DataFrame,
    market_window: pd.Series,
) -> pd.DataFrame:
    """LW single-index covariance, annualised (Ledoit-Wolf 2003).

    Shrinks toward the CAPM single-factor target, so it needs the market-proxy
    return series aligned to the SAME dates as ``returns_window``.
    """
    cov, _alpha = ledoit_wolf_single_index(
        returns=returns_window.values,
        market_return=market_window.values,
        demean=True,
    )
    return pd.DataFrame(cov * 52, index=returns_window.columns, columns=returns_window.columns)


def compute_cov_preferred(
    returns_window: pd.DataFrame,
    market_window: pd.Series | None = None,
    method: str = COV_METHOD,
) -> pd.DataFrame:
    """The preferred covariance for a returns window.

    One place to choose the estimator. ``method`` selects between the two
    Ledoit-Wolf targets:
        "constant" — constant-variance (LW 2004); no market series needed.
        "single"   — single-index (LW 2003); requires ``market_window``,
                     aligned to the SAME dates as ``returns_window``.

    Args:
        returns_window: Weekly log returns, shape (T, N).
        market_window: Market-proxy returns aligned to ``returns_window``;
            required only for ``method="single"``.
        method: "constant" or "single".

    Returns:
        Annualised covariance DataFrame, name-indexed.

    Raises:
        ValueError: On an unknown method, or "single" without a market window.
    """
    if method == "constant":
        return compute_cov_lw_constant(returns_window)
    if method == "single":
        if market_window is None:
            raise ValueError("method='single' requires market_window.")
        return compute_cov_lw_single(returns_window, market_window)
    raise ValueError(f"Unknown method '{method}'. Use 'constant' or 'single'.")


# ============================================================
# Diagnostics — run only when executed directly (`python -m
# portfolio_optimization.correlation_matrix`). Reproduces the original
# script's console output and correlation-matrix comparison plot.
# ============================================================

if __name__ == "__main__":
    universe = load_universe()
    prices, returns = universe.prices, universe.returns
    market_price, market_returns = universe.market_price, universe.market_returns

    print("\nFirst and Last Price of Market Proxy")
    print(market_price.iloc[0], market_price.iloc[-1])

    avg_return_market = market_price.pct_change().mean() * 252
    print(f"\nAverage Annual Return of Market Proxy: {avg_return_market:.4f}")

    print("\nPrices (tail)")
    print(prices.tail())

    print("\nReturns (tail)")
    print(returns.tail())

    T, N = returns.shape
    print(f"\nDimensions: T={T} weeks, N={N} assets")

    # --- Sample covariance (baseline) ---
    cov_sample = returns.cov() * 52
    corr_sample = returns.corr()

    print("\n--- Sample Covariance Matrix (annualised) ---")
    print(cov_sample)
    print("\n--- Sample Correlation Matrix ---")
    print(corr_sample)

    # --- Ledoit-Wolf constant-variance ---
    cov_lw_constant = compute_cov_lw_constant(returns)
    corr_lw_constant = pd.DataFrame(
        cov_to_corr_matrix(cov_lw_constant.values),
        index=returns.columns, columns=returns.columns,
    )
    print("\n--- LW Constant-Variance Covariance (annualised) ---")
    print(cov_lw_constant)
    print("\n--- LW Constant-Variance Correlation ---")
    print(corr_lw_constant)

    # --- Ledoit-Wolf single-index ---
    cov_lw_single = compute_cov_lw_single(returns, market_returns)
    corr_lw_single = pd.DataFrame(
        cov_to_corr_matrix(cov_lw_single.values),
        index=returns.columns, columns=returns.columns,
    )
    print("\n--- LW Single-Index Covariance (annualised) ---")
    print(cov_lw_single)
    print("\n--- LW Single-Index Correlation ---")
    print(corr_lw_single)

    # --- Diagnostics: compare annualised volatilities across methods ---
    vol_comparison = pd.concat(
        [
            pd.Series(np.sqrt(np.diag(cov_sample.values)), index=returns.columns, name="Sample"),
            pd.Series(np.sqrt(np.diag(cov_lw_constant.values)), index=returns.columns, name="LW Constant"),
            pd.Series(np.sqrt(np.diag(cov_lw_single.values)), index=returns.columns, name="LW Single-Index"),
        ],
        axis=1,
    )
    print("\n--- Annualised Volatility Comparison ---")
    print(vol_comparison.to_string(float_format="{:.4f}".format))

    # --- Plot: correlation matrices side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    assets = list(returns.columns)
    for ax, corr_mat, title in zip(
        axes,
        [corr_sample, corr_lw_constant, corr_lw_single],
        ["Sample", "LW Constant-Variance", "LW Single-Index"],
    ):
        im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap="RdYlGn")
        ax.set_xticks(range(N))
        ax.set_yticks(range(N))
        ax.set_xticklabels(assets, rotation=45, ha="right")
        ax.set_yticklabels(assets)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.suptitle("Correlation Matrices: Sample vs Ledoit-Wolf Estimators", y=1.02)
    plt.tight_layout()
    plt.show()

    # We decide on the constant version. cov_preferred is exposed as a
    # function (compute_cov_preferred), not a module-level value — see
    # get_default_universe() / compute_cov_preferred() for downstream use.
    cov_preferred = compute_cov_preferred(returns, market_returns, method=COV_METHOD)
    print("\n--- cov_preferred (Stage 3 output) ---")
    print(cov_preferred)
