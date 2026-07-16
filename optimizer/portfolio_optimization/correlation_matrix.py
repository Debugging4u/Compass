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
# The assets we actually hold and optimise over. Comment any of
# them out freely — including the global fund — without breaking
# the market-proxy logic below.
# ============================================================

tickers = {
    "Global": "IWDA.AS",
    "Asia": "AAXJ",
    "Europe": "IMAE.AS",
    "Norway": "OSEBX.OL",
    "Technology": "XDWT.DE",
    "HBONDS": "0P0001ILS7.IR",
    "GRID": "GRID.DE",
 #   "Industrials": "ESIN.DE",
 #   "Atea": "ATEA.OL",
 #   "Austevoll": "AUSS.OL",
 #   "DNB": "DNB.OL",
 #   "Equinor": "EQNR.OL",
 #   "SalMar": "SALM.OL",
 #   "Veidekke": "VEI.OL",
}

# ============================================================
# MARKET PROXY (for the single-index shrinkage target only)
# Downloaded independently of the universe, so the single-index
# estimator always has a valid market series whether or not the
# global fund is one of the holdings above. This is a benchmark/
# factor input — never an entry in the covariance matrix.
# ============================================================

MARKET_PROXY_SYMBOL = "IWDA.AS"   # MSCI World — broad world-market proxy
MARKET_PROXY_NAME = "Market"      # internal column name, never a holding

# ============================================================
# DOWNLOAD DATA
# Universe + proxy fetched in one call so they share one date
# grid. We key the raw frame by SYMBOL and slice afterwards,
# which avoids any duplicate-column collision if the proxy symbol
# also happens to be one of the universe tickers.
# ============================================================

download_symbols = list(dict.fromkeys(list(tickers.values()) + [MARKET_PROXY_SYMBOL]))

raw = yf.download(
    download_symbols,
    start="2018-01-01",
    auto_adjust=True,
    progress=False,
)["Close"]

# Guard: the market proxy must have actually downloaded.
if MARKET_PROXY_SYMBOL not in raw.columns or not raw[MARKET_PROXY_SYMBOL].notna().any():
    raise RuntimeError(
        f"Market proxy '{MARKET_PROXY_SYMBOL}' failed to download — "
        f"the single-index shrinkage target cannot be built."
    )

# Universe prices, mapped symbol -> friendly name (order-independent).
prices = raw[list(tickers.values())].rename(
    columns={sym: name for name, sym in tickers.items()}
)

# Market proxy as its own series, independent of the universe.
market_price = raw[MARKET_PROXY_SYMBOL].rename(MARKET_PROXY_NAME)

# Align universe + proxy on a common, fully-populated date grid.
# Doing the ffill/dropna jointly guarantees the proxy is non-NaN on
# every retained date, so it lines up perfectly with the universe.
combined = pd.concat([prices, market_price], axis=1)
combined = combined.dropna(axis=1, how="all").ffill().dropna()

prices = combined[list(prices.columns)]
market_price = combined[MARKET_PROXY_NAME]

print("\nFirst and Last Price of Market Proxy")
print(market_price.iloc[0], market_price.iloc[-1])

avg_return_market = market_price.pct_change().mean() * 252
print(f"\nAverage Annual Return of Market Proxy: {avg_return_market:.4f}")

print("\nPrices (tail)")
print(prices.tail())

# ============================================================
# CALCULATE WEEKLY LOG RETURNS
# Universe and proxy returns are computed off the SAME weekly
# frame, so they share an identical index after the shift/dropna.
# ============================================================

weekly = combined.resample("W-FRI").last()
weekly_returns_all = np.log(weekly / weekly.shift(1)).dropna()

# Universe returns -> covariance estimation
returns = weekly_returns_all[list(prices.columns)]

# Market proxy returns -> single-index target only (NOT in the universe)
market_returns = weekly_returns_all[MARKET_PROXY_NAME]

print("\nReturns (tail)")
print(returns.tail())

T, N = returns.shape
print(f"\nDimensions: T={T} weeks, N={N} assets")

# ============================================================
# SAMPLE COVARIANCE (BASELINE)
# Annualised by multiplying by 52 (weeks per year).
# This is the noisy baseline we are trying to improve upon.
# ============================================================

cov_sample = returns.cov() * 52
corr_sample = returns.corr()

print("\n--- Sample Covariance Matrix (annualised) ---")
print(cov_sample)

print("\n--- Sample Correlation Matrix ---")
print(corr_sample)

#=============================================================
#Defining LEDOIT-WOLF COAVARIANCE FUNCIONS

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
    method: str = "constant",
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
# LEDOIT-WOLF (2004) — CONSTANT-VARIANCE TARGET
#
# Theory (Ledoit & Wolf, 2004, J. Multivariate Analysis):
#   Shrink S toward m*I, where m = tr(S)/N is the average
#   sample variance.  The optimal shrinkage intensity alpha is
#   chosen analytically to minimise the expected Frobenius-norm
#   loss between the estimator and the true covariance matrix.
#
#   Sigma* = alpha * m*I  +  (1 - alpha) * S
#
# This is our fallback / sanity-check estimator.  It is always
# positive definite and well-conditioned by construction —
# no post-hoc PSD repair is needed.
#
# Note: ledoit_wolf_constant_variance() expects raw (non-
# annualised) returns and handles demeaning internally.
# We annualise the result afterwards.
# ============================================================

returns_np = returns.values  # shape (T, N)

cov_lw_constant = compute_cov_lw_constant(returns)

corr_lw_constant = pd.DataFrame(
    cov_to_corr_matrix(cov_lw_constant.values),
    index=returns.columns,
    columns=returns.columns,
)

print("\n--- LW Constant-Variance Covariance (annualised) ---")
print(cov_lw_constant)

print("\n--- LW Constant-Variance Correlation ---")
print(corr_lw_constant)

# ============================================================
# LEDOIT-WOLF (2003) — SINGLE-INDEX (MARKET) TARGET
#
# Theory (Ledoit & Wolf, 2003, Journal of Empirical Finance):
#   Shrink S toward the single-factor covariance implied by the
#   CAPM.  The shrinkage target F is constructed as:
#
#       beta_i  = Cov(r_i, r_m) / Var(r_m)
#       F_ij    = beta_i * beta_j * Var(r_m)   for i != j
#       F_ii    = S_ii                           (sample variances kept)
#
#   The optimal shrinkage intensity alpha is estimated from the
#   data via the analytical formula in Ledoit & Wolf (2003).
#   The estimator is:
#
#       Sigma* = alpha * F  +  (1 - alpha) * S
#
# Market proxy: supplied via `market_returns`, which is downloaded
# independently of the investable universe (MARKET_PROXY_SYMBOL
# above). This decouples the market estimate from whether the
# global fund is currently a holding — removing "Global" from the
# universe no longer breaks this block.
# ============================================================

cov_lw_single   = compute_cov_lw_single(returns, market_returns)

corr_lw_single = pd.DataFrame(
    cov_to_corr_matrix(cov_lw_single.values),
    index=returns.columns,
    columns=returns.columns,
)

print("\n--- LW Single-Index Covariance (annualised) ---")
print(cov_lw_single)

print("\n--- LW Single-Index Correlation ---")
print(corr_lw_single)

# ============================================================
# DIAGNOSTICS — compare annualised volatilities across methods
# ============================================================

vols_sample = pd.Series(
    np.sqrt(np.diag(cov_sample.values)),
    index=returns.columns,
    name="Sample",
)
vols_lw_const = pd.Series(
    np.sqrt(np.diag(cov_lw_constant.values)),
    index=returns.columns,
    name="LW Constant",
)
vols_lw_single = pd.Series(
    np.sqrt(np.diag(cov_lw_single.values)),
    index=returns.columns,
    name="LW Single-Index",
)

vol_comparison = pd.concat(
    [vols_sample, vols_lw_const, vols_lw_single], axis=1
)

print("\n--- Annualised Volatility Comparison ---")
print(vol_comparison.to_string(float_format="{:.4f}".format))

# ============================================================
# EXPOSE THE PREFERRED COVARIANCE MATRIX FOR DOWNSTREAM USE
#
# cov_sample          — noisy baseline (kept for reference)
# cov_lw_single       — LW single-index shrinkage
# cov_lw_constant     — PREFERRED: LW constant-variance (no factor
#                       assumptions; valid with fixed income)
# corr_lw_single      — correlation version of the single-index estimate
# ============================================================

# ============================================================
# PLOT — Correlation matrices side by side
# ============================================================

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

# We decide on the constant version. Function is to allow reiteration over different timeframes for backtesting.
COV_METHOD = "constant"
cov_preferred = compute_cov_preferred(returns, market_returns, method=COV_METHOD)
cov_preferred