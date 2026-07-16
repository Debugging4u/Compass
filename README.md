# Compass

Personal portfolio planning app. Combines live portfolio/watchlist tracking
(ported from `Ledger`) with quantitative allocation optimization (ported
from `Finance`'s `portfolio_optimization` pipeline) into one dashboard.

Guiding principle carried over from Finance: robustness over precision. On a
small, collinear ETF universe, forecast-driven optimization mostly manufactures
overfit — the goal here is a disciplined, shrunk, uncertainty-aware process,
not an edge-generating signal layer.

## Project structure

```
compass/
  client/       React frontend (Vite) — portfolio table, watchlist, AI brief
  server/       Express backend — holds API keys, proxies price fetches
  optimizer/
    portfolio_optimization/   Black-Litterman + Ledoit-Wolf + CVXPY pipeline
    codelib/                  vendored helper library (BL, mean-variance,
                               statistics, curves) the pipeline calls directly
    api/                      placeholder for the FastAPI wrapper (not built yet)
    requirements.txt
```

## Status: scaffold only

This repo is currently a straight move of `client/` + `server/` from Ledger
and `portfolio_optimization/` + `codelib/` from Finance, with dead
coursework files trimmed (`optimal_portfolio_old.py`, `codelib/week5`,
`codelib/week6`, `codelib/week_7` — unreferenced by the pipeline). No
integration work has happened yet:

- `optimizer/` is not wired to `server/` — there's no API layer.
- The pipeline modules (`correlation_matrix.py` and everything that imports
  from it) still run as top-to-bottom scripts: importing any of them
  triggers a live `yfinance` download and a blocking `plt.show()`. They need
  to become pure, callable functions before `optimizer/api/` can wrap them.
- `requirements.txt` was re-saved as UTF-8 (the Finance original was UTF-16
  and would fail `pip install -r` as-is). Python version is not yet pinned.
- No shared database — still `localStorage` on the client, same as Ledger.

## Running the dashboard (client + server)

**1. Get an Anthropic API key**
Sign up at anthropic.com, create a key, copy it.

**2. Set the key**
```bash
cp server/.env.example server/.env
# Edit server/.env and paste your key as ANTHROPIC_API_KEY=sk-ant-...
```

**3. Install dependencies**
```bash
cd server && npm install
cd ../client && npm install
```

**4. Start both** (in two separate terminals)
```bash
# Terminal 1 — backend
cd server && npm run dev

# Terminal 2 — frontend
cd client && npm run dev
```

Open http://localhost:5173.

## Running the optimizer (currently standalone)

```bash
cd optimizer
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m portfolio_optimization.optimal_portfolio    # or .backtest, .weight_sensitivity
```

Not yet connected to the dashboard — see Status above.
