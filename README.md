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
    api/                      FastAPI wrapper exposing the pipeline as JSON
    requirements.txt
```

## Status

- The pipeline modules no longer run as top-to-bottom scripts. Importing any
  of them does nothing (no network call, no printing, no plotting) — see
  `correlation_matrix.py`'s `load_universe()` / `get_default_universe()`.
  Console output and plots still happen when a stage is run directly
  (`python -m portfolio_optimization.<stage>`), guarded behind
  `if __name__ == "__main__":`.
- `optimizer/api/` wraps the pipeline as a FastAPI service (`/universe`,
  `/portfolio`, `/frontier`, `/correlation`, `/backtest`, `/refresh`).
  `server/` proxies to it (`/api/optimizer/*`, see `OPTIMIZER_URL`) — the
  browser never talks to the Python service directly.
- **Not done yet:** no UI consumes these endpoints — `client/` still only
  shows the Ledger portfolio table/watchlist/brief. That's the next piece.
- `requirements.txt` was re-saved as UTF-8 (the Finance original was UTF-16
  and would fail `pip install -r` as-is); `cvxpy` was bumped from 1.3.2 to
  1.9.2 (the pinned version didn't install against the pinned `scipy`).
  Python version itself is still not pinned.
- No shared database — still `localStorage` on the client, same as Ledger.
- Dead coursework files trimmed during the move from Finance
  (`optimal_portfolio_old.py`, `codelib/week5`, `codelib/week6`,
  `codelib/week_7` — unreferenced by the pipeline).

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

## Running the optimizer

**As a CLI stage** (prints diagnostics, opens plots):
```bash
cd optimizer
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m portfolio_optimization.optimal_portfolio    # or .backtest, .weight_sensitivity
```

**As an API** (what `server/` proxies to):
```bash
cd optimizer && source .venv/bin/activate
uvicorn api.main:app --reload --port 8000
```

With that running, `server/` (started as above) exposes it at
`/api/optimizer/*`, e.g. `http://localhost:3001/api/optimizer/portfolio`.
See `optimizer/api/README.md` for the full route list.
