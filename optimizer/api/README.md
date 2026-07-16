# api

FastAPI wrapper around `portfolio_optimization/`. Every route calls the same
pure functions the CLI stages use — see `main.py`.

Run (from `optimizer/`, venv active):
```bash
uvicorn api.main:app --reload --port 8000
```

Routes:
- `GET  /health` — liveness check.
- `GET  /universe` — asset names + tickers (no network call).
- `POST /refresh` — force a fresh price download.
- `GET  /portfolio?target_return=0.105` — mu, GMV / Max Sharpe / target-return weights + stats.
- `GET  /frontier?n_points=50` — efficient frontier points (return, vol, weights).
- `GET  /correlation` — Ledoit-Wolf correlation matrix.
- `GET  /backtest` — walk-forward performance table + equity curves.

No auth, no rate limiting — matches the pipeline's current single-user scope.
`server/` proxies to this service rather than the browser calling it directly
(see `server/index.js`, `OPTIMIZER_URL`).
