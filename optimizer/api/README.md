# api

Placeholder for the FastAPI service that will wrap `portfolio_optimization/`
as JSON endpoints. Not implemented yet — the pipeline modules still run as
top-to-bottom scripts (live yfinance download + blocking `plt.show()` at
import time), so they can't be called from a web request as-is. Refactoring
them into pure, importable functions is the next step before this folder
gets real code.
