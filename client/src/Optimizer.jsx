import { useState, useEffect, useCallback } from "react";
import { Compass, RefreshCw, AlertCircle } from "lucide-react";
import { WeightBars, FrontierChart, EquityChart, colorMap } from "./charts";

const DEFAULT_TARGET_PCT = 10.5; // matches optimizer/portfolio_optimization TARGET_RETURN_ANNUAL

const PORTFOLIO_LABEL = { gmv: "Min. variance (GMV)", max_sharpe: "Max Sharpe" };
const PORTFOLIO_MARKER_COLOR = { gmv: "#2a78d6", max_sharpe: "#008300", target: "#eb6834" };

async function getJSON(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

export default function Optimizer() {
  const [portfolioData, setPortfolioData] = useState(null);
  const [frontierData, setFrontierData] = useState(null);
  const [backtestData, setBacktestData] = useState(null);
  const [targetPct, setTargetPct] = useState(DEFAULT_TARGET_PCT);
  const [targetInput, setTargetInput] = useState(String(DEFAULT_TARGET_PCT));
  const [loading, setLoading] = useState(false);
  const [btLoading, setBtLoading] = useState(false);
  const [error, setError] = useState("");

  const loadPortfolio = useCallback(async (pct) => {
    setLoading(true);
    setError("");
    try {
      const [p, f] = await Promise.all([
        getJSON(`/api/optimizer/portfolio?target_return=${pct / 100}`),
        getJSON(`/api/optimizer/frontier?n_points=60`),
      ]);
      setPortfolioData(p);
      setFrontierData(f);
    } catch (e) {
      setError(e.message || "Couldn't reach the optimizer service.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPortfolio(DEFAULT_TARGET_PCT);
  }, [loadPortfolio]);

  const runBacktest = useCallback(async (pct) => {
    setBtLoading(true);
    setError("");
    try {
      setBacktestData(await getJSON(`/api/optimizer/backtest?target_return=${pct / 100}`));
    } catch (e) {
      setError(e.message || "Backtest failed.");
    } finally {
      setBtLoading(false);
    }
  }, []);

  const applyTarget = () => {
    const pct = parseFloat(targetInput);
    if (isNaN(pct)) return;
    setTargetPct(pct);
    loadPortfolio(pct);
    // Keep the backtest in sync with the target control, same as the
    // portfolio/frontier above it — but only if it's already been run once;
    // it stays on-demand otherwise (it's the expensive call).
    if (backtestData) runBacktest(pct);
  };

  const assetNames = portfolioData ? Object.keys(portfolioData.mu_weekly) : [];
  const assetColors = colorMap(assetNames);
  const activePortfolios = portfolioData
    ? ["gmv", "max_sharpe", "target"].map((k) => portfolioData.portfolios[k]).filter(Boolean)
    : [];
  const maxWeight = activePortfolios.length
    ? Math.max(0.01, ...activePortfolios.flatMap((p) => Object.values(p.weights)))
    : 0.01;

  return (
    <section className="block">
      <style>{css}</style>
      <div className="block-head">
        <h2><Compass size={16} /> Optimizer</h2>
        <div className="target-ctrl">
          <label htmlFor="target-return">Target return</label>
          <input
            id="target-return"
            className="in sm"
            inputMode="decimal"
            value={targetInput}
            onChange={(e) => setTargetInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && applyTarget()}
          />
          <span className="pct-sign">%</span>
          <button className="add-btn" onClick={applyTarget} disabled={loading}>
            <RefreshCw size={13} className={loading ? "spin" : ""} />
            {loading ? "Solving…" : "Recalculate"}
          </button>
        </div>
      </div>

      {error && <div className="err"><AlertCircle size={15} /> {error}</div>}

      {!portfolioData && !error && (
        <p className="empty">
          Loading the optimizer — make sure the FastAPI service is running
          (<code>cd optimizer &amp;&amp; uvicorn api.main:app --port 8000</code>).
        </p>
      )}

      {portfolioData && (
        <>
          <div className="opt-cards">
            {["gmv", "max_sharpe", "target"].map((key) => {
              const p = portfolioData.portfolios[key];
              const label =
                key === "target" ? `Target ${targetPct.toFixed(1)}%` : PORTFOLIO_LABEL[key];
              return (
                <div className="opt-card" key={key}>
                  <h3>{label}</h3>
                  {p ? (
                    <>
                      <div className="opt-stats">
                        <span>Return <strong>{(p.return_annual * 100).toFixed(1)}%</strong></span>
                        <span>Vol <strong>{(p.vol_annual * 100).toFixed(1)}%</strong></span>
                        <span>Sharpe <strong>{p.sharpe != null ? p.sharpe.toFixed(2) : "—"}</strong></span>
                      </div>
                      <WeightBars weights={p.weights} colors={assetColors} maxWeight={maxWeight} />
                    </>
                  ) : (
                    <p className="empty">{portfolioData.target_error || "Not attainable at this target."}</p>
                  )}
                </div>
              );
            })}
          </div>

          {frontierData && (
            <div className="opt-frontier">
              <h3>Efficient frontier</h3>
              <FrontierChart
                points={frontierData.points}
                markers={["gmv", "max_sharpe", "target"]
                  .filter((k) => portfolioData.portfolios[k])
                  .map((k) => ({
                    key: k,
                    label: k === "target" ? `Target ${targetPct.toFixed(1)}%` : PORTFOLIO_LABEL[k],
                    color: PORTFOLIO_MARKER_COLOR[k],
                    ...portfolioData.portfolios[k],
                  }))}
              />
            </div>
          )}

          <div className="opt-backtest">
            <div className="opt-backtest-head">
              <h3>Walk-forward backtest</h3>
              <button className="add-btn" onClick={() => runBacktest(targetPct)} disabled={btLoading}>
                {btLoading ? "Running…" : "Run backtest"}
              </button>
            </div>
            {!backtestData && (
              <p className="empty">
                Runs the pipeline's out-of-sample walk-forward test against 1/N and the
                policy weights — a few seconds of solving, so it's on demand rather than
                automatic on every load.
              </p>
            )}
            {backtestData && (
              <>
                <EquityChart
                  series={backtestData.equity_curves}
                  colors={colorMap(Object.keys(backtestData.equity_curves))}
                />
                <div className="opt-perf-table">
                  <div className="opt-perf-row opt-perf-head">
                    <span>Strategy</span>
                    <span className="r">Ann. return</span>
                    <span className="r">Ann. vol</span>
                    <span className="r">Sharpe</span>
                    <span className="r">Max DD</span>
                  </div>
                  {Object.entries(backtestData.performance).map(([name, row]) => (
                    <div className="opt-perf-row" key={name}>
                      <span>{name}</span>
                      <span className="r mono">{(row["Ann.return"] * 100).toFixed(1)}%</span>
                      <span className="r mono">{(row["Ann.vol"] * 100).toFixed(1)}%</span>
                      <span className="r mono">{row["Sharpe"].toFixed(2)}</span>
                      <span className="r mono">{(row["MaxDD"] * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </>
      )}
    </section>
  );
}

// ---------- styles (scoped by the viz-/opt- prefixes; shares tokens with App.css) ----------
const css = `
.target-ctrl{display:flex; align-items:center; gap:8px; font-size:12.5px; color:var(--muted);}
.target-ctrl label{white-space:nowrap;}
.target-ctrl .in.sm{flex:0 0 64px; min-width:0; padding:7px 8px;}
.pct-sign{color:var(--muted);}

code{font-family:'IBM Plex Mono',monospace; font-size:12px; background:var(--panel); padding:1px 5px; border-radius:2px;}

.opt-cards{display:grid; grid-template-columns:repeat(auto-fit, minmax(230px, 1fr)); gap:14px; margin-top:6px;}
.opt-card{background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:14px 16px;}
.opt-card h3{margin:0 0 8px; font-size:13.5px; font-weight:600;}
.opt-stats{display:flex; gap:14px; font-size:12px; color:var(--muted); margin-bottom:12px;}
.opt-stats strong{color:var(--ink); font-family:'IBM Plex Mono',monospace; font-weight:500;}

.viz-bars{display:flex; flex-direction:column; gap:6px;}
.viz-bar-row{display:grid; grid-template-columns:74px 1fr 46px; align-items:center; gap:8px; padding:2px 0;}
.viz-bar-label{font-size:11.5px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.viz-bar-track{height:14px; background:var(--paper); border-radius:2px; overflow:hidden;}
.viz-bar-fill{display:block; height:100%; border-radius:0 3px 3px 0; transition:width .2s ease;}
.viz-bar-value{font-size:11px; text-align:right;}

.opt-frontier, .opt-backtest{margin-top:26px;}
.opt-frontier h3, .opt-backtest-head h3{margin:0 0 10px; font-size:14.5px; font-weight:600; font-family:'Fraunces',serif;}
.opt-backtest-head{display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap;}

.viz-frontier, .viz-equity{background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:14px; position:relative;}
.viz-frontier svg, .viz-equity svg{width:100%; height:auto; display:block; overflow:visible;}
.viz-grid{stroke:var(--line); stroke-width:1;}
.viz-baseline{stroke:#c3c2b7; stroke-width:1;}
.viz-axis{font-size:9.5px; fill:var(--muted); font-family:'IBM Plex Mono',monospace;}
.viz-axis-title{font-size:10.5px; color:var(--muted); text-align:center; margin-top:6px;}
.viz-frontier-line{stroke:var(--accent); stroke-width:2; stroke-linecap:round; stroke-linejoin:round;}
.viz-line{stroke-width:2; stroke-linecap:round; stroke-linejoin:round;}
.viz-dot{stroke:var(--panel); stroke-width:2;}
.viz-point-label{font-size:11px; fill:var(--ink); font-weight:500;}
.viz-line-label{font-size:10.5px; fill:var(--muted);}
.viz-leader{stroke:var(--muted); stroke-width:1; opacity:.5;}
.viz-crosshair{stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3;}

.viz-legend{display:flex; flex-wrap:wrap; gap:14px; margin-top:10px;}
.viz-legend-item{display:inline-flex; align-items:center; gap:6px; font-size:11.5px; color:var(--muted);}
.viz-legend-key{display:inline-block; width:14px; height:2px; border-radius:1px;}

.viz-tip{position:fixed; z-index:20; background:var(--ink); color:#fff; font-size:11.5px; line-height:1.5;
  padding:7px 10px; border-radius:3px; pointer-events:none; max-width:220px; box-shadow:0 4px 14px rgba(0,0,0,.18);}
.viz-tip strong{font-family:'IBM Plex Mono',monospace; font-weight:600;}
.viz-tip-date{font-family:'IBM Plex Mono',monospace; opacity:.75; margin-bottom:3px;}
.viz-tip-row{display:flex; align-items:center; gap:6px;}
.viz-tip-key{display:inline-block; width:10px; height:2px; border-radius:1px;}

.opt-perf-table{display:flex; flex-direction:column; margin-top:14px;}
.opt-perf-row{display:grid; grid-template-columns:1.6fr .9fr .9fr .8fr .9fr; gap:10px; padding:9px 0;
  border-bottom:1px solid var(--line); font-size:13px;}
.opt-perf-head{font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
  border-bottom:1px solid var(--ink); padding-bottom:7px;}

@media(max-width:680px){
  .opt-perf-row{grid-template-columns:1.3fr 1fr 1fr;}
  .opt-perf-row span:nth-child(4),.opt-perf-row span:nth-child(5){display:none;}
}
`;
