import { useState, useEffect, useCallback } from "react";
import {
  Compass, RefreshCw, AlertCircle, ChevronDown, ChevronRight, RotateCcw, Plus, X,
} from "lucide-react";
import { WeightBars, FrontierChart, EquityChart, colorMap } from "./charts";

const DEFAULT_TARGET_PCT = 10.5; // matches optimizer/portfolio_optimization TARGET_RETURN_ANNUAL
const DEFAULT_RF_PCT = 4.0;      // matches optimizer/portfolio_optimization RF_ANNUAL

const PORTFOLIO_LABEL = { gmv: "Min. variance (GMV)", max_sharpe: "Max Sharpe" };
const PORTFOLIO_MARKER_COLOR = { gmv: "#2a78d6", max_sharpe: "#008300", target: "#eb6834" };

const uid = () => Math.random().toString(36).slice(2, 9);

async function getJSON(path) {
  const res = await fetch(path);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

async function postJSON(path, payload) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

// Strings -> numbers, or null if anything doesn't parse (caller shows an error
// rather than silently sending NaN, which JSON.stringify turns into `null`).
function parseWeights(weightsInput) {
  const out = {};
  for (const [name, v] of Object.entries(weightsInput)) {
    const n = parseFloat(v);
    if (isNaN(n)) return null;
    out[name] = n;
  }
  return out;
}

// API view dict -> an editable row. target comes back as a fraction (0.052);
// the row keeps it as a percent string (5.2) to match the input's own unit.
function viewFromApi(v) {
  return {
    id: uid(),
    type: v.type,
    asset: v.type === "absolute" ? v.asset : "",
    longAsset: v.type === "relative" ? v.long_asset : "",
    shortAsset: v.type === "relative" ? v.short_asset : "",
    target: String((v.target * 100).toFixed(2)),
    confidence: v.confidence != null ? String(v.confidence) : "",
  };
}

// Editable rows -> API payload, or null if anything doesn't parse (same
// fail-closed contract as parseWeights).
function parseViews(viewsInput) {
  const out = [];
  for (const v of viewsInput) {
    const target = parseFloat(v.target);
    if (isNaN(target)) return null;
    let confidence = null;
    if (v.confidence.trim() !== "") {
      confidence = parseFloat(v.confidence);
      if (isNaN(confidence)) return null;
    }
    if (v.type === "absolute") {
      if (!v.asset) return null;
      out.push({ type: "absolute", asset: v.asset, target: target / 100, confidence });
    } else {
      if (!v.longAsset || !v.shortAsset) return null;
      out.push({
        type: "relative", long_asset: v.longAsset, short_asset: v.shortAsset,
        target: target / 100, confidence,
      });
    }
  }
  return out;
}

export default function Optimizer() {
  const [assetNames, setAssetNames] = useState([]);
  const [defaultWeights, setDefaultWeights] = useState(null);
  const [weightsInput, setWeightsInput] = useState({});
  const [showWeights, setShowWeights] = useState(false);

  const [defaultViews, setDefaultViews] = useState(null);
  const [defaultConfidence, setDefaultConfidence] = useState(1.0);
  const [viewsInput, setViewsInput] = useState([]);
  const [showViews, setShowViews] = useState(false);

  const [portfolioData, setPortfolioData] = useState(null);
  const [frontierData, setFrontierData] = useState(null);
  const [backtestData, setBacktestData] = useState(null);
  const [targetPct, setTargetPct] = useState(DEFAULT_TARGET_PCT);
  const [targetInput, setTargetInput] = useState(String(DEFAULT_TARGET_PCT));
  const [rfPct, setRfPct] = useState(DEFAULT_RF_PCT);
  const [rfInput, setRfInput] = useState(String(DEFAULT_RF_PCT));
  const [neutralPrior, setNeutralPrior] = useState(false);
  const [loading, setLoading] = useState(false);
  const [btLoading, setBtLoading] = useState(false);
  const [error, setError] = useState("");

  const loadPortfolio = useCallback(async (pct, rf, weights, views) => {
    setLoading(true);
    setError("");
    try {
      const [p, f] = await Promise.all([
        postJSON("/api/optimizer/portfolio", {
          target_return: pct / 100, rf_annual: rf / 100, strategic_weights: weights, views,
        }),
        postJSON("/api/optimizer/frontier", {
          n_points: 60, rf_annual: rf / 100, strategic_weights: weights, views,
        }),
      ]);
      setPortfolioData(p);
      setFrontierData(f);
    } catch (e) {
      setError(e.message || "Couldn't reach the optimizer service.");
    } finally {
      setLoading(false);
    }
  }, []);

  // On mount: fetch the universe (asset order + the pipeline's default policy
  // weights/views, to pre-populate the editors), then load portfolio/frontier.
  useEffect(() => {
    (async () => {
      try {
        const u = await getJSON("/api/optimizer/universe");
        setAssetNames(u.asset_names);
        setDefaultWeights(u.strategic_weights);
        setDefaultViews(u.views);
        setDefaultConfidence(u.default_confidence);

        const inputs = {};
        for (const name of u.asset_names) inputs[name] = String(u.strategic_weights[name]);
        setWeightsInput(inputs);
        setViewsInput(u.views.map(viewFromApi));

        // u.views is already ViewRequest-shaped (that's what _views_to_dicts
        // produces server-side), so it can go straight back into the request body.
        await loadPortfolio(DEFAULT_TARGET_PCT, DEFAULT_RF_PCT, u.strategic_weights, u.views);
      } catch (e) {
        setError(e.message || "Couldn't reach the optimizer service.");
      }
    })();
  }, [loadPortfolio]);

  const runBacktest = useCallback(async (pct, rf, neutral, weights, views) => {
    setBtLoading(true);
    setError("");
    try {
      setBacktestData(
        await postJSON("/api/optimizer/backtest", {
          target_return: pct / 100, rf_annual: rf / 100, neutral_prior: neutral,
          strategic_weights: weights, views,
        }),
      );
    } catch (e) {
      setError(e.message || "Backtest failed.");
    } finally {
      setBtLoading(false);
    }
  }, []);

  const applyControls = () => {
    const pct = parseFloat(targetInput);
    const rf = parseFloat(rfInput);
    const weights = parseWeights(weightsInput);
    const views = parseViews(viewsInput);
    if (isNaN(pct) || isNaN(rf)) return;
    if (!weights) {
      setError("Strategic weights must all be numbers.");
      return;
    }
    if (!views) {
      setError("Views have a field that isn't filled in — check target/confidence and asset selections.");
      return;
    }
    setTargetPct(pct);
    setRfPct(rf);
    loadPortfolio(pct, rf, weights, views);
    // Keep the backtest in sync with these controls, same as the
    // portfolio/frontier above it — but only if it's already been run once;
    // it stays on-demand otherwise (it's the expensive call).
    if (backtestData) runBacktest(pct, rf, neutralPrior, weights, views);
  };

  const toggleNeutralPrior = (checked) => {
    setNeutralPrior(checked);
    if (backtestData) {
      const weights = parseWeights(weightsInput);
      const views = parseViews(viewsInput);
      if (weights && views) runBacktest(targetPct, rfPct, checked, weights, views);
    }
  };

  const resetWeights = () => {
    if (!defaultWeights) return;
    const inputs = {};
    for (const name of assetNames) inputs[name] = String(defaultWeights[name]);
    setWeightsInput(inputs);
    const views = parseViews(viewsInput);
    if (views) {
      loadPortfolio(targetPct, rfPct, defaultWeights, views);
      if (backtestData) runBacktest(targetPct, rfPct, neutralPrior, defaultWeights, views);
    }
  };

  const resetViews = () => {
    if (!defaultViews) return;
    setViewsInput(defaultViews.map(viewFromApi));
    const weights = parseWeights(weightsInput);
    if (weights) {
      loadPortfolio(targetPct, rfPct, weights, defaultViews);
      if (backtestData) runBacktest(targetPct, rfPct, neutralPrior, weights, defaultViews);
    }
  };

  const addView = () => {
    setViewsInput((vs) => [...vs, {
      id: uid(), type: "absolute",
      asset: assetNames[0] || "", longAsset: assetNames[0] || "", shortAsset: assetNames[1] || assetNames[0] || "",
      target: "", confidence: "",
    }]);
  };

  const removeView = (id) => setViewsInput((vs) => vs.filter((v) => v.id !== id));
  const updateView = (id, patch) => setViewsInput((vs) => vs.map((v) => (v.id === id ? { ...v, ...patch } : v)));

  const assetColors = colorMap(assetNames);
  const activePortfolios = portfolioData
    ? ["gmv", "max_sharpe", "target"].map((k) => portfolioData.portfolios[k]).filter(Boolean)
    : [];
  const maxWeight = activePortfolios.length
    ? Math.max(0.01, ...activePortfolios.flatMap((p) => Object.values(p.weights)))
    : 0.01;

  const weightsSum = Object.values(weightsInput).reduce((s, v) => s + (parseFloat(v) || 0), 0);

  return (
    <section className="block">
      <style>{css}</style>
      <div className="block-head">
        <h2><Compass size={16} /> Optimizer</h2>
        <div className="controls-row">
          <div className="ctrl-field">
            <label htmlFor="target-return">Target return</label>
            <input
              id="target-return"
              className="in sm"
              inputMode="decimal"
              value={targetInput}
              onChange={(e) => setTargetInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && applyControls()}
            />
            <span className="pct-sign">%</span>
          </div>
          <div className="ctrl-field">
            <label htmlFor="rf-rate">Risk-free rate</label>
            <input
              id="rf-rate"
              className="in sm"
              inputMode="decimal"
              value={rfInput}
              onChange={(e) => setRfInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && applyControls()}
            />
            <span className="pct-sign">%</span>
          </div>
          <button className="add-btn" onClick={applyControls} disabled={loading}>
            <RefreshCw size={13} className={loading ? "spin" : ""} />
            {loading ? "Solving…" : "Recalculate"}
          </button>
        </div>
      </div>

      {assetNames.length > 0 && (
        <div className="weights-block">
          <button className="weights-toggle" onClick={() => setShowWeights((v) => !v)}>
            {showWeights ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Policy weights (Black-Litterman equilibrium prior)
          </button>
          {showWeights && (
            <div className="weights-editor">
              <p className="weights-help">
                Relative sizes for the BL equilibrium prior — the neutral, opinion-free
                reference every view tilts away from. Don't need to sum to 100, only
                relative proportions matter. This is the single most consequential input
                to the model, more so than any individual view.
              </p>
              <div className="weights-rows">
                {assetNames.map((name) => {
                  const v = parseFloat(weightsInput[name]);
                  const pct = weightsSum > 0 && !isNaN(v) ? (v / weightsSum) * 100 : 0;
                  return (
                    <div className="weights-row" key={name}>
                      <span className="weights-swatch" style={{ background: assetColors[name] }} />
                      <span className="weights-label">{name}</span>
                      <input
                        className="in sm"
                        inputMode="decimal"
                        value={weightsInput[name] ?? ""}
                        onChange={(e) =>
                          setWeightsInput((w) => ({ ...w, [name]: e.target.value }))
                        }
                        onKeyDown={(e) => e.key === "Enter" && applyControls()}
                      />
                      <span className="weights-pct mono">{pct.toFixed(1)}%</span>
                    </div>
                  );
                })}
              </div>
              <div className="weights-actions">
                <button className="link-btn" onClick={resetWeights}>
                  <RotateCcw size={12} /> Reset to defaults
                </button>
                <span className="weights-hint">Edit values above, then Recalculate.</span>
              </div>
            </div>
          )}
        </div>
      )}

      {assetNames.length > 0 && (
        <div className="weights-block">
          <button className="weights-toggle" onClick={() => setShowViews((v) => !v)}>
            {showViews ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            Views (tilts on the equilibrium prior)
          </button>
          {showViews && (
            <div className="weights-editor">
              <p className="weights-help">
                Explicit opinions layered on top of the policy weights above — e.g. "Tech
                returns 12% p.a." or "Asia beats Europe by 3% p.a." Leave confidence blank
                for the default ({defaultConfidence.toFixed(1)}); higher tightens conviction,
                lower loosens it. No views = pure equilibrium.
              </p>
              <div className="views-rows">
                {viewsInput.map((v) => (
                  <div className="view-row" key={v.id}>
                    <select
                      className="in sm view-type"
                      value={v.type}
                      onChange={(e) => updateView(v.id, { type: e.target.value })}
                    >
                      <option value="absolute">Absolute</option>
                      <option value="relative">Relative</option>
                    </select>

                    {v.type === "absolute" ? (
                      <>
                        <select
                          className="in sm view-asset"
                          value={v.asset}
                          onChange={(e) => updateView(v.id, { asset: e.target.value })}
                        >
                          {assetNames.map((n) => <option key={n} value={n}>{n}</option>)}
                        </select>
                        <span className="view-word">returns</span>
                      </>
                    ) : (
                      <>
                        <select
                          className="in sm view-asset"
                          value={v.longAsset}
                          onChange={(e) => updateView(v.id, { longAsset: e.target.value })}
                        >
                          {assetNames.map((n) => <option key={n} value={n}>{n}</option>)}
                        </select>
                        <span className="view-word">beats</span>
                        <select
                          className="in sm view-asset"
                          value={v.shortAsset}
                          onChange={(e) => updateView(v.id, { shortAsset: e.target.value })}
                        >
                          {assetNames.map((n) => <option key={n} value={n}>{n}</option>)}
                        </select>
                        <span className="view-word">by</span>
                      </>
                    )}

                    <input
                      className="in sm view-num"
                      inputMode="decimal"
                      placeholder="0.0"
                      value={v.target}
                      onChange={(e) => updateView(v.id, { target: e.target.value })}
                      onKeyDown={(e) => e.key === "Enter" && applyControls()}
                    />
                    <span className="view-word">% p.a., confidence</span>
                    <input
                      className="in sm view-num"
                      inputMode="decimal"
                      placeholder={defaultConfidence.toFixed(1)}
                      value={v.confidence}
                      onChange={(e) => updateView(v.id, { confidence: e.target.value })}
                      onKeyDown={(e) => e.key === "Enter" && applyControls()}
                    />

                    <button className="view-del" onClick={() => removeView(v.id)}>
                      <X size={13} />
                    </button>
                  </div>
                ))}
                {viewsInput.length === 0 && (
                  <p className="empty">No views — pure equilibrium. Add one below.</p>
                )}
              </div>
              <div className="weights-actions">
                <button className="link-btn" onClick={addView}>
                  <Plus size={12} /> Add view
                </button>
                <button className="link-btn" onClick={resetViews}>
                  <RotateCcw size={12} /> Reset to defaults
                </button>
                <span className="weights-hint">Edit above, then Recalculate.</span>
              </div>
            </div>
          )}
        </div>
      )}

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
              <div className="bt-controls">
                <label className="bt-toggle">
                  <input
                    type="checkbox"
                    checked={neutralPrior}
                    onChange={(e) => toggleNeutralPrior(e.target.checked)}
                  />
                  Neutral prior (ignore my views)
                </label>
                <button
                  className="add-btn"
                  onClick={() => {
                    const weights = parseWeights(weightsInput);
                    const views = parseViews(viewsInput);
                    if (weights && views) runBacktest(targetPct, rfPct, neutralPrior, weights, views);
                  }}
                  disabled={btLoading}
                >
                  {btLoading ? "Running…" : "Run backtest"}
                </button>
              </div>
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
                <p className="bt-mode-note">
                  {backtestData.neutral_prior ? (
                    <>
                      <strong>Neutral prior.</strong> Every rebalance re-estimated mu from an
                      equal-weight prior with no views — this tests the shrinkage + optimisation
                      machinery in isolation, uncontaminated by hindsight.
                    </>
                  ) : (
                    <>
                      <strong>Live prior.</strong> Every rebalance used your current policy
                      weights + views, even on historical windows — so Max Sharpe and Target
                      here are partly a test of beliefs formed with full-sample hindsight, not
                      just the machinery. GMV, 1/N and Strategic don't depend on mu, so they're
                      identical either way — toggle "Neutral prior" to isolate the rest.
                    </>
                  )}
                </p>
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
.controls-row{display:flex; align-items:center; gap:16px; flex-wrap:wrap; font-size:12.5px; color:var(--muted);}
.ctrl-field{display:flex; align-items:center; gap:8px;}
.ctrl-field label{white-space:nowrap;}
.ctrl-field .in.sm{flex:0 0 60px; min-width:0; padding:7px 8px;}
.pct-sign{color:var(--muted);}

code{font-family:'IBM Plex Mono',monospace; font-size:12px; background:var(--panel); padding:1px 5px; border-radius:2px;}

.weights-block{margin-top:14px;}
.weights-toggle{display:inline-flex; align-items:center; gap:6px; background:none; border:none; cursor:pointer;
  color:var(--accent); font-size:12.5px; font-weight:500; padding:4px 0;}
.weights-editor{margin-top:8px; padding:14px 16px; background:var(--panel); border:1px solid var(--line); border-radius:3px;}
.weights-help{font-size:12px; color:var(--muted); line-height:1.55; margin:0 0 12px;}
.weights-rows{display:flex; flex-direction:column; gap:7px;}
.weights-row{display:grid; grid-template-columns:10px 88px 90px 50px; align-items:center; gap:9px;}
.weights-swatch{width:10px; height:10px; border-radius:2px;}
.weights-label{font-size:12px; color:var(--ink);}
.weights-row .in.sm{padding:6px 8px; font-size:12.5px;}
.weights-pct{font-size:11px; color:var(--muted);}
.weights-actions{display:flex; align-items:center; gap:12px; margin-top:12px; padding-top:12px; border-top:1px solid var(--line); flex-wrap:wrap;}
.link-btn{display:inline-flex; align-items:center; gap:5px; background:none; border:none; color:var(--accent);
  font-size:12px; font-weight:500; cursor:pointer; padding:0;}
.link-btn:hover{opacity:.8;}
.weights-hint{font-size:11.5px; color:var(--muted); font-style:italic;}

.views-rows{display:flex; flex-direction:column; gap:9px;}
.view-row{display:flex; align-items:center; gap:7px; flex-wrap:wrap; padding:8px; background:var(--paper); border-radius:2px;}
.view-row .in.sm{padding:6px 8px; font-size:12px;}
.view-type{flex:0 0 84px;}
.view-asset{flex:0 0 108px;}
.view-num{flex:0 0 62px; text-align:right;}
.view-word{font-size:11.5px; color:var(--muted); white-space:nowrap;}
.view-del{background:none; border:none; color:var(--muted); cursor:pointer; padding:3px; display:flex; margin-left:auto;}
.view-del:hover{color:var(--down);}

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
.bt-controls{display:flex; align-items:center; gap:14px; flex-wrap:wrap;}
.bt-toggle{display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); cursor:pointer; white-space:nowrap;}
.bt-toggle input{accent-color:var(--accent); cursor:pointer;}
.bt-mode-note{font-size:12px; color:var(--muted); line-height:1.55; margin:0 0 12px; padding:9px 12px;
  background:var(--panel); border-left:2px solid var(--accent); border-radius:0 2px 2px 0;}
.bt-mode-note strong{color:var(--ink);}

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
