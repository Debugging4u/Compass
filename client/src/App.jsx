import { useState, useEffect, useCallback } from "react";
import { RefreshCw, Plus, X, TrendingUp, TrendingDown, Eye, Briefcase, AlertCircle } from "lucide-react";

// ---------- helpers ----------
const uid = () => Math.random().toString(36).slice(2, 9);

const fmtMoney = (n, cur = "USD") => {
  if (n == null || isNaN(n)) return "—";
  const sym = { USD: "$", EUR: "€", GBP: "£" }[cur] || "";
  const v = Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${n < 0 ? "−" : ""}${sym}${v}${sym ? "" : " " + cur}`;
};
const fmtPct = (n) => (n == null || isNaN(n) ? "—" : `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(2)}%`);
const fmtNum = (n) => (n == null || isNaN(n) ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 4 }));

// ---------- storage helpers ----------
// localStorage is built into every browser — no setup needed.
// It stores plain strings, so we JSON-encode values on the way in and decode on the way out.
// This replaces the claude.ai-only window.storage API from the original file.
const HOLD_KEY = "desk-holdings-v2";
const CACHE_KEY = "desk-cache-v2";

const storage = {
  get: (key) => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? { value: raw } : null;
    } catch {
      return null;
    }
  },
  set: (key, value) => {
    try {
      localStorage.setItem(key, value);
    } catch {}
  },
};

// ---------- AI / data fetch ----------
// Calls our own backend proxy at /api/market instead of Anthropic directly.
// The backend holds the API key — it should never live in frontend code.
async function fetchMarket(tickers) {
  if (!tickers.length) return { quotes: {}, brief: "" };

  const res = await fetch("/api/market", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tickers }),
  });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

// ---------- seed data ----------
const DEFAULT_PORTFOLIO = [
  { id: "s1", ticker: "ATEA.OL", shares: 24, cost: 122.56 },
  { id: "s2", ticker: "AUSS.OL", shares: 36, cost: 90.85 },
  { id: "s3", ticker: "DNB.OL", shares: 15, cost: 218.33 },
  { id: "s4", ticker: "EQNR.OL", shares: 14, cost: 268.84 },
  { id: "s5", ticker: "SALM.OL", shares: 9, cost: 487.39 },
  { id: "s6", ticker: "MING.OL", shares: 18, cost: 177.77 },
  { id: "s7", ticker: "VEI.OL", shares: 27, cost: 124.39 },
  { id: "s8", ticker: "PPI.OL", shares: 213, cost: 19.61 },
  { id: "s9", ticker: "ESIN", shares: 25, cost: 8.06 },
];

const DEFAULT_WATCHLIST = [
  { id: "w1", ticker: "URTH", label: "Global ≈ DNB Global Indeks" },
  { id: "w2", ticker: "AAXJ", label: "Asia ≈ KLP AksjeAsia" },
  { id: "w3", ticker: "VGK", label: "Europe ≈ Nordnet Europa" },
  { id: "w4", ticker: "VGT", label: "Tech ≈ Nordnet Teknologi" },
  { id: "w5", ticker: "NORW", label: "Norway ≈ Nordnet Norge" },
  { id: "w6", ticker: "GRID", label: "Your First Trust smart-grid ETF" },
];

export default function App() {
  const [portfolio, setPortfolio] = useState([]);
  const [watchlist, setWatchlist] = useState([]);
  const [quotes, setQuotes] = useState({});
  const [brief, setBrief] = useState("");
  const [updatedAt, setUpdatedAt] = useState(null);
  const [loaded, setLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // load on mount
  useEffect(() => {
    const h = storage.get(HOLD_KEY);
    if (h?.value) {
      const v = JSON.parse(h.value);
      setPortfolio(v.portfolio || []);
      setWatchlist(v.watchlist || []);
    } else {
      setPortfolio(DEFAULT_PORTFOLIO);
      setWatchlist(DEFAULT_WATCHLIST);
    }

    const c = storage.get(CACHE_KEY);
    if (c?.value) {
      const v = JSON.parse(c.value);
      setQuotes(v.quotes || {});
      setBrief(v.brief || "");
      setUpdatedAt(v.updatedAt || null);
    }

    setLoaded(true);
  }, []);

  // persist holdings whenever they change
  useEffect(() => {
    if (!loaded) return;
    storage.set(HOLD_KEY, JSON.stringify({ portfolio, watchlist }));
  }, [portfolio, watchlist, loaded]);

  const refresh = useCallback(async () => {
    const tickers = [...new Set([...portfolio.map((p) => p.ticker), ...watchlist.map((w) => w.ticker)])];
    if (!tickers.length) {
      setError("Add a holding or a ticker to watch first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const { quotes: q, brief: b } = await fetchMarket(tickers);
      const ts = Date.now();
      setQuotes(q);
      setBrief(b);
      setUpdatedAt(ts);
      storage.set(CACHE_KEY, JSON.stringify({ quotes: q, brief: b, updatedAt: ts }));
    } catch (e) {
      setError(e.message || "Couldn't pull the latest. Try again in a moment.");
    } finally {
      setLoading(false);
    }
  }, [portfolio, watchlist]);

  // ---------- derived portfolio totals ----------
  let totalValue = 0, totalCost = 0, totalDay = 0, haveValue = false;
  portfolio.forEach((p) => {
    const q = quotes[p.ticker];
    if (q && q.price != null) {
      haveValue = true;
      totalValue += q.price * p.shares;
      if (p.cost != null) totalCost += p.cost * p.shares;
      if (q.change != null) totalDay += q.change * p.shares;
    }
  });
  const totalGain = haveValue && totalCost ? totalValue - totalCost : null;
  const totalGainPct = totalGain != null && totalCost ? (totalGain / totalCost) * 100 : null;

  return (
    <div className="desk">
      <style>{css}</style>

      <header className="hd">
        <div className="brand">
          <span className="mark">LEDGER</span>
          <span className="tag">watchlist &amp; portfolio brief</span>
        </div>
        <div className="hd-right">
          <span className="stamp">
            {updatedAt
              ? `updated ${new Date(updatedAt).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}`
              : "not yet pulled"}
          </span>
          <button className="refresh" onClick={refresh} disabled={loading}>
            <RefreshCw size={14} className={loading ? "spin" : ""} />
            {loading ? "Pulling…" : "Refresh"}
          </button>
        </div>
      </header>

      {error && (
        <div className="err"><AlertCircle size={15} /> {error}</div>
      )}

      <section className="brief">
        <div className="brief-eyebrow">THE BRIEF</div>
        <p className="brief-body">
          {loading && !brief
            ? "Reading the latest across your names…"
            : brief
            ? brief
            : "Your brief lands here once you've added a few names and hit Refresh — a quick read on what's moving and why."}
        </p>
      </section>

      {/* portfolio */}
      <section className="block">
        <div className="block-head">
          <h2><Briefcase size={16} /> Portfolio</h2>
          {haveValue && (
            <div className="totals">
              <div className="t-item">
                <span className="t-label">Value</span>
                <span className="t-val">{fmtMoney(totalValue)}</span>
              </div>
              <div className="t-item">
                <span className="t-label">Day</span>
                <span className={`t-val ${totalDay >= 0 ? "up" : "down"}`}>
                  {totalDay >= 0 ? "+" : "−"}{fmtMoney(Math.abs(totalDay)).replace("−", "")}
                </span>
              </div>
              {totalGain != null && (
                <div className="t-item">
                  <span className="t-label">Total P/L</span>
                  <span className={`t-val ${totalGain >= 0 ? "up" : "down"}`}>
                    {fmtMoney(totalGain)} <small>{fmtPct(totalGainPct)}</small>
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {portfolio.length === 0 ? (
          <p className="empty">No holdings yet. Add one below to start tracking your position and P/L.</p>
        ) : (
          <div className="table">
            <div className="row head">
              <span>Ticker</span><span className="r">Shares</span><span className="r">Avg cost</span>
              <span className="r">Price</span><span className="r">Day</span><span className="r">Value</span>
              <span className="r">P/L</span><span />
            </div>
            {portfolio.map((p) => {
              const q = quotes[p.ticker] || {};
              const value = q.price != null ? q.price * p.shares : null;
              const pl = value != null && p.cost != null ? value - p.cost * p.shares : null;
              const plPct = pl != null && p.cost ? (pl / (p.cost * p.shares)) * 100 : null;
              return (
                <div className="row" key={p.id}>
                  <span className="tk">{p.ticker}{q.name && <em>{q.name}</em>}</span>
                  <span className="r mono">{fmtNum(p.shares)}</span>
                  <span className="r mono">{p.cost != null ? fmtMoney(p.cost, q.currency) : "—"}</span>
                  <span className="r mono">{fmtMoney(q.price, q.currency)}</span>
                  <span className={`r mono ${q.changePct >= 0 ? "up" : "down"}`}>{fmtPct(q.changePct)}</span>
                  <span className="r mono">{fmtMoney(value, q.currency)}</span>
                  <span className={`r mono ${pl >= 0 ? "up" : "down"}`}>
                    {pl != null ? <>{fmtMoney(pl, q.currency)} <small>{fmtPct(plPct)}</small></> : "—"}
                  </span>
                  <span className="r">
                    <button className="del" onClick={() => setPortfolio((s) => s.filter((x) => x.id !== p.id))}>
                      <X size={14} />
                    </button>
                  </span>
                </div>
              );
            })}
          </div>
        )}
        <AddHolding onAdd={(h) => setPortfolio((s) => [...s, h])} />
      </section>

      {/* watchlist */}
      <section className="block">
        <div className="block-head"><h2><Eye size={16} /> Watchlist</h2></div>
        {watchlist.length === 0 ? (
          <p className="empty">Nothing on the watch yet. Drop in a ticker you're keeping an eye on.</p>
        ) : (
          <div className="chips">
            {watchlist.map((w) => {
              const q = quotes[w.ticker] || {};
              const up = q.changePct >= 0;
              return (
                <div className="chip" key={w.id}>
                  <button className="chip-del" onClick={() => setWatchlist((s) => s.filter((x) => x.id !== w.id))}>
                    <X size={12} />
                  </button>
                  <div className="chip-tk">{w.ticker}</div>
                  {w.label && <div className="chip-label">{w.label}</div>}
                  <div className="chip-price mono">{fmtMoney(q.price, q.currency)}</div>
                  <div className={`chip-chg mono ${up ? "up" : "down"}`}>
                    {q.changePct != null && (up ? <TrendingUp size={12} /> : <TrendingDown size={12} />)}
                    {fmtPct(q.changePct)}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        <AddTicker onAdd={(t) => setWatchlist((s) => [...s, { id: uid(), ticker: t }])} />
      </section>

      <footer className="ft">
        Prices are indicative and may be delayed — pulled live from the web, not an exchange feed. The Portfolio total
        adds up positions in their own currency (NOK, SEK, EUR) without converting, so it's approximate — fixing that
        with live FX rates is a great first project. Watchlist tickers are ETF proxies that track the same markets as
        your funds, not your actual fund holdings. This is information, not financial advice; do your own diligence
        before trading.
      </footer>
    </div>
  );
}

// ---------- sub-forms ----------
function AddHolding({ onAdd }) {
  const [t, setT] = useState("");
  const [sh, setSh] = useState("");
  const [c, setC] = useState("");
  const submit = () => {
    const tk = t.trim().toUpperCase();
    const shares = parseFloat(sh);
    if (!tk || !shares || shares <= 0) return;
    onAdd({ id: uid(), ticker: tk, shares, cost: c ? parseFloat(c) : null });
    setT(""); setSh(""); setC("");
  };
  return (
    <div className="add">
      <input className="in" placeholder="Ticker (e.g. AAPL)" value={t} onChange={(e) => setT(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      <input className="in sm" placeholder="Shares" inputMode="decimal" value={sh} onChange={(e) => setSh(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      <input className="in sm" placeholder="Avg cost" inputMode="decimal" value={c} onChange={(e) => setC(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      <button className="add-btn" onClick={submit}><Plus size={15} /> Add holding</button>
    </div>
  );
}

function AddTicker({ onAdd }) {
  const [t, setT] = useState("");
  const submit = () => {
    const tk = t.trim().toUpperCase();
    if (!tk) return;
    onAdd(tk); setT("");
  };
  return (
    <div className="add">
      <input className="in" placeholder="Ticker to watch" value={t} onChange={(e) => setT(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      <button className="add-btn" onClick={submit}><Plus size={15} /> Watch</button>
    </div>
  );
}

// ---------- styles ----------
const css = `
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

.desk{
  --paper:#F1EEE7; --ink:#23241F; --accent:#0F5C5C; --accent-soft:#0f5c5c1a;
  --up:#2E7D5B; --down:#B5503C; --muted:#7C786E; --line:#DCD8CE; --panel:#FBFAF6;
  max-width:920px; margin:0 auto; padding:28px 22px 48px;
  background:var(--paper); color:var(--ink);
  font-family:'Inter',system-ui,sans-serif; -webkit-font-smoothing:antialiased;
  min-height:100vh;
}
.mono{font-family:'IBM Plex Mono',ui-monospace,monospace; font-feature-settings:"tnum";}
.up{color:var(--up);} .down{color:var(--down);}
.r{text-align:right; justify-content:flex-end;}

.hd{display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; padding-bottom:16px; border-bottom:2px solid var(--ink);}
.brand{display:flex; flex-direction:column; gap:2px;}
.mark{font-family:'Fraunces',serif; font-weight:600; font-size:30px; letter-spacing:.06em;}
.tag{font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--muted);}
.hd-right{display:flex; align-items:center; gap:14px;}
.stamp{font-size:11px; color:var(--muted); font-family:'IBM Plex Mono',monospace;}
.refresh{display:inline-flex; align-items:center; gap:7px; background:var(--accent); color:#fff; border:none;
  padding:9px 15px; border-radius:2px; font-size:13px; font-weight:500; cursor:pointer; transition:opacity .15s;}
.refresh:hover{opacity:.88;} .refresh:disabled{opacity:.55; cursor:default;}
.spin{animation:sp 1s linear infinite;} @keyframes sp{to{transform:rotate(360deg);}}

.err{display:flex; align-items:center; gap:8px; margin-top:16px; padding:10px 14px;
  background:#B5503C14; color:var(--down); border-left:3px solid var(--down); font-size:13px;}

.brief{margin:26px 0 8px; padding:4px 0 4px 18px; border-left:3px solid var(--accent);}
.brief-eyebrow{font-size:10.5px; letter-spacing:.22em; color:var(--accent); font-weight:600; margin-bottom:8px;}
.brief-body{font-family:'Fraunces',serif; font-size:21px; line-height:1.42; font-weight:400; margin:0; color:var(--ink);}

.block{margin-top:38px;}
.block-head{display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap; margin-bottom:14px;}
.block-head h2{display:flex; align-items:center; gap:8px; margin:0; font-family:'Fraunces',serif; font-size:19px; font-weight:600;}
.block-head h2 svg{color:var(--accent);}
.totals{display:flex; gap:22px;}
.t-item{display:flex; flex-direction:column; align-items:flex-end; gap:1px;}
.t-label{font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted);}
.t-val{font-family:'IBM Plex Mono',monospace; font-size:15px; font-weight:500;}
.t-val small{font-size:11px; opacity:.8;}

.empty{color:var(--muted); font-size:14px; font-style:italic; padding:14px 0; margin:0; border-top:1px solid var(--line);}

.table{display:flex; flex-direction:column;}
.row{display:grid; grid-template-columns:1.6fr .8fr 1fr 1fr .9fr 1.1fr 1.3fr 32px; gap:10px; align-items:center;
  padding:11px 0; border-bottom:1px solid var(--line); font-size:13.5px;}
.row.head{font-size:10.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); border-bottom:1px solid var(--ink); padding-bottom:8px;}
.row.head span{font-family:'Inter',sans-serif;}
.row span{display:flex; align-items:center;}
.tk{flex-direction:column; align-items:flex-start !important; gap:1px; font-weight:600;}
.tk em{font-style:normal; font-size:11px; font-weight:400; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:150px;}
.row small{font-size:10.5px; opacity:.75; margin-left:3px;}
.del{background:none; border:none; color:var(--muted); cursor:pointer; padding:3px; border-radius:2px; display:flex;}
.del:hover{color:var(--down); background:#B5503C14;}

.chips{display:flex; flex-wrap:wrap; gap:10px;}
.chip{position:relative; background:var(--panel); border:1px solid var(--line); border-radius:3px; padding:12px 16px 11px; min-width:150px; max-width:200px;}
.chip-del{position:absolute; top:6px; right:6px; background:none; border:none; color:var(--line); cursor:pointer; padding:1px; display:flex;}
.chip-del:hover{color:var(--down);}
.chip-tk{font-weight:600; font-size:14px; letter-spacing:.02em;}
.chip-label{font-size:10.5px; color:var(--muted); margin-top:2px; line-height:1.3;}
.chip-price{font-size:15px; margin-top:5px;}
.chip-chg{display:flex; align-items:center; gap:3px; font-size:12px; margin-top:2px;}

.add{display:flex; gap:8px; margin-top:16px; flex-wrap:wrap;}
.in{flex:1; min-width:130px; background:var(--panel); border:1px solid var(--line); border-radius:2px;
  padding:9px 12px; font-size:13px; color:var(--ink); font-family:'Inter',sans-serif;}
.in.sm{flex:0 0 110px; min-width:90px;}
.in:focus{outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft);}
.in::placeholder{color:#a8a399;}
.add-btn{display:inline-flex; align-items:center; gap:6px; background:none; color:var(--accent);
  border:1px solid var(--accent); padding:9px 15px; border-radius:2px; font-size:13px; font-weight:500; cursor:pointer; transition:.15s;}
.add-btn:hover{background:var(--accent); color:#fff;}

.ft{margin-top:42px; padding-top:16px; border-top:1px solid var(--line); font-size:11.5px; color:var(--muted); line-height:1.6;}

@media(max-width:680px){
  .row{grid-template-columns:1.4fr 1fr 1fr 28px; row-gap:2px;}
  .row span:nth-child(2),.row span:nth-child(3),.row.head span:nth-child(2),.row.head span:nth-child(3){display:none;}
  .row span:nth-child(6){display:none;} .row.head span:nth-child(6){display:none;}
  .brief-body{font-size:18px;}
  .totals{gap:16px;}
}
`;
