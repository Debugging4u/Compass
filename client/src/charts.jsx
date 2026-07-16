import { useState } from "react";

// ---------------------------------------------------------------------------
// Categorical palette — fixed hue order (blue, green, magenta, yellow, aqua,
// orange, violet, red). Validated as an adjacent pairlist (bars/lines) against
// this app's panel surface (#FBFAF6): worst adjacent CVD ΔE 9.1, worst
// adjacent normal-vision ΔE 19.6 — both clear their floors. Magenta/yellow/aqua
// sit below 3:1 contrast on this surface, so every chart that uses them always
// carries a visible direct label alongside the color — never color alone.
// ---------------------------------------------------------------------------
const HUES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"];

export function colorMap(names) {
  const map = {};
  names.forEach((name, i) => { map[name] = HUES[i % HUES.length]; });
  return map;
}

function niceTicks(min, max, n) {
  const ticks = [];
  for (let i = 0; i <= n; i++) ticks.push(min + (i / n) * (max - min));
  return ticks;
}

// De-collide a set of end-of-line labels along one axis: push overlapping
// labels apart (forward pass), then compact back inside [lo, hi] if that
// pushed the last one out of bounds. Returns {key: adjustedY}, so callers can
// draw a leader line wherever adjustedY != the label's true data position.
function declutterLabels(items, minGap, lo, hi) {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].y < sorted[i - 1].y + minGap) sorted[i].y = sorted[i - 1].y + minGap;
  }
  const overflow = sorted.length ? sorted[sorted.length - 1].y - hi : 0;
  if (overflow > 0) sorted.forEach((s) => { s.y -= overflow; });
  for (let i = sorted.length - 2; i >= 0; i--) {
    if (sorted[i].y > sorted[i + 1].y - minGap) sorted[i].y = sorted[i + 1].y - minGap;
  }
  sorted.forEach((s) => { s.y = Math.max(lo, Math.min(hi, s.y)); });
  return Object.fromEntries(sorted.map((s) => [s.key, s.y]));
}

// ---------- shared tooltip (fixed-position, follows the pointer) ----------
export function ChartTooltip({ x, y, children }) {
  if (x == null) return null;
  return (
    <div className="viz-tip" style={{ left: x + 14, top: y + 14 }}>
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Weight bars — one horizontal bar per asset, fixed row order (never sorted
// by value) so the same asset lands on the same row across every portfolio
// panel — that's what makes the small-multiples comparison honest.
// ---------------------------------------------------------------------------
export function WeightBars({ weights, colors, maxWeight }) {
  const [hover, setHover] = useState(null);
  const names = Object.keys(weights);

  return (
    <div className="viz-bars">
      {names.map((name) => {
        const v = weights[name];
        const pct = Math.max(0, v) / maxWeight;
        return (
          <div
            className="viz-bar-row"
            key={name}
            tabIndex={0}
            onMouseEnter={(e) => setHover({ name, v, x: e.clientX, y: e.clientY })}
            onMouseMove={(e) => setHover({ name, v, x: e.clientX, y: e.clientY })}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover({ name, v, x: null, y: null })}
          >
            <span className="viz-bar-label">{name}</span>
            <span className="viz-bar-track">
              <span className="viz-bar-fill" style={{ width: `${pct * 100}%`, background: colors[name] }} />
            </span>
            <span className="viz-bar-value mono">{(v * 100).toFixed(1)}%</span>
          </div>
        );
      })}
      {hover && hover.x != null && (
        <ChartTooltip x={hover.x} y={hover.y}>
          <strong>{(hover.v * 100).toFixed(1)}%</strong> {hover.name}
        </ChartTooltip>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Efficient frontier — the frontier curve (single series, no legend needed)
// plus a few directly-labeled reference points (GMV / Max Sharpe / Target).
// ---------------------------------------------------------------------------
export function FrontierChart({ points, markers }) {
  const [hover, setHover] = useState(null);
  const width = 560, height = 300;
  const pad = { l: 50, r: 20, t: 16, b: 34 };

  const allX = [...points.map((p) => p.vol_annual), ...markers.map((m) => m.vol_annual)];
  const allY = [...points.map((p) => p.return_annual), ...markers.map((m) => m.return_annual)];
  const xMax = Math.max(...allX, 0.01) * 1.1;
  const yMin = Math.min(0, ...allY);
  const yMax = Math.max(...allY, 0.01) * 1.15;

  const sx = (v) => pad.l + (v / xMax) * (width - pad.l - pad.r);
  const sy = (v) => height - pad.b - ((v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);

  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.vol_annual).toFixed(2)} ${sy(p.return_annual).toFixed(2)}`)
    .join(" ");

  const yTicks = niceTicks(yMin, yMax, 4);
  const xTicks = niceTicks(0, xMax, 4);

  return (
    <div className="viz-frontier">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Efficient frontier: volatility vs return">
        {yTicks.map((t) => (
          <g key={`y${t}`}>
            <line x1={pad.l} x2={width - pad.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
            <text x={pad.l - 8} y={sy(t)} textAnchor="end" dominantBaseline="middle" className="viz-axis">
              {(t * 100).toFixed(0)}%
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={`x${t}`} x={sx(t)} y={height - pad.b + 20} textAnchor="middle" className="viz-axis">
            {(t * 100).toFixed(0)}%
          </text>
        ))}
        <line x1={pad.l} x2={width - pad.r} y1={height - pad.b} y2={height - pad.b} className="viz-baseline" />

        <path d={path} className="viz-frontier-line" fill="none" />

        {markers.map((m) => (
          <g key={m.key}>
            <circle cx={sx(m.vol_annual)} cy={sy(m.return_annual)} r="6" fill={m.color} className="viz-dot" />
            <text x={sx(m.vol_annual) + 10} y={sy(m.return_annual) + 4} className="viz-point-label">
              {m.label}
            </text>
            {/* transparent hit area — bigger than the visible dot, per interaction spec */}
            <circle
              cx={sx(m.vol_annual)} cy={sy(m.return_annual)} r="14" fill="transparent"
              tabIndex={0}
              onMouseEnter={(e) => setHover({ ...m, x: e.clientX, y: e.clientY })}
              onMouseMove={(e) => setHover({ ...m, x: e.clientX, y: e.clientY })}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover({ ...m, x: null, y: null })}
              style={{ cursor: "pointer" }}
            />
          </g>
        ))}
      </svg>
      <div className="viz-axis-title">Volatility (annualised) · Return (annualised)</div>
      {hover && (
        <ChartTooltip x={hover.x} y={hover.y}>
          <strong>{hover.label}</strong>
          <div>Return {(hover.return_annual * 100).toFixed(1)}% · Vol {(hover.vol_annual * 100).toFixed(1)}%</div>
          {hover.sharpe != null && <div>Sharpe {hover.sharpe.toFixed(2)}</div>}
        </ChartTooltip>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Backtest equity curves — multi-line, always-present legend (>=2 series),
// end-of-line direct labels, and one shared crosshair tooltip listing every
// series at the hovered date.
// ---------------------------------------------------------------------------
export function EquityChart({ series, colors }) {
  const [hover, setHover] = useState(null);
  const names = Object.keys(series);
  const dates = names.length ? Object.keys(series[names[0]]) : [];
  const width = 640, height = 320;
  const pad = { l: 50, r: 120, t: 16, b: 30 };

  if (!dates.length) return null;

  const allVals = names.flatMap((n) => dates.map((d) => series[n][d]));
  const yMin = Math.min(1, ...allVals) * 0.98;
  const yMax = Math.max(...allVals) * 1.02;

  const sx = (i) => pad.l + (i / (dates.length - 1)) * (width - pad.l - pad.r);
  const sy = (v) => height - pad.b - ((v - yMin) / (yMax - yMin)) * (height - pad.t - pad.b);
  const yTicks = niceTicks(yMin, yMax, 4);

  const lastDate = dates[dates.length - 1];
  const labelY = declutterLabels(
    names.map((name) => ({ key: name, y: sy(series[name][lastDate]) })),
    13, pad.t + 6, height - pad.b - 6,
  );

  const handleMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const relX = ((e.clientX - rect.left) / rect.width) * width;
    const idx = Math.round(((relX - pad.l) / (width - pad.l - pad.r)) * (dates.length - 1));
    setHover({ idx: Math.max(0, Math.min(dates.length - 1, idx)), x: e.clientX, y: e.clientY });
  };

  return (
    <div className="viz-equity">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img" aria-label="Backtest equity curves"
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={width - pad.r} y1={sy(t)} y2={sy(t)} className="viz-grid" />
            <text x={pad.l - 8} y={sy(t)} textAnchor="end" dominantBaseline="middle" className="viz-axis">
              {t.toFixed(2)}x
            </text>
          </g>
        ))}
        <line x1={pad.l} x2={width - pad.r} y1={sy(1)} y2={sy(1)} className="viz-baseline" />

        {names.map((name) => {
          const d = dates
            .map((date, i) => `${i === 0 ? "M" : "L"} ${sx(i).toFixed(2)} ${sy(series[name][date]).toFixed(2)}`)
            .join(" ");
          return <path key={name} d={d} stroke={colors[name]} className="viz-line" fill="none" />;
        })}

        {names.map((name) => {
          const rawY = sy(series[name][lastDate]);
          const adjY = labelY[name];
          const lineEndX = sx(dates.length - 1);
          const needsLeader = Math.abs(adjY - rawY) > 1;
          return (
            <g key={`lbl-${name}`}>
              {needsLeader && (
                <line x1={lineEndX + 3} y1={rawY} x2={lineEndX + 14} y2={adjY} className="viz-leader" />
              )}
              <text x={lineEndX + 18} y={adjY} dominantBaseline="middle" className="viz-line-label">
                {name}
              </text>
            </g>
          );
        })}

        {hover && (
          <>
            <line x1={sx(hover.idx)} x2={sx(hover.idx)} y1={pad.t} y2={height - pad.b} className="viz-crosshair" />
            {names.map((name) => (
              <circle
                key={`d-${name}`}
                cx={sx(hover.idx)} cy={sy(series[name][dates[hover.idx]])}
                r="4" fill={colors[name]} className="viz-dot"
              />
            ))}
          </>
        )}
      </svg>

      <div className="viz-legend">
        {names.map((name) => (
          <span className="viz-legend-item" key={name}>
            <span className="viz-legend-key" style={{ background: colors[name] }} />
            {name}
          </span>
        ))}
      </div>

      {hover && (
        <ChartTooltip x={hover.x} y={hover.y}>
          <div className="viz-tip-date">{dates[hover.idx]}</div>
          {names.map((name) => (
            <div key={name} className="viz-tip-row">
              <span className="viz-tip-key" style={{ background: colors[name] }} />
              <strong>{series[name][dates[hover.idx]].toFixed(3)}x</strong> {name}
            </div>
          ))}
        </ChartTooltip>
      )}
    </div>
  );
}
