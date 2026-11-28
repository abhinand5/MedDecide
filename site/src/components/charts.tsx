// Small, dependency-free SVG chart primitives shared by the analytics tabs.
import { useState, type ReactNode } from 'react';
import { C } from '../lib/colors';

export interface Series {
  id: string;
  color: string;
  dashed?: boolean;
  points: { x: number; y: number; n?: number; label?: string }[];
}

export function LineChart({
  series,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  xFmt = (v) => v.toFixed(1),
  yFmt = (v) => v.toFixed(1),
  diagonal = false,
  sizeByN = false,
  xTicks,
  yTicks,
  annotations,
  height = 380,
}: {
  series: Series[];
  xDomain: [number, number];
  yDomain: [number, number];
  xLabel: string;
  yLabel: string;
  xFmt?: (v: number) => string;
  yFmt?: (v: number) => string;
  diagonal?: boolean;
  sizeByN?: boolean;
  xTicks: number[];
  yTicks: number[];
  annotations?: (sx: (v: number) => number, sy: (v: number) => number) => ReactNode;
  height?: number;
}) {
  const W = 640;
  const H = height;
  const M = { t: 16, r: 20, b: 48, l: 56 };
  const IW = W - M.l - M.r;
  const IH = H - M.t - M.b;
  const sx = (v: number) => ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * IW;
  const sy = (v: number) => IH - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * IH;
  const [hover, setHover] = useState<{ s: string; i: number } | null>(null);
  const maxN = Math.max(1, ...series.flatMap((s) => s.points.map((p) => p.n ?? 1)));
  const r = (n?: number) => (sizeByN ? 2 + Math.sqrt((n ?? 1) / maxN) * 7 : 3.2);
  const hs = hover ? series.find((s) => s.id === hover.s) : undefined;
  const hp = hs ? hs.points[hover!.i] : undefined;

  return (
    <div className="overflow-x-auto">
    <div className="relative min-w-[520px]">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`${yLabel} vs ${xLabel}`}>
        <g transform={`translate(${M.l},${M.t})`}>
          <g className="chart-axis">
            {xTicks.map((t) => (
              <g key={`x${t}`}>
                <line className="chart-grid" x1={sx(t)} x2={sx(t)} y1={0} y2={IH} strokeDasharray="2 4" />
                <text x={sx(t)} y={IH + 17} textAnchor="middle">
                  {xFmt(t)}
                </text>
              </g>
            ))}
            {yTicks.map((t) => (
              <g key={`y${t}`}>
                <line className="chart-grid" x1={0} x2={IW} y1={sy(t)} y2={sy(t)} strokeDasharray="2 4" />
                <text x={-9} y={sy(t) + 3.5} textAnchor="end">
                  {yFmt(t)}
                </text>
              </g>
            ))}
            <line x1={0} x2={IW} y1={IH} y2={IH} stroke={C.line} />
            <line x1={0} x2={0} y1={0} y2={IH} stroke={C.line} />
          </g>
          <text x={IW / 2} y={IH + 40} textAnchor="middle" className="fill-muted font-sans text-[11.5px]">
            {xLabel}
          </text>
          <text transform={`translate(-42,${IH / 2}) rotate(-90)`} textAnchor="middle" className="fill-muted font-sans text-[11.5px]">
            {yLabel}
          </text>
          {diagonal && (
            <line
              x1={sx(Math.max(xDomain[0], yDomain[0]))}
              y1={sy(Math.max(xDomain[0], yDomain[0]))}
              x2={sx(Math.min(xDomain[1], yDomain[1]))}
              y2={sy(Math.min(xDomain[1], yDomain[1]))}
              stroke={C.muted}
              strokeDasharray="4 4"
              strokeWidth={1}
            />
          )}
          {annotations?.(sx, sy)}
          {series.map((s) => {
            const d = s.points.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join(' ');
            const dim = hover && hover.s !== s.id;
            return (
              <g key={s.id} style={{ opacity: dim ? 0.3 : 1, transition: 'opacity 150ms' }}>
                <path d={d} fill="none" stroke={s.color} strokeWidth={1.8} strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" />
                {s.points.map((p, i) => (
                  <circle
                    key={i}
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={r(p.n)}
                    fill={s.dashed ? C.card : s.color}
                    stroke={s.color}
                    strokeWidth={1.4}
                    onMouseEnter={() => setHover({ s: s.id, i })}
                    onMouseLeave={() => setHover(null)}
                    className="cursor-pointer"
                  />
                ))}
              </g>
            );
          })}
        </g>
      </svg>
      {hs && hp && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-line bg-card/95 px-3 py-2 text-[11.5px] shadow-lg backdrop-blur"
          style={{
            left: `${((sx(hp.x) + M.l) / W) * 100}%`,
            top: `${((sy(hp.y) + M.t) / H) * 100}%`,
            transform: `translate(${sx(hp.x) > IW * 0.6 ? 'calc(-100% - 12px)' : '12px'}, -110%)`,
          }}
        >
          <div className="mb-1 flex items-center gap-1.5 font-medium text-ink">
            <span className="h-2 w-2 rounded-full" style={{ background: hs.color }} />
            {hs.id}
          </div>
          <div className="num text-muted">{hp.label ?? `${xFmt(hp.x)} → ${yFmt(hp.y)}`}</div>
        </div>
      )}
    </div>
    </div>
  );
}

export function Legend({ items }: { items: { id: string; color: string; dashed?: boolean; note?: string }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[12px]">
      {items.map((it) => (
        <span key={it.id} className="inline-flex items-center gap-1.5 text-muted">
          <svg width="18" height="8">
            <line x1="0" y1="4" x2="18" y2="4" stroke={it.color} strokeWidth="2" strokeDasharray={it.dashed ? '4 3' : undefined} />
          </svg>
          <span className="text-ink">{it.id}</span>
          {it.note && <span className="num text-[11px]">{it.note}</span>}
        </span>
      ))}
    </div>
  );
}
