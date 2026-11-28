import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import {
  EHR_SETS,
  GENERAL_SETS,
  coverageOf,
  fmt,
  frontier2d,
  models,
  sizeOk,
  type Model,
} from '../lib/data';
import { C, seriesColor } from '../lib/colors';
import { Pills, SectionHeader, TierBadge } from './ui';

type Size = 'all' | 'le10' | 'le4';

const W = 760;
const H = 500;
const M = { t: 24, r: 28, b: 56, l: 64 };
const IW = W - M.l - M.r;
const IH = H - M.t - M.b;

const ARROWS: [string, string][] = [
  ['d3-flash', 'Osler-9B'],
  ['pplx-flash', 'Osler-4B'],
];

// candidate index: 0 right, 1 left, 2 above, 3 below
const PREFER: Record<string, number> = { 'Osler-9B': 2, 'pplx-27B': 3, 'Osler-4B': 1 };

function radius(p: number) {
  return 5 + Math.sqrt(p) * 2.3;
}

function colorFor(m: Model) {
  if (m.id === 'Osler-9B') return C.o9;
  if (m.id === 'Osler-4B') return C.o4;
  if (m.id === 'pplx-27B') return C.p27;
  return seriesColor(m.id) === C.other ? C.other : C.base;
}

interface Placed {
  m: Model;
  x: number;
  y: number;
  r: number;
  full: boolean;
  lx: number;
  ly: number;
  anchor: 'start' | 'end' | 'middle';
}

function placeLabels(pts: Omit<Placed, 'lx' | 'ly' | 'anchor'>[]): Placed[] {
  const boxes: { x0: number; x1: number; y0: number; y1: number }[] = [];
  // Reserve the bubbles themselves so labels avoid covering points.
  pts.forEach((p) => boxes.push({ x0: p.x - p.r, x1: p.x + p.r, y0: p.y - p.r, y1: p.y + p.r }));
  const out: Placed[] = [];
  const order = [...pts].sort((a, b) => (b.m.is_osler ? 1 : 0) - (a.m.is_osler ? 1 : 0) || a.y - b.y);
  for (const p of order) {
    const w = p.m.id.length * 6.6 + 4;
    const h = 13;
    const cands: { lx: number; ly: number; anchor: Placed['anchor']; box: (typeof boxes)[number] }[] = [
      { lx: p.x + p.r + 5, ly: p.y + 4, anchor: 'start', box: { x0: p.x + p.r + 3, x1: p.x + p.r + 5 + w, y0: p.y - h / 2, y1: p.y + h / 2 } },
      { lx: p.x - p.r - 5, ly: p.y + 4, anchor: 'end', box: { x0: p.x - p.r - 5 - w, x1: p.x - p.r - 3, y0: p.y - h / 2, y1: p.y + h / 2 } },
      { lx: p.x, ly: p.y - p.r - 6, anchor: 'middle', box: { x0: p.x - w / 2, x1: p.x + w / 2, y0: p.y - p.r - 6 - h, y1: p.y - p.r - 4 } },
      { lx: p.x, ly: p.y + p.r + 14, anchor: 'middle', box: { x0: p.x - w / 2, x1: p.x + w / 2, y0: p.y + p.r + 2, y1: p.y + p.r + 16 } },
      { lx: p.x + p.r + 5, ly: p.y + 17, anchor: 'start', box: { x0: p.x + p.r + 3, x1: p.x + p.r + 5 + w, y0: p.y + 6, y1: p.y + 6 + h } },
      { lx: p.x + p.r + 5, ly: p.y - 9, anchor: 'start', box: { x0: p.x + p.r + 3, x1: p.x + p.r + 5 + w, y0: p.y - 9 - h, y1: p.y - 9 } },
      { lx: p.x - p.r - 5, ly: p.y + 17, anchor: 'end', box: { x0: p.x - p.r - 5 - w, x1: p.x - p.r - 3, y0: p.y + 6, y1: p.y + 6 + h } },
    ];
    const self = { x0: p.x - p.r, x1: p.x + p.r, y0: p.y - p.r, y1: p.y + p.r };
    const hit = (b: (typeof boxes)[number]) =>
      boxes.some(
        (o) => !(o.x0 === self.x0 && o.y0 === self.y0) && b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0,
      ) ||
      b.x0 < 0 ||
      b.x1 > IW + M.r ||
      b.y0 < -M.t;
    // Preferred placements keep labels off the frontier step and the training arrows.
    const pref = PREFER[p.m.id];
    const ordered = pref !== undefined ? [cands[pref], ...cands.filter((_, i) => i !== pref)] : cands;
    const pick = ordered.find((c) => !hit(c.box)) ?? ordered[0];
    boxes.push(pick.box);
    out.push({ ...p, lx: pick.lx, ly: pick.ly, anchor: pick.anchor });
  }
  return out;
}

export function Pareto({ initialSize = 'all' }: { initialSize?: Size }) {
  const [size, setSize] = useState<Size>(initialSize);
  const [hover, setHover] = useState<string | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  // On narrow screens the chart scrolls; start at the right edge, where the frontier is.
  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, []);

  const plotted = models.filter((m) => m.general_medical_avg !== null && m.ehr_docs_avg !== null);
  const xs = plotted.map((m) => m.general_medical_avg!);
  const ys = plotted.map((m) => m.ehr_docs_avg!);
  const x0 = Math.floor((Math.min(...xs) - 0.02) * 50) / 50;
  const x1 = Math.ceil((Math.max(...xs) + 0.015) * 50) / 50;
  const y0 = Math.floor((Math.min(...ys) - 0.02) * 50) / 50;
  const y1 = Math.ceil((Math.max(...ys) + 0.02) * 50) / 50;
  const sx = (v: number) => ((v - x0) / (x1 - x0)) * IW;
  const sy = (v: number) => IH - ((v - y0) / (y1 - y0)) * IH;

  const visible = plotted.filter((m) => sizeOk(m, size));
  // "All" uses the published global front; tier views recompute it on visible full-coverage points.
  const front = size === 'all' ? new Set(models.filter((m) => m.on_pareto_front).map((m) => m.id)) : frontier2d(visible);

  const placed = useMemo(
    () =>
      placeLabels(
        visible.map((m) => ({
          m,
          x: sx(m.general_medical_avg!),
          y: sy(m.ehr_docs_avg!),
          r: radius(m.params_b),
          full: coverageOf(m, GENERAL_SETS) === GENERAL_SETS.length && coverageOf(m, EHR_SETS) === EHR_SETS.length,
        })),
      ),
    [size], // eslint-disable-line

  );

  const frontPts = placed.filter((p) => front.has(p.m.id)).sort((a, b) => a.x - b.x);
  const frontPath =
    frontPts.length > 1
      ? frontPts
          .map((p, i) => {
            if (i === 0) return `M${p.x},${p.y}`;
            const prev = frontPts[i - 1];
            return `L${p.x},${prev.y} L${p.x},${p.y}`;
          })
          .join(' ')
      : '';
  // Staircase envelope: extend to axes so the dominated region is visible.
  const envelope =
    frontPts.length > 0
      ? `M0,${frontPts[0].y} ` +
        frontPts.map((p, i) => (i === 0 ? `L${p.x},${p.y}` : `L${p.x},${frontPts[i - 1].y} L${p.x},${p.y}`)).join(' ') +
        ` L${frontPts[frontPts.length - 1].x},${IH} L0,${IH} Z`
      : '';

  const hovered = placed.find((p) => p.m.id === hover);
  const xticks: number[] = [];
  for (let v = Math.ceil(x0 / 0.05 - 1e-9) * 0.05; v <= x1 + 1e-9; v += 0.05) xticks.push(+v.toFixed(2));
  const yticks: number[] = [];
  for (let v = Math.ceil(y0 / 0.05 - 1e-9) * 0.05; v <= y1 + 1e-9; v += 0.05) yticks.push(+v.toFixed(2));

  const byId = (id: string) => placed.find((p) => p.m.id === id);

  return (
    <section id="pareto" className="section py-20">
      <SectionHeader
        eyebrow="03 · Pareto frontier"
        title={
          <>
            General medical QA <span className="italic text-muted">vs.</span> EHR & clinical documentation
          </>
        }
        lede={
          <>
            X = mean accuracy over <code className="num">confirm_general</code>, <code className="num">knowledge</code>,{' '}
            <code className="num">panel</code>. Y = mean over <code className="num">mr_verdict</code>, <code className="num">mimic_suite</code>.
            Bubble area tracks parameter count; arrows show each Osler model's jump from its zero-shot base.
          </>
        }
        right={
          <Pills
            label="Size filter"
            value={size}
            onChange={setSize}
            options={[
              { value: 'all', label: 'All' },
              { value: 'le10', label: '≤10B' },
              { value: 'le4', label: '≤4B' },
            ]}
          />
        }
      />

      <div className="card relative p-3 sm:p-6">
        <div ref={scroller} className="-mx-3 overflow-x-auto px-3 sm:mx-0 sm:px-0">
        <div className="relative min-w-[640px]">
          <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Pareto scatter of general medical vs EHR accuracy">
            <defs>
              <marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,1 L9,5 L0,9 Z" fill={C.muted} />
              </marker>
            </defs>
            <g transform={`translate(${M.l},${M.t})`}>
              <g className="chart-axis">
                {xticks.map((t) => (
                  <g key={`x${t}`}>
                    <line className="chart-grid" x1={sx(t)} x2={sx(t)} y1={0} y2={IH} strokeDasharray="2 4" />
                    <text x={sx(t)} y={IH + 18} textAnchor="middle">
                      {t.toFixed(2)}
                    </text>
                  </g>
                ))}
                {yticks.map((t) => (
                  <g key={`y${t}`}>
                    <line className="chart-grid" x1={0} x2={IW} y1={sy(t)} y2={sy(t)} strokeDasharray="2 4" />
                    <text x={-10} y={sy(t) + 3.5} textAnchor="end">
                      {t.toFixed(2)}
                    </text>
                  </g>
                ))}
                <line x1={0} x2={IW} y1={IH} y2={IH} stroke={C.line} />
                <line x1={0} x2={0} y1={0} y2={IH} stroke={C.line} />
              </g>
              <text x={IW / 2} y={IH + 44} textAnchor="middle" className="fill-muted font-sans text-[12px]">
                General Medical QA (mean accuracy) →
              </text>
              <text transform={`translate(${-48},${IH / 2}) rotate(-90)`} textAnchor="middle" className="fill-muted font-sans text-[12px]">
                EHR & Clinical Docs (mean accuracy) →
              </text>

              {envelope && <path d={envelope} fill={C.o9} opacity={0.045} />}
              {frontPath && (
                <path d={frontPath} fill="none" stroke={C.ink} strokeWidth={1.4} strokeOpacity={0.55} strokeLinejoin="round" />
              )}

              {ARROWS.map(([a, b]) => {
                const pa = byId(a);
                const pb = byId(b);
                if (!pa || !pb) return null;
                const dx = pb.x - pa.x;
                const dy = pb.y - pa.y;
                const len = Math.hypot(dx, dy) || 1;
                const ux = dx / len;
                const uy = dy / len;
                return (
                  <line
                    key={a + b}
                    x1={pa.x + ux * (pa.r + 3)}
                    y1={pa.y + uy * (pa.r + 3)}
                    x2={pb.x - ux * (pb.r + 4)}
                    y2={pb.y - uy * (pb.r + 4)}
                    stroke={C.muted}
                    strokeWidth={1.3}
                    strokeDasharray="5 4"
                    markerEnd="url(#arrowhead)"
                  />
                );
              })}

              {placed.map((p) => {
                const col = colorFor(p.m);
                const isFront = front.has(p.m.id);
                const dim = hover && hover !== p.m.id;
                return (
                  <g
                    key={p.m.id}
                    tabIndex={0}
                    role="button"
                    aria-label={`${p.m.id}: general ${fmt(p.m.general_medical_avg)}, EHR ${fmt(p.m.ehr_docs_avg)}`}
                    onMouseEnter={() => setHover(p.m.id)}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover(p.m.id)}
                    onBlur={() => setHover(null)}
                    className="cursor-pointer outline-none"
                    style={{ opacity: dim ? 0.35 : 1, transition: 'opacity 150ms' }}
                  >
                    {isFront && <circle cx={p.x} cy={p.y} r={p.r + 4.5} fill="none" stroke={col} strokeOpacity={0.35} strokeWidth={1} />}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={p.r}
                      fill={p.full ? col : C.card}
                      fillOpacity={p.full ? (p.m.is_osler || p.m.id === 'pplx-27B' ? 0.9 : 0.55) : 1}
                      stroke={col}
                      strokeWidth={p.full ? 1.5 : 1.6}
                      strokeDasharray={p.full ? undefined : '3 2'}
                    />
                    <text
                      x={p.lx}
                      y={p.ly}
                      textAnchor={p.anchor}
                      className={`font-mono text-[11px] ${p.m.is_osler ? 'font-semibold' : ''}`}
                      fill={p.m.is_osler || isFront ? C.ink : C.muted}
                    >
                      {p.m.id}
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>

          {hovered && (
            <div
              className="pointer-events-none absolute z-10 w-64 rounded-xl border border-line bg-card/95 p-3.5 text-[12px] shadow-xl backdrop-blur"
              style={{
                left: `${((hovered.x + M.l) / W) * 100}%`,
                top: `${((hovered.y + M.t) / H) * 100}%`,
                transform: `translate(${hovered.x > IW * 0.6 ? 'calc(-100% - 18px)' : '18px'}, -50%)`,
              }}
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="font-semibold text-ink">{hovered.m.id}</span>
                <TierBadge tier={hovered.m.verification_tier} />
              </div>
              <div className="mb-2 text-[11px] text-muted">{hovered.m.role}</div>
              <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
                <dt className="text-muted">General QA</dt>
                <dd className="num text-right text-ink">
                  {fmt(hovered.m.general_medical_avg)}
                  <span className="ml-1 text-muted">
                    ({coverageOf(hovered.m, GENERAL_SETS)}/{GENERAL_SETS.length} sets)
                  </span>
                </dd>
                <dt className="text-muted">EHR & Docs</dt>
                <dd className="num text-right text-ink">
                  {fmt(hovered.m.ehr_docs_avg)}
                  <span className="ml-1 text-muted">
                    ({coverageOf(hovered.m, EHR_SETS)}/{EHR_SETS.length} sets)
                  </span>
                </dd>
                <dt className="text-muted">5-set macro</dt>
                <dd className="num text-right text-ink">{hovered.m.five_set_macro !== null ? fmt(hovered.m.five_set_macro) : 'NOT MEASURED'}</dd>
                <dt className="text-muted">Params</dt>
                <dd className="num text-right text-ink">{hovered.m.params_b.toFixed(2)}B</dd>
                <dt className="text-muted">p50 (b=1)</dt>
                <dd className="num text-right text-ink">{hovered.m.p50_ms_b1 !== null ? `${hovered.m.p50_ms_b1.toFixed(2)} ms` : 'NOT MEASURED'}</dd>
                <dt className="text-muted">Peak VRAM</dt>
                <dd className="num text-right text-ink">{hovered.m.peak_vram_gb !== null ? `${hovered.m.peak_vram_gb.toFixed(2)} GB` : 'NOT MEASURED'}</dd>
              </dl>
              {!hovered.full && (
                <p className="mt-2 border-t border-line pt-2 text-[11px] leading-snug text-warn">
                  Partial coverage: axis value averages only the measured sets; missing cells are not imputed.
                </p>
              )}
            </div>
          )}
        </div>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-line px-2 pt-4 text-[12px] text-muted">
          <span className="flex items-center gap-1.5">
            <svg width="22" height="10"><path d="M0,8 L11,8 L11,2 L22,2" stroke={C.ink} strokeOpacity={0.55} fill="none" strokeWidth="1.4" /></svg>
            {size === 'all' ? 'Global Pareto front' : 'Frontier among visible full-coverage models'}
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="24" height="10"><line x1="0" y1="5" x2="24" y2="5" stroke={C.muted} strokeDasharray="5 4" strokeWidth="1.3" /></svg>
            Zero-shot base → Osler (training jump)
          </span>
          <span className="flex items-center gap-1.5">
            <svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="none" stroke={C.muted} strokeDasharray="3 2" strokeWidth="1.5" /></svg>
            Hollow = partial coverage (excluded from frontier)
          </span>
          <span className="ml-auto flex items-center gap-1 text-[11px]">
            Source <code className="num">loops/osler_rc0/PARETO.md</code> <ArrowUpRight size={12} />
          </span>
        </div>
      </div>
    </section>
  );
}
