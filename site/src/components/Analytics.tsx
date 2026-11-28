import { useState, type ReactNode } from 'react';
import { Activity, AlertTriangle, Gauge, Layers, Lock, ShieldCheck, ShieldOff } from 'lucide-react';
import { VERBATIM_PROBE_4B, data, fmt, getModel, metric, models, pct, type SetId } from '../lib/data';
import { C, seriesColor } from '../lib/colors';
import { LineChart, Legend } from './charts';
import { Pills, SectionHeader } from './ui';

export type Tab = 'calibration' | 'selective' | 'cardinality' | 'privacy';

// ---------------------------------------------------------------- Calibration
const REL_SETS: SetId[] = ['panel', 'mr_verdict', 'confirm_general'];
const REL_DEFAULT = ['Osler-9B', 'Osler-9B (raw)', 'd3-flash', 'pplx-27B'];

function binStats(id: string, set: string) {
  const bins = data.reliability[set]?.[id];
  if (!bins || !bins.length) return null;
  const N = bins.reduce((a, b) => a + b.n, 0);
  const ece = bins.reduce((a, b) => a + b.n * Math.abs(b.acc - b.conf), 0) / N;
  const gap = bins.reduce((a, b) => a + b.n * (b.acc - b.conf), 0) / N; // > 0 ⇒ under-confident
  return { N, ece, gap };
}

function Calibration({ initialSet = 'panel' }: { initialSet?: SetId }) {
  const [set, setSet] = useState<SetId>(initialSet);
  const available = Object.keys(data.reliability[set] ?? {});
  const [on, setOn] = useState<string[]>(REL_DEFAULT);
  const shown = available.filter((id) => on.includes(id));
  const series = shown.map((id) => ({
    id,
    color: seriesColor(id),
    dashed: id.endsWith('(raw)'),
    points: (data.reliability[set]?.[id] ?? []).map((b) => ({
      x: b.conf,
      y: b.acc,
      n: b.n,
      label: `conf ${b.conf.toFixed(3)} · acc ${b.acc.toFixed(3)} · n=${b.n.toLocaleString()}`,
    })),
  }));
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const s9 = binStats('Osler-9B', set);
  const s9r = binStats('Osler-9B (raw)', set);
  const s4 = binStats('Osler-4B', set);
  const s4r = binStats('Osler-4B (raw)', set);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div>
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <Pills size="sm" label="Eval set" value={set} onChange={setSet} options={REL_SETS.map((s) => ({ value: s, label: s }))} />
        </div>
        <LineChart
          series={series}
          xDomain={[0, 1]}
          yDomain={[0, 1]}
          xTicks={ticks}
          yTicks={ticks}
          xLabel="Mean confidence (top option)"
          yLabel="Observed accuracy"
          diagonal
          sizeByN
        />
        <p className="mt-2 text-[11.5px] text-muted">Marker area ∝ items in bin; sparse low-confidence bins hold only a handful of items.</p>
      </div>
      <div className="space-y-4">
        <div>
          <div className="eyebrow mb-2">Series · bin-derived ECE</div>
          <div className="space-y-1.5">
            {available.map((id) => {
              const st = binStats(id, set);
              const checked = on.includes(id);
              return (
                <label key={id} className="flex cursor-pointer items-center justify-between gap-3 rounded-lg border border-line px-3 py-1.5 text-[12.5px] hover:bg-subtle/60">
                  <span className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => setOn((o) => (checked ? o.filter((x) => x !== id) : [...o, id]))}
                      className="accent-[rgb(var(--accent))]"
                    />
                    <Legend items={[{ id, color: seriesColor(id), dashed: id.endsWith('(raw)') }]} />
                  </span>
                  <span className="num text-muted">{st ? st.ece.toFixed(4) : '—'}</span>
                </label>
              );
            })}
          </div>
        </div>
        {set === 'confirm_general' && s9 && s9r && s4 && s4r ? (
          <div className="rounded-xl border border-warn/30 bg-warn/[0.06] p-4 text-[13px] leading-relaxed text-ink">
            <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-warn">
              <AlertTriangle size={14} /> Honest note: under-confident on fresh items
            </div>
            The temperatures were fitted on dev only, and on <code className="num">confirm_general</code> they leave both models
            under-confident. Accuracy exceeds confidence by <span className="num">{(s9.gap * 100).toFixed(1)}</span> pts (9B) and{' '}
            <span className="num">{(s4.gap * 100).toFixed(1)}</span> pts (4B) on average. Calibrated ECE is{' '}
            <span className="num">{s9.ece.toFixed(3)}</span> / <span className="num">{s4.ece.toFixed(3)}</span>, against{' '}
            <span className="num">{s9r.ece.toFixed(3)}</span> / <span className="num">{s4r.ece.toFixed(3)}</span> for the raw outputs. Nothing
            was refitted on test.
          </div>
        ) : (
          s9 &&
          s9r && (
            <div className="rounded-xl border border-accent/30 bg-accent/[0.05] p-4 text-[13px] leading-relaxed text-ink">
              <div className="mb-1.5 flex items-center gap-1.5 font-semibold text-accent">
                <Activity size={14} /> Dev-fitted calibration on <code className="num">{set}</code>
              </div>
              Osler-9B ECE goes from <span className="num">{s9r.ece.toFixed(4)}</span> raw to <span className="num">{s9.ece.toFixed(4)}</span> calibrated (
              <span className="num">n={s9.N.toLocaleString()}</span>). Temperatures are fitted per question type on dev splits and never on test.
            </div>
          )
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Selective
const SEL_SETS = ['mr_verdict', 'mimic_suite', 'panel'];

function Selective({ initialSet = 'mr_verdict' }: { initialSet?: string }) {
  const [set, setSet] = useState(initialSet);
  const per = data.selective_prediction[set] ?? {};
  const ids = Object.keys(per);
  const series = ids.map((id) => ({
    id,
    color: seriesColor(id),
    points: (per[id] ?? []).map((p) => ({
      x: p.coverage,
      y: p.accuracy,
      label: `coverage ${pct(p.coverage)} · acc ${pct(p.accuracy)} · τ ≥ ${p.min_conf.toFixed(3)}`,
    })),
  }));
  const allY = series.flatMap((s) => s.points.map((p) => p.y));
  const yMin = Math.floor(Math.min(...allY) * 20) / 20;
  const yTicks: number[] = [];
  for (let v = yMin; v <= 1.0001; v += 0.05) yTicks.push(+v.toFixed(2));
  const at = (id: string, c: number) => per[id]?.find((p) => Math.abs(p.coverage - c) < 0.01);
  const o80 = at('Osler-9B', 0.8);
  const o50 = at('Osler-9B', 0.5);
  const o100 = at('Osler-9B', 1.0);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
      <div>
        <div className="mb-4">
          <Pills size="sm" label="Eval set" value={set} onChange={setSet} options={SEL_SETS.map((s) => ({ value: s, label: s }))} />
        </div>
        <LineChart
          series={series}
          xDomain={[0.1, 1]}
          yDomain={[yMin, 1]}
          xTicks={[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
          yTicks={yTicks}
          xFmt={(v) => `${Math.round(v * 100)}%`}
          yFmt={(v) => `${Math.round(v * 100)}%`}
          xLabel="Coverage (fraction answered, most-confident first)"
          yLabel="Accuracy on answered items"
          annotations={(sx, sy) =>
            [o50, o80].map(
              (p, i) =>
                p && (
                  <g key={i}>
                    <line x1={sx(p.coverage)} x2={sx(p.coverage)} y1={sy(p.accuracy)} y2={sy(yMin)} stroke={C.o9} strokeOpacity={0.35} strokeDasharray="2 3" />
                    <circle cx={sx(p.coverage)} cy={sy(p.accuracy)} r={7} fill="none" stroke={C.o9} strokeWidth={1.4} />
                  </g>
                ),
            )
          }
        />
        <div className="mt-3">
          <Legend items={ids.map((id) => ({ id, color: seriesColor(id) }))} />
        </div>
      </div>
      <div className="space-y-4">
        <div className="grid grid-cols-2 gap-3">
          {[
            { k: '80% coverage', p: o80 },
            { k: '50% coverage', p: o50 },
          ].map(({ k, p }) => (
            <div key={k} className="card p-4">
              <div className="eyebrow mb-1">Osler-9B · {k}</div>
              <div className="num text-[1.6rem] font-semibold text-accent">{p ? pct(p.accuracy) : '—'}</div>
              <div className="num text-[11px] text-muted">abstain if conf &lt; {p ? p.min_conf.toFixed(3) : '—'}</div>
            </div>
          ))}
        </div>
        <p className="text-[13px] leading-relaxed text-muted">
          Calibrated probabilities make abstention a single threshold. On <code className="num">{set}</code> the 9B answers every item at{' '}
          <span className="num text-ink">{pct(o100?.accuracy)}</span>. If it answers only its most confident 80% of items, accuracy is{' '}
          <span className="num text-ink">{pct(o80?.accuracy)}</span>, and at 50% it is <span className="num text-ink">{pct(o50?.accuracy)}</span>.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-muted">
                <th className="pb-1.5 text-left font-medium">Model</th>
                <th className="pb-1.5 text-right font-medium">@50%</th>
                <th className="pb-1.5 text-right font-medium">@80%</th>
                <th className="pb-1.5 text-right font-medium">@100%</th>
              </tr>
            </thead>
            <tbody>
              {ids.map((id) => (
                <tr key={id} className="border-t border-line/70">
                  <td className={`py-1.5 ${id.startsWith('Osler') ? 'font-medium text-accent' : 'text-ink'}`}>{id}</td>
                  {[0.5, 0.8, 1].map((c) => (
                    <td key={c} className="num py-1.5 text-right">
                      {pct(at(id, c)?.accuracy)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Cardinality
const BUCKETS = ['2', '3-4', '5-10', '11-26'];
const OPT_SETS = ['mr_verdict', 'panel', 'mimic_suite'];

function OptionScaling({ initialSet = 'mr_verdict' }: { initialSet?: string }) {
  const [set, setSet] = useState(initialSet);
  const buckets = data.option_scaling[set] ?? {};
  const present = BUCKETS.filter((b) => buckets[b]);
  const ids = Array.from(new Set(present.flatMap((b) => Object.keys(buckets[b]?.models ?? {}))));
  const order = ['Osler-9B', 'Osler-4B', 'd3-flash', 'pplx-flash', 'pplx-27B', 'MedDecider-9B'];
  ids.sort((a, b) => (order.indexOf(a) + 99) % 99 - ((order.indexOf(b) + 99) % 99));

  const W = 1040;
  const H = 380;
  const M = { t: 14, r: 10, b: 48, l: 44 };
  const IW = W - M.l - M.r;
  const IH = H - M.t - M.b;
  const gw = IW / present.length;
  const bw = Math.min(16, (gw - 24) / ids.length);
  const sy = (v: number) => IH - v * IH;

  return (
    <div>
      <div className="mb-4">
        <Pills size="sm" label="Eval set" value={set} onChange={setSet} options={OPT_SETS.map((s) => ({ value: s, label: s }))} />
      </div>
      <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[620px]" role="img" aria-label="Accuracy by number of options">
        <g transform={`translate(${M.l},${M.t})`}>
          <g className="chart-axis">
            {[0, 0.25, 0.5, 0.75, 1].map((t) => (
              <g key={t}>
                <line className="chart-grid" x1={0} x2={IW} y1={sy(t)} y2={sy(t)} strokeDasharray="2 4" />
                <text x={-8} y={sy(t) + 3.5} textAnchor="end">
                  {Math.round(t * 100)}%
                </text>
              </g>
            ))}
          </g>
          {present.map((b, gi) => {
            const bucket = buckets[b]!;
            const gx = gi * gw + (gw - bw * ids.length) / 2;
            return (
              <g key={b}>
                {ids.map((id, i) => {
                  const v = bucket.models[id];
                  const x = gx + i * bw;
                  if (!v)
                    return (
                      <g key={id}>
                        <line x1={x + 2} x2={x + bw - 2} y1={IH - 1} y2={IH - 1} stroke={seriesColor(id)} strokeWidth={2} strokeDasharray="2 2" />
                        <text x={x + bw / 2} y={IH - 6} textAnchor="middle" className="fill-muted font-mono text-[8px]">
                          n/a
                        </text>
                      </g>
                    );
                  return (
                    <rect key={id} x={x + 1} y={sy(v.accuracy)} width={bw - 2} height={IH - sy(v.accuracy)} rx={2} fill={seriesColor(id)} opacity={id.startsWith('Osler') ? 0.95 : 0.6}>
                      <title>{`${id} · ${b} options · acc ${v.accuracy.toFixed(4)} · Brier ${v.brier.toFixed(4)}`}</title>
                    </rect>
                  );
                })}
                <text x={gi * gw + gw / 2} y={IH + 18} textAnchor="middle" className="fill-ink font-mono text-[11px]">
                  {b} options
                </text>
                <text x={gi * gw + gw / 2} y={IH + 32} textAnchor="middle" className="fill-muted font-mono text-[10px]">
                  n={bucket.n.toLocaleString()}
                </text>
              </g>
            );
          })}
          <line x1={0} x2={IW} y1={IH} y2={IH} stroke={C.line} />
        </g>
      </svg>
      </div>
      <div className="mt-2">
        <Legend items={ids.map((id) => ({ id, color: seriesColor(id) }))} />
      </div>
      <p className="mt-3 text-[12.5px] leading-relaxed text-muted">
        <span className="num">n/a</span> marks a model with no score in that bucket. MedDecider's verbalizer readout is capped at 10 options,
        so it cannot answer 11–26-option items; the Osler option-code readout covers up to 255 codes.
        {!buckets['11-26'] && ' This set has no items with more than 10 options.'}
      </p>
    </div>
  );
}

function Efficiency() {
  const measured = models.filter((m) => m.p50_ms_b1 !== null && m.peak_vram_gb !== null).sort((a, b) => a.p50_ms_b1! - b.p50_ms_b1!);
  const missing = models.filter((m) => m.p50_ms_b1 === null || m.peak_vram_gb === null);
  const maxL = Math.max(...measured.map((m) => m.p50_ms_b1!));
  const maxV = Math.max(...measured.map((m) => m.peak_vram_gb!));
  const Row = ({ label, v, max, unit, color }: { label: string; v: number; max: number; unit: string; color: string }) => (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-[12px] text-ink">{label}</span>
      <div className="relative h-5 flex-1 rounded bg-subtle">
        <div className="bar-grow absolute inset-y-0 left-0 rounded" style={{ width: `${(v / max) * 100}%`, background: color, opacity: 0.85 }} />
      </div>
      <span className="num w-20 shrink-0 text-right text-[12px]">
        {v.toFixed(2)} {unit}
      </span>
    </div>
  );
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div>
        <div className="eyebrow mb-3">Batch-1 p50 latency per item (↓)</div>
        <div className="space-y-2">
          {measured.map((m) => (
            <Row key={m.id} label={m.id} v={m.p50_ms_b1!} max={maxL} unit="ms" color={seriesColor(m.id)} />
          ))}
        </div>
      </div>
      <div>
        <div className="eyebrow mb-3">Peak VRAM (↓)</div>
        <div className="space-y-2">
          {measured.map((m) => (
            <Row key={m.id} label={m.id} v={m.peak_vram_gb!} max={maxV} unit="GB" color={seriesColor(m.id)} />
          ))}
        </div>
      </div>
      <p className="text-[12px] text-muted md:col-span-2">
        Measured under the same RC0 latency harness. NOT MEASURED (outside the latency timebox, or the runner cannot batch):{' '}
        {missing.map((m) => m.id).join(', ')}.
      </p>
    </div>
  );
}

function Cardinality() {
  return (
    <div className="space-y-10">
      <div>
        <h4 className="mb-1 flex items-center gap-2 text-[15px] font-semibold text-ink">
          <Layers size={15} className="text-accent" /> Accuracy by option count
        </h4>
        <p className="mb-4 text-[13px] text-muted">The same models, split by how many options each item offers.</p>
        <OptionScaling />
      </div>
      <div className="hairline pt-8">
        <h4 className="mb-4 flex items-center gap-2 text-[15px] font-semibold text-ink">
          <Gauge size={15} className="text-accent" /> Standardized efficiency
        </h4>
        <Efficiency />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Privacy & limitations
function acc(id: string, set: SetId) {
  const m = getModel(id);
  return m ? metric(m, set, 'accuracy') : null;
}
function srcAcc(set: string, src: string, id: string): number | null {
  const v = data.per_source[set]?.[src]?.models[id] as { accuracy?: number } | undefined;
  return typeof v?.accuracy === 'number' ? v.accuracy : null;
}

function Privacy() {
  const o9 = getModel('Osler-9B');
  const o4 = getModel('Osler-4B');
  const cg9 = o9?.sets.confirm_general;
  const cg4 = o4?.sets.confirm_general;
  const mrh9 = o9?.sets.mr_heldout_test;
  const lim: { title: string; body: ReactNode }[] = [
    {
      title: 'Osler-9B ties its base on out-of-domain QA',
      body: (
        <>
          On the external panel the 9B scores <N v={acc('Osler-9B', 'panel')} /> against <N v={acc('d3-flash', 'panel')} /> for d3-flash. On
          knowledge it scores <N v={acc('Osler-9B', 'knowledge')} /> against <N v={acc('d3-flash', 'knowledge')} />. These are ties, not gains.
        </>
      ),
    },
    {
      title: 'Trails pplx-27B on hard multi-step QA',
      body: (
        <>
          Knowledge: <N v={acc('Osler-9B', 'knowledge')} /> vs <N v={acc('pplx-27B', 'knowledge')} />. MedXpertQA (text):{' '}
          <N v={srcAcc('panel', 'medxpertqa_text', 'Osler-9B')} /> vs <N v={srcAcc('panel', 'medxpertqa_text', 'pplx-27B')} />, and also
          below the d3-flash base (<N v={srcAcc('panel', 'medxpertqa_text', 'd3-flash')} />).
        </>
      ),
    },
    {
      title: 'Osler-4B gives up about 1 pt out of domain',
      body: (
        <>
          Against its pplx-flash base: panel <N v={acc('Osler-4B', 'panel')} /> vs <N v={acc('pplx-flash', 'panel')} />, knowledge{' '}
          <N v={acc('Osler-4B', 'knowledge')} /> vs <N v={acc('pplx-flash', 'knowledge')} />. On fresh <code className="num">confirm_general</code>{' '}
          the 4B (<N v={acc('Osler-4B', 'confirm_general')} />) is level with its base (<N v={acc('pplx-flash', 'confirm_general')} />). Only the
          9B leads that set.
        </>
      ),
    },
    {
      title: 'Under-confident on fresh items',
      body: (
        <>
          The dev-fitted temperatures give an ECE of <N v={cg9?.ece ?? null} d={3} /> (9B) and <N v={cg4?.ece ?? null} d={3} /> (4B) on{' '}
          <code className="num">confirm_general</code>. The reliability curves sit above the diagonal; see the Calibration tab.
        </>
      ),
    },
    {
      title: 'Fail a held-out synthetic indicator template',
      body: (
        <>
          On <code className="num">synthea_indicator_evidence_v1</code>, a template excluded from training, both release candidates score
          below chance while their bases pass. The set holding it (<code className="num">mr_heldout_test</code>, 3 templates) is flagged{' '}
          <code className="num">{mrh9?.readout_health}</code>. Accuracy over the whole set: 9B <N v={acc('Osler-9B', 'mr_heldout_test')} /> vs
          d3-flash <N v={acc('d3-flash', 'mr_heldout_test')} />; 4B <N v={acc('Osler-4B', 'mr_heldout_test')} /> vs pplx-flash{' '}
          <N v={acc('pplx-flash', 'mr_heldout_test')} />.
        </>
      ),
    },
    {
      title: 'Held-out MIMIC family is not a win',
      body: (
        <>
          On the held-out heart-failure ICD family (<code className="num">mimic_heldout</code>, n={o9?.sets.mimic_heldout?.n}) the 9B scores{' '}
          <N v={acc('Osler-9B', 'mimic_heldout')} />, against <N v={acc('pplx-27B', 'mimic_heldout')} /> for pplx-27B and{' '}
          <N v={acc('Clef-Flash', 'mimic_heldout')} /> for Clef-Flash. The 95% CI is wide: [<N v={o9?.sets.mimic_heldout?.ci_lo ?? null} />,{' '}
          <N v={o9?.sets.mimic_heldout?.ci_hi ?? null} />].
        </>
      ),
    },
  ];

  return (
    <div className="space-y-8">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="card overflow-x-auto p-5">
          <div className="mb-3 flex items-center gap-2">
            <Lock size={15} className="text-accent" />
            <h4 className="text-[15px] font-semibold text-ink">D28 local MIMIC membership audit</h4>
          </div>
          <table className="w-full min-w-[520px] text-[12.5px]">
            <thead>
              <tr className="text-left text-muted">
                <th className="pb-2 font-medium">Model</th>
                <th className="pb-2 text-right font-medium">Visited-member AUC</th>
                <th className="pb-2 text-right font-medium">95% CI</th>
                <th className="pb-2 text-right font-medium">Control (zero-shot)</th>
                <th className="pb-2 text-right font-medium">ΔAUC</th>
                <th className="pb-2 text-right font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {data.phi_audit.map((r) => (
                <tr key={r.model} className="border-t border-line/70">
                  <td className="py-2 font-medium text-accent">{r.model}</td>
                  <td className="num py-2 text-right">{fmt(r.auc)}</td>
                  <td className="num py-2 text-right text-muted">
                    [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]
                  </td>
                  <td className="num py-2 text-right">
                    {fmt(r.control_auc)} <span className="text-muted">({r.control})</span>
                  </td>
                  <td className="num py-2 text-right">
                    {r.delta_auc >= 0 ? '+' : ''}
                    {fmt(r.delta_auc)}
                  </td>
                  <td className="py-2 text-right">
                    <span className={`badge ${r.clear ? 'bg-accent/10 text-accent' : 'bg-bad/10 text-bad'}`}>{r.clear ? 'CLEAR' : 'NOT CLEAR'}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-[11.5px] leading-relaxed text-muted">
            AUC is the probability that a training row seen during fine-tuning has lower loss than a late-anchor non-member (0.5 = no
            signal). Non-members are later admissions, so part of the AUC is train-to-test shift; the zero-shot control on the same items
            measures that part. Rule (a): CLEAR when AUC minus control is ≤ 0.05, or when the 95% CI of the difference includes 0. Only
            aggregates are shown, and no MIMIC text left the local machine. Full report: <code className="num">loops/osler_rc0/AUDIT.md</code>.
          </p>
        </div>
        <div className="grid gap-4">
          <div className="card p-5">
            <div className="mb-2 flex items-center gap-2 text-[14px] font-semibold text-ink">
              <ShieldOff size={15} className="text-accent" /> Osler-9B: non-generative
            </div>
            <p className="text-[13px] leading-relaxed text-muted">
              The checkpoint has only a LoRA and a <code className="num">[255, 4096]</code> option-code readout. It has no{' '}
              <code className="num">lm_head</code>, so it cannot generate text. A verbatim probe does not apply to it.
            </p>
          </div>
          <div className="card p-5">
            <div className="mb-2 flex items-center gap-2 text-[14px] font-semibold text-ink">
              <ShieldCheck size={15} className="text-accent" /> Osler-4B: verbatim probe
            </div>
            <div className="num text-[1.6rem] font-semibold text-ink">
              {VERBATIM_PROBE_4B.hits}/{VERBATIM_PROBE_4B.notes}
            </div>
            <p className="text-[13px] leading-relaxed text-muted">
              No continuation of at least {VERBATIM_PROBE_4B.minRunTokens} tokens matched any of {VERBATIM_PROBE_4B.notes} train-split MIMIC
              discharge notes (128-token prefix, 64 greedy tokens).
            </p>
          </div>
        </div>
      </div>

      <div>
        <h4 className="mb-4 flex items-center gap-2 text-[15px] font-semibold text-ink">
          <AlertTriangle size={15} className="text-warn" /> Measured limitations
        </h4>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {lim.map((l) => (
            <div key={l.title} className="card border-l-[3px] border-l-warn/60 p-5">
              <div className="mb-2 text-[14px] font-semibold leading-snug text-ink">{l.title}</div>
              <p className="text-[13px] leading-relaxed text-muted">{l.body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function N({ v, d = 4 }: { v: number | null; d?: number }) {
  return <span className="num text-ink">{v === null ? 'NOT MEASURED' : v.toFixed(d)}</span>;
}

// ---------------------------------------------------------------- Shell
export function Analytics({ initialTab = 'calibration' }: { initialTab?: Tab }) {
  const [tab, setTab] = useState<Tab>(initialTab);
  const tabs: { id: Tab; label: string; icon: ReactNode }[] = [
    { id: 'calibration', label: 'Calibration', icon: <Activity size={14} /> },
    { id: 'selective', label: 'Selective Abstention', icon: <ShieldCheck size={14} /> },
    { id: 'cardinality', label: 'Cardinality & Efficiency', icon: <Layers size={14} /> },
    { id: 'privacy', label: 'Privacy Audit & Limitations', icon: <Lock size={14} /> },
  ];
  return (
    <section id="analytics" className="section py-20">
      <SectionHeader
        eyebrow="05 · Deep-dive analytics"
        title={
          <>
            Calibration, abstention, scale <span className="italic text-muted">and what still fails.</span>
          </>
        }
      />
      <div className="card overflow-hidden">
        <div role="tablist" className="flex overflow-x-auto border-b border-line">
          {tabs.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => setTab(t.id)}
              className={`relative flex shrink-0 items-center gap-2 px-5 py-3.5 text-[13px] font-medium transition ${
                tab === t.id ? 'text-ink' : 'text-muted hover:text-ink'
              }`}
            >
              {t.icon}
              {t.label}
              {tab === t.id && <span className="absolute inset-x-4 bottom-0 h-0.5 rounded-full bg-accent" />}
            </button>
          ))}
        </div>
        <div key={tab} className="fade-up p-5 sm:p-7">
          {tab === 'calibration' && <Calibration />}
          {tab === 'selective' && <Selective />}
          {tab === 'cardinality' && <Cardinality />}
          {tab === 'privacy' && <Privacy />}
        </div>
      </div>
    </section>
  );
}

/** Exposed for the SSR smoke test only. */
export const __test = { Calibration, Selective, OptionScaling, REL_SETS, SEL_SETS, OPT_SETS };
