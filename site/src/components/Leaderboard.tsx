import { Fragment, useMemo, useState } from 'react';
import { ArrowDown, ArrowUp, ChevronDown, ExternalLink, Search, Send } from 'lucide-react';
import {
  ADDITIONAL_SETS,
  GENERAL_SETS,
  coverageOf,
  EHR_SETS,
  HEADLINE_SETS,
  SET_LABEL,
  cell,
  evalSet,
  fmt,
  isMeasured,
  links,
  metric,
  missingReason,
  models,
  readoutFail,
  sizeOk,
  type Model,
  type SetId,
} from '../lib/data';
import { Pills, ParetoBadge, ReadoutFailMark, SectionHeader, TierBadge } from './ui';

export type View = 'all' | 'full' | 'public';
type Size = 'all' | 'le10' | 'le4';
export type Metric = 'accuracy' | 'brier' | 'ece' | 'latency';

interface Col {
  key: string;
  label: string;
  sub?: string;
  get: (m: Model) => number | null;
  higherBetter: boolean;
  digits?: number;
  set?: SetId;
  unit?: string;
  /** For composite averages published in the JSON: the component sets, to flag partial coverage. */
  parts?: SetId[];
}

function setCols(sets: SetId[], key: 'accuracy' | 'brier' | 'ece'): Col[] {
  return sets.map((s) => ({
    key: `${s}.${key}`,
    label: SET_LABEL[s],
    sub: s,
    set: s,
    get: (m: Model) => metric(m, s, key),
    higherBetter: key === 'accuracy',
  }));
}

function avg(vals: (number | null)[]): number | null {
  const v = vals.filter((x): x is number => x !== null);
  return v.length === vals.length && v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
}

function columnsFor(metricKey: Metric, view: View, extra: boolean): Col[] {
  const sets: SetId[] = view === 'public' ? ['confirm_general'] : [...HEADLINE_SETS, ...(extra ? ADDITIONAL_SETS : [])];
  if (metricKey === 'latency') {
    return [
      { key: 'params', label: 'Params', unit: 'B', get: (m) => m.params_b, higherBetter: false, digits: 2 },
      { key: 'p50', label: 'p50 latency', sub: 'batch 1', unit: 'ms', get: (m) => m.p50_ms_b1, higherBetter: false, digits: 2 },
      { key: 'vram', label: 'Peak VRAM', unit: 'GB', get: (m) => m.peak_vram_gb, higherBetter: false, digits: 2 },
      { key: 'macro5', label: '5-set macro', get: (m) => m.five_set_macro, higherBetter: true },
    ];
  }
  if (metricKey === 'accuracy') {
    const lead: Col[] =
      view === 'public'
        ? [
            {
              key: 'cg.macro',
              label: 'Template macro',
              sub: 'confirm_general',
              get: (m) => metric(m, 'confirm_general', 'macro'),
              higherBetter: true,
            },
          ]
        : [
            { key: 'macro5', label: '5-set macro', get: (m) => m.five_set_macro, higherBetter: true },
            { key: 'gen', label: 'General QA', sub: 'mean of 3', get: (m) => m.general_medical_avg, higherBetter: true, parts: GENERAL_SETS },
            { key: 'ehr', label: 'EHR & Docs', sub: 'mean of 2', get: (m) => m.ehr_docs_avg, higherBetter: true, parts: EHR_SETS },
          ];
    return [...lead, ...setCols(sets, 'accuracy')];
  }
  const k = metricKey;
  const mean: Col = {
    key: `mean.${k}`,
    label: `Mean ${k === 'brier' ? 'Brier' : 'ECE'}`,
    sub: view === 'public' ? 'confirm_general' : '5 headline sets',
    get: (m) => (view === 'public' ? metric(m, 'confirm_general', k) : avg(HEADLINE_SETS.map((s) => metric(m, s, k)))),
    higherBetter: false,
  };
  return view === 'public' ? setCols(sets, k) : [mean, ...setCols(sets, k)];
}

function CellView({ m, col, best }: { m: Model; col: Col; best: boolean }) {
  const v = col.get(m);
  if (v === null) {
    if (col.set && evalSet(col.set)?.is_public_tier === false && !isMeasured(cell(m, col.set))) {
      return (
        <a
          href={links.private_eval_issue}
          target="_blank"
          rel="noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1 rounded-md border border-dashed border-accent/40 px-1.5 py-0.5 text-[10.5px] text-accent hover:bg-accent/5"
          title={`${missingReason(m, col.set)} — private EHR set; request a local evaluation`}
        >
          <Send size={10} /> Request Eval
        </a>
      );
    }
    const reason = col.set ? missingReason(m, col.set) : 'NOT MEASURED';
    return (
      <span className="text-[11px] text-muted/70" title={reason}>
        N/M
      </span>
    );
  }
  const fail = col.set ? readoutFail(cell(m, col.set)) : null;
  return (
    <span className={`num ${best ? 'font-bold text-ink' : 'text-ink/80'}`}>
      {v.toFixed(col.digits ?? 4)}
      {fail && <ReadoutFailMark reason={fail} />}
      {col.parts && coverageOf(m, col.parts) < col.parts.length && (
        <sup className="ml-0.5 text-[9px] font-normal text-warn" title="Partial: averages only the measured component sets">
          {coverageOf(m, col.parts)}/{col.parts.length}
        </sup>
      )}
    </span>
  );
}

function Detail({ m }: { m: Model }) {
  const sets = [...HEADLINE_SETS, ...ADDITIONAL_SETS];
  return (
    <div className="grid gap-6 px-4 py-5 sm:px-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-[12px]">
          <thead>
            <tr className="text-left text-muted">
              <th className="pb-2 font-medium">Set</th>
              <th className="pb-2 text-right font-medium">Accuracy</th>
              <th className="pb-2 text-right font-medium">95% CI</th>
              <th className="pb-2 text-right font-medium">Brier</th>
              <th className="pb-2 text-right font-medium">ECE</th>
              <th className="pb-2 text-right font-medium">Scored / n</th>
              <th className="pb-2 pl-3 font-medium">Readout health</th>
            </tr>
          </thead>
          <tbody>
            {sets.map((s) => {
              const c = cell(m, s);
              const meas = isMeasured(c);
              return (
                <tr key={s} className="border-t border-line/70">
                  <td className="py-1.5 pr-2">
                    <span className="text-ink">{SET_LABEL[s]}</span>{' '}
                    <span className="font-mono text-[10.5px] text-muted">{s}</span>
                  </td>
                  {meas ? (
                    <>
                      <td className="num py-1.5 text-right">{fmt(c.accuracy)}</td>
                      <td className="num whitespace-nowrap py-1.5 pl-3 text-right text-muted">
                        [{fmt(c.ci_lo)}, {fmt(c.ci_hi)}]
                      </td>
                      <td className="num py-1.5 pl-3 text-right">{fmt(c.brier)}</td>
                      <td className="num py-1.5 pl-3 text-right">{fmt(c.ece)}</td>
                      <td className="num whitespace-nowrap py-1.5 pl-3 text-right text-muted">
                        {c.scored?.toLocaleString()} / {c.n?.toLocaleString()}
                      </td>
                      <td className={`py-1.5 pl-3 font-mono text-[10.5px] ${readoutFail(c) ? 'text-warn' : 'text-accent'}`}>
                        {c.readout_health ?? '—'}
                      </td>
                    </>
                  ) : (
                    <td colSpan={6} className="py-1.5 text-right font-mono text-[10.5px] text-muted">
                      {missingReason(m, s)}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <dl className="grid content-start grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[12.5px]">
        <dt className="text-muted">Role</dt>
        <dd className="text-ink">{m.role}</dd>
        <dt className="text-muted">Base model</dt>
        <dd className="break-all font-mono text-[11.5px] text-ink">{m.base_repo}</dd>
        <dt className="text-muted">Readout</dt>
        <dd className="font-mono text-[11.5px] text-ink">{m.readout_type}</dd>
        <dt className="text-muted">Non-generative</dt>
        <dd className="text-ink">{m.non_generative ? 'Yes — no lm_head' : 'No'}</dd>
        <dt className="text-muted">Params</dt>
        <dd className="num text-ink">{m.params_b.toFixed(2)}B</dd>
        <dt className="text-muted">License</dt>
        <dd className="text-ink">{m.license}</dd>
        <dt className="text-muted">Weights</dt>
        <dd>
          <a href={m.hf_url} target="_blank" rel="noreferrer" className="link inline-flex items-center gap-1 break-all font-mono text-[11.5px]">
            {m.hf_repo} <ExternalLink size={11} />
          </a>
        </dd>
      </dl>
    </div>
  );
}

export function Leaderboard({
  initial = {},
}: {
  initial?: { view?: View; size?: Size; metric?: Metric; extra?: boolean; open?: string };
}) {
  const [view, setView] = useState<View>(initial.view ?? 'all');
  const [size, setSize] = useState<Size>(initial.size ?? 'all');
  const [q, setQ] = useState('');
  const [metricKey, setMetric] = useState<Metric>(initial.metric ?? 'accuracy');
  const [extra, setExtra] = useState(initial.extra ?? false);
  const [sort, setSort] = useState<{ key: string; dir: 'asc' | 'desc' } | null>(null);
  const [open, setOpen] = useState<string | null>(initial.open ?? null);

  const cols = columnsFor(metricKey, view, extra);
  const defaultCol = cols[0];
  const activeSort = sort && cols.some((c) => c.key === sort.key) ? sort : { key: defaultCol.key, dir: defaultCol.higherBetter ? 'desc' : 'asc' } as const;
  const sortCol = cols.find((c) => c.key === activeSort.key) ?? defaultCol;

  const rows = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const filtered = models.filter((m) => {
      if (view === 'full' && m.verification_tier !== 'verified_full') return false;
      if (view === 'public' && !isMeasured(cell(m, 'confirm_general')) && m.verification_tier !== 'public_only') return false;
      if (!sizeOk(m, size)) return false;
      if (ql && !`${m.id} ${m.hf_repo} ${m.role}`.toLowerCase().includes(ql)) return false;
      return true;
    });
    return filtered.sort((a, b) => {
      const va = sortCol.get(a);
      const vb = sortCol.get(b);
      if (va === null && vb === null) return a.id.localeCompare(b.id);
      if (va === null) return 1; // nulls always last, never imputed
      if (vb === null) return -1;
      return activeSort.dir === 'desc' ? vb - va : va - vb;
    });
  }, [view, size, q, sortCol, activeSort.dir]);

  const best: Record<string, number | null> = {};
  for (const c of cols) {
    const vals = rows
      .filter((m) => !c.parts || coverageOf(m, c.parts) === c.parts.length)
      .map((m) => c.get(m))
      .filter((v): v is number => v !== null);
    best[c.key] = vals.length ? (c.higherBetter ? Math.max(...vals) : Math.min(...vals)) : null;
  }

  const clickSort = (c: Col) =>
    setSort((s) =>
      s && s.key === c.key ? { key: c.key, dir: s.dir === 'desc' ? 'asc' : 'desc' } : { key: c.key, dir: c.higherBetter ? 'desc' : 'asc' },
    );

  const counts = {
    all: models.length,
    full: models.filter((m) => m.verification_tier === 'verified_full').length,
    public: models.filter((m) => isMeasured(cell(m, 'confirm_general')) || m.verification_tier === 'public_only').length,
  };
  const cg = evalSet('confirm_general');

  return (
    <section id="leaderboard" className="section py-20">
      <SectionHeader
        eyebrow="04 · Unified two-tier leaderboard"
        title={
          <>
            Twelve models, nine frozen sets. <span className="italic text-muted">Nothing imputed.</span>
          </>
        }
        lede={
          <>
            The public tier is <code className="num">confirm_general</code> ({cg?.n_items.toLocaleString()} post-cutoff structured-gold items,
            anyone can submit). The private tier adds the MIMIC-IV and medical-record suites, run locally under the PhysioNet DUA. Click a row for
            95% CIs, coverage and readout health.
          </>
        }
      />

      <div className="card overflow-hidden">
        <div className="flex flex-col gap-3 border-b border-line p-4 sm:p-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <Pills
              label="View"
              value={view}
              onChange={(v) => {
                setView(v);
                setSort(null);
              }}
              options={[
                { value: 'all', label: `All Models (${counts.all})` },
                { value: 'full', label: `Verified Full Suite (${counts.full})` },
                { value: 'public', label: `Public Tier · confirm_general (${counts.public})` },
              ]}
            />
            <label className="relative block lg:w-64">
              <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
              <input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                placeholder="Search models…"
                aria-label="Search models"
                className="w-full rounded-full border border-line bg-canvas py-1.5 pl-8 pr-3 text-[13px] text-ink outline-none placeholder:text-muted/70 focus:border-accent/60"
              />
            </label>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-3">
              <Pills
                size="sm"
                label="Metric"
                value={metricKey}
                onChange={(v) => {
                  setMetric(v);
                  setSort(null);
                }}
                options={[
                  { value: 'accuracy', label: 'Accuracy' },
                  { value: 'brier', label: 'Brier (↓)' },
                  { value: 'ece', label: 'ECE (↓)' },
                  { value: 'latency', label: 'Latency & VRAM' },
                ]}
              />
              <span className="hidden h-5 w-px bg-line sm:block" />
              <Pills
                size="sm"
                label="Size"
                value={size}
                onChange={setSize}
                options={[
                  { value: 'all', label: 'All sizes' },
                  { value: 'le10', label: '≤10B' },
                  { value: 'le4', label: '≤4B' },
                ]}
              />
            </div>
            {view !== 'public' && metricKey !== 'latency' && (
              <label className="flex cursor-pointer items-center gap-2 text-[12.5px] text-muted">
                <input type="checkbox" checked={extra} onChange={(e) => setExtra(e.target.checked)} className="accent-[rgb(var(--accent))]" />
                Show 4 additional sets
              </label>
            )}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className={`w-full border-collapse text-[13px] ${extra && view !== 'public' && metricKey !== 'latency' ? 'min-w-[1380px]' : 'min-w-[980px]'}`}>
            <thead>
              <tr className="border-b border-line bg-subtle/60 text-left">
                <th className="w-10 px-4 py-3 font-mono text-[10.5px] font-medium uppercase tracking-wider text-muted">#</th>
                <th className="min-w-[240px] px-2 py-3 font-mono text-[10.5px] font-medium uppercase tracking-wider text-muted">Model</th>
                {cols.map((c) => {
                  const active = c.key === activeSort.key;
                  return (
                    <th key={c.key} className="whitespace-nowrap px-3 py-2.5 text-right align-bottom">
                      <button
                        onClick={() => clickSort(c)}
                        className={`inline-flex flex-col items-end gap-0.5 text-right transition ${active ? 'text-ink' : 'text-muted hover:text-ink'}`}
                        aria-label={`Sort by ${c.label}`}
                      >
                        <span className="inline-flex items-center gap-1 text-[12px] font-semibold">
                          {c.label}
                          {active ? (activeSort.dir === 'desc' ? <ArrowDown size={12} /> : <ArrowUp size={12} />) : null}
                        </span>
                        <span className="font-mono text-[10px] font-normal opacity-70">
                          {c.sub ?? (c.unit ? c.unit : '')}
                          {!c.higherBetter && metricKey !== 'latency' ? ' ↓' : ''}
                        </span>
                      </button>
                    </th>
                  );
                })}
                <th className="w-8" />
              </tr>
            </thead>
            <tbody>
              {rows.map((m, i) => {
                const isOpen = open === m.id;
                return (
                  <Fragment key={m.id}>
                    <tr
                      onClick={() => setOpen(isOpen ? null : m.id)}
                      className={`cursor-pointer border-b border-line/70 transition-colors hover:bg-subtle/60 ${
                        m.is_osler ? 'bg-accent/[0.035]' : ''
                      } ${isOpen ? 'bg-subtle/60' : ''}`}
                      aria-expanded={isOpen}
                    >
                      <td className="num px-4 py-3 text-[12px] text-muted">{sortCol.get(m) === null ? '—' : i + 1}</td>
                      <td className="px-2 py-3">
                        <div className="flex flex-wrap items-center gap-1.5">
                          <span className={`font-medium ${m.is_osler ? 'text-accent' : 'text-ink'}`}>{m.id}</span>
                          {m.on_pareto_front && <ParetoBadge />}
                          <TierBadge tier={m.verification_tier} />
                        </div>
                        <div className="mt-0.5 max-w-[260px] truncate font-mono text-[10.5px] text-muted" title={m.role}>
                          {m.params_b.toFixed(2)}B · {m.role}
                        </div>
                      </td>
                      {cols.map((c) => (
                        <td key={c.key} className="whitespace-nowrap px-3 py-3 text-right">
                          <CellView m={m} col={c} best={best[c.key] !== null && c.get(m) === best[c.key] && (!c.parts || coverageOf(m, c.parts) === c.parts.length)} />
                        </td>
                      ))}
                      <td className="pr-3 text-muted">
                        <ChevronDown size={15} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                      </td>
                    </tr>
                    {isOpen && (
                      <tr className="border-b border-line bg-canvas/60">
                        <td colSpan={cols.length + 3}>
                          <Detail m={m} />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
              {rows.length === 0 && (
                <tr>
                  <td colSpan={cols.length + 3} className="px-4 py-10 text-center text-muted">
                    No models match these filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <div className="flex flex-col gap-1.5 border-t border-line px-4 py-3 text-[11.5px] leading-relaxed text-muted sm:px-5">
          <span>
            <strong className="font-semibold text-ink">Bold</strong> = best among visible rows. <span className="text-warn">⚠</span> ={' '}
            <code className="num">READOUT_FAIL</code> on at least one template (hover for detail). <span className="num">N/M</span> = not measured on a public set; <span className="text-accent">Request Eval</span> = not measured on a private
            (PhysioNet) set (hover for reason). Composite columns are blank unless every component set is measured
            {metricKey === 'accuracy' && view !== 'public' ? '; General QA averages marked 2/3 cover only the measured sets' : ''}.
          </span>
          {metricKey === 'latency' && (
            <span>Latency: batch-1 p50 per item and peak VRAM from the RC0 latency run (R5); models without a measurement within its timebox are NOT MEASURED.</span>
          )}
        </div>
      </div>
    </section>
  );
}
