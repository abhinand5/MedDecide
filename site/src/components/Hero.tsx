import type { ReactNode } from 'react';
import { ArrowRight, Database, Gauge, ShieldCheck, Target, Trophy } from 'lucide-react';
import { data, derived, fmt, links, metric, phiAuditAllClear } from '../lib/data';
import { C } from '../lib/colors';
import { HfIcon } from './Icons';

function Kpi({
  icon,
  value,
  label,
  sub,
}: {
  icon: ReactNode;
  value: ReactNode;
  label: string;
  sub: string;
}) {
  return (
    <div className="card group relative flex flex-col gap-3 p-5 transition hover:-translate-y-0.5 hover:shadow-[0_8px_30px_-12px_rgb(15_23_42/0.18)]">
      <div className="flex items-center justify-between text-muted">
        <span className="eyebrow">{label}</span>
        <span className="text-accent">{icon}</span>
      </div>
      <div className="num whitespace-nowrap text-[1.5rem] font-semibold leading-none tracking-tight text-ink">{value}</div>
      <p className="text-[13px] leading-snug text-muted">{sub}</p>
    </div>
  );
}

/** A real recorded output (clinical_examples[0]) rendered as the hero illustration. */
function DecisionCard() {
  const ex = data.clinical_examples[0];
  const q = ex?.questions.find((x) => x.type === 'choice') ?? ex?.questions[0];
  if (!ex || !q) return null;
  const opts = q.options ?? [];
  return (
    <div className="fade-up relative hidden lg:block">
      <div className="absolute -inset-3 -z-10 rounded-[28px] bg-accent/[0.06] blur-2xl" />
      <div className="card overflow-hidden shadow-[0_24px_60px_-30px_rgb(15_23_42/0.35)]">
        <div className="flex items-center justify-between border-b border-line px-5 py-3">
          <span className="font-mono text-[11px] text-muted">model.system_one(state, questions)</span>
          <span className="flex gap-1">
            <span className="h-2 w-2 rounded-full bg-line" />
            <span className="h-2 w-2 rounded-full bg-line" />
            <span className="h-2 w-2 rounded-full bg-line" />
          </span>
        </div>
        <div className="space-y-4 p-5">
          <div>
            <div className="eyebrow mb-1.5">state</div>
            <p className="line-clamp-3 font-mono text-[11.5px] leading-relaxed text-ink/80">{ex.state}</p>
          </div>
          <div>
            <div className="eyebrow mb-1.5">question · {q.type}</div>
            <p className="text-[13.5px] font-medium text-ink">{q.prompt}</p>
          </div>
          <div className="space-y-2.5">
            {opts.map((o) => (
              <div key={o.key}>
                <div className="mb-1 flex justify-between text-[12px]">
                  <span className={o.label === q.expected ? 'font-medium text-ink' : 'text-muted'}>{o.label}</span>
                  <span className="num text-ink">{o.p9b.toFixed(3)}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-subtle">
                  <div className="bar-grow h-full rounded-full" style={{ width: `${Math.max(o.p9b * 100, 0.8)}%`, background: C.o9 }} />
                </div>
              </div>
            ))}
          </div>
          <div className="flex items-center justify-between border-t border-line pt-3 font-mono text-[10.5px] text-muted">
            <span>Osler-9B · one prefill · Σp = 1</span>
            <a href="#examples" className="text-accent hover:underline">6 vignettes →</a>
          </div>
        </div>
      </div>
    </div>
  );
}

export function Hero() {
  const { o9, o4, p27, macroRank, macroRankedCount, panelEceRank, panelEceCount, speedup, vramRatio, freshRank } = derived;
  const totalModels = 12;
  const ehrGain = o9?.ehr_docs_avg != null && p27?.ehr_docs_avg != null ? (o9.ehr_docs_avg - p27.ehr_docs_avg) * 100 : null;

  return (
    <section id="overview" className="relative overflow-hidden">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 opacity-[0.5] [background-image:linear-gradient(rgb(var(--line))_1px,transparent_1px),linear-gradient(90deg,rgb(var(--line))_1px,transparent_1px)] [background-size:56px_56px] [mask-image:radial-gradient(ellipse_70%_60%_at_50%_0%,black,transparent)]"
      />
      <div className="section relative pb-16 pt-14 sm:pt-20">
        <div className="fade-up inline-flex flex-wrap items-center gap-2 rounded-full border border-line bg-card px-3 py-1.5 font-mono text-[11px] text-muted">
          <span className="h-1.5 w-1.5 rounded-full bg-accent2" />
          Apache-2.0 · Standalone Merged Weights · Single-Pass Calibrated Probabilities
        </div>

        <div className="grid items-center gap-12 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        <div>
        <h1 className="h-display fade-up mt-7 max-w-4xl text-[2.9rem] leading-[1.02] sm:text-[4.4rem]">
          Medical decisions,
          <br />
          <span className="italic text-accent">read once,</span> calibrated.
        </h1>
        <p className="fade-up mt-6 max-w-2xl text-[17px] leading-relaxed text-muted">
          <strong className="font-semibold text-ink">Osler-9B</strong> and <strong className="font-semibold text-ink">Osler-4B</strong> are open
          general medical decision models that return a calibrated probability for every allowed option in one prefill pass, with no text
          generation. Both read EHR notes and clinical documentation more accurately than a 27B decider
          {ehrGain !== null && (
            <>
              {' '}
              (<span className="num text-ink">+{ehrGain.toFixed(1)} pts</span> for the 9B)
            </>
          )}
          , while keeping general medical QA in range; the 9B also ranks{' '}
          <span className="num text-ink">#{freshRank.rank}</span> of {freshRank.of} on fresh post-cutoff decisions.
        </p>

        <div className="fade-up mt-8 flex flex-wrap gap-2.5">
          <a href={links.hf_osler_9b} target="_blank" rel="noreferrer" className="btn-primary">
            <HfIcon size={15} /> Osler-9B on HF
          </a>
          <a href={links.hf_osler_4b} target="_blank" rel="noreferrer" className="btn-ghost">
            <HfIcon size={15} /> Osler-4B on HF
          </a>
          <a href="#leaderboard" className="btn-ghost">
            Explore Leaderboard <ArrowRight size={15} />
          </a>
          <a href={links.hf_bench} target="_blank" rel="noreferrer" className="btn-ghost">
            <Database size={15} /> MedDecide-Bench
          </a>
        </div>
        </div>
        <DecisionCard />
        </div>

        <div className="mt-12 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <Kpi
            icon={<Trophy size={16} />}
            label="5-Set Macro · 9B"
            value={fmt(o9?.five_set_macro)}
            sub={`#${macroRank} of ${totalModels} evaluated (${macroRankedCount} with all five headline sets measured).`}
          />
          <Kpi
            icon={<Gauge size={16} />}
            label="Batch-1 p50 · VRAM"
            value={
              <>
                {o9?.p50_ms_b1?.toFixed(1)}
                <span className="text-[0.95rem] font-medium text-muted"> ms</span>
                <span className="font-normal text-muted"> / </span>
                {o9?.peak_vram_gb?.toFixed(1)}
                <span className="text-[0.95rem] font-medium text-muted"> GB</span>
              </>
            }
            sub={`${speedup?.toFixed(1)}× faster and ${vramRatio?.toFixed(1)}× less peak VRAM than pplx-27B (${p27?.p50_ms_b1?.toFixed(1)} ms / ${p27?.peak_vram_gb?.toFixed(1)} GB).`}
          />
          <Kpi
            icon={<Target size={16} />}
            label="Panel ECE · 9B"
            value={fmt(o9 ? metric(o9, 'panel', 'ece') : null)}
            sub={`Best calibrated of ${panelEceCount} on the 7-dataset external panel (rank #${panelEceRank}; lower is better).`}
          />
          <Kpi
            icon={<ShieldCheck size={16} />}
            label="MIMIC PHI audit"
            value={phiAuditAllClear ? 'CLEAR' : 'NOT CLEAR'}
            sub={`Local D28 memorisation audit, both models. Osler-9B is non-generative (no lm_head)${o4 && !o4.non_generative ? '; the 4B passed a verbatim probe' : ''}.`}
          />
        </div>
      </div>
    </section>
  );
}
