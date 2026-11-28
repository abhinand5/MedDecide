import { useState } from 'react';
import { CheckCircle2, FileText, Info } from 'lucide-react';
import { data, type ExampleQuestion } from '../lib/data';
import { C } from '../lib/colors';
import { SectionHeader, TypeBadge } from './ui';

function Bar({ p, color, label, delay = 0 }: { p: number; color: string; label: string; delay?: number }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="w-[4.2rem] shrink-0 font-mono text-[10.5px] text-muted">{label}</span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-subtle">
        <div
          className="bar-grow absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${Math.max(p * 100, 0.6)}%`, background: color, animationDelay: `${delay}ms` }}
        />
      </div>
      <span className="num w-[3.6rem] shrink-0 text-right text-[12px] text-ink">{(p * 100).toFixed(1)}%</span>
    </div>
  );
}

interface Row {
  label: string;
  p9: number;
  p4: number;
  expected: boolean;
}

function rowsOf(q: ExampleQuestion): Row[] {
  if (q.type === 'noul' && q.osler_9b && q.osler_4b) {
    return [
      { label: 'Yes', p9: q.osler_9b.yes, p4: q.osler_4b.yes, expected: q.expected === 'Yes' },
      { label: 'No', p9: q.osler_9b.no, p4: q.osler_4b.no, expected: q.expected === 'No' },
    ];
  }
  const opts = q.options ?? [];
  // Expected answer may be a prefix of the option label (e.g. "Insufficient information").
  return opts.map((o) => ({
    label: o.label,
    p9: o.p9b,
    p4: o.p4b,
    expected: o.label === q.expected || o.label.startsWith(q.expected),
  }));
}

function QuestionCard({ q, idx }: { q: ExampleQuestion; idx: number }) {
  const rows = rowsOf(q);
  return (
    <div className="card p-5">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <div className="mb-1.5 flex items-center gap-2">
            <span className="num text-[11px] text-muted">Q{idx + 1}</span>
            <TypeBadge type={q.type} />
          </div>
          <p className="text-[15px] font-medium leading-snug text-ink">{q.prompt}</p>
        </div>
      </div>
      <div className="space-y-2">
        {rows.map((r, i) => (
          <div
            key={r.label}
            className={`rounded-xl border px-3.5 py-2.5 transition ${
              r.expected ? 'border-accent/40 bg-accent/[0.05]' : 'border-line'
            }`}
          >
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <span className={`text-[13.5px] ${r.expected ? 'font-semibold text-ink' : 'text-muted'}`}>{r.label}</span>
              {r.expected && (
                <span className="badge bg-accent/10 text-accent">
                  <CheckCircle2 size={11} /> expected
                </span>
              )}
            </div>
            <div className="space-y-1">
              <Bar p={r.p9} color={C.o9} label="Osler-9B" delay={i * 60} />
              <Bar p={r.p4} color={C.o4} label="Osler-4B" delay={i * 60 + 30} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Explorer({ initialId }: { initialId?: string }) {
  const examples = data.clinical_examples;
  const [sel, setSel] = useState(initialId ?? examples[0]?.id ?? '');
  const ex = examples.find((e) => e.id === sel) ?? examples[0];
  if (!ex) return null;

  return (
    <section id="examples" className="section py-20">
      <SectionHeader
        eyebrow="02 · System One explorer"
        title={
          <>
            One prefill pass. <span className="italic text-muted">A probability for every option.</span>
          </>
        }
        lede={
          <>
            Each question is typed — <code className="num text-ink">noul</code> returns p(yes), <code className="num text-ink">choice</code>{' '}
            returns a distribution over the offered options. Bars show the calibrated outputs of both released checkpoints.
          </>
        }
      />

      <div className="mb-6 flex gap-2 overflow-x-auto pb-1 [scrollbar-width:thin] lg:grid lg:grid-cols-3 lg:overflow-visible">
        {examples.map((e, i) => (
          <button
            key={e.id}
            onClick={() => setSel(e.id)}
            className={`shrink-0 rounded-xl border px-3.5 py-2 text-left transition ${
              e.id === ex.id ? 'border-ink bg-ink text-canvas' : 'border-line bg-card text-muted hover:border-ink/30 hover:text-ink'
            }`}
          >
            <span className="block font-mono text-[10px] uppercase tracking-wider opacity-70">
              {String(i + 1).padStart(2, '0')} · {e.tag}
            </span>
            <span className="block truncate text-[13px] font-medium">{e.title}</span>
          </button>
        ))}
      </div>

      <div key={ex.id} className="fade-up grid gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        <div className="card flex flex-col self-start p-6 lg:sticky lg:top-24">
          <div className="mb-4 flex items-center gap-2 text-muted">
            <FileText size={15} />
            <span className="eyebrow">Clinical state</span>
          </div>
          <h3 className="h-display mb-4 text-[1.6rem] leading-tight">{ex.title}</h3>
          <div className="space-y-3 border-l-2 border-accent/40 pl-4 font-mono text-[12.5px] leading-[1.75] text-ink/90">
            {ex.state.split('\n').map((para, i) => (
              <p key={i}>{para}</p>
            ))}
          </div>
          <div className="flex items-start gap-2 pt-6 text-[12px] leading-snug text-muted">
            <Info size={13} className="mt-0.5 shrink-0" />
            <span>
              Synthetic vignette written for illustration — not a benchmark item. Outputs measured from the standalone merged checkpoints, one
              question per call, built-in dev calibration.
            </span>
          </div>
        </div>
        <div className="grid gap-4">
          {ex.questions.map((q, i) => (
            <QuestionCard key={q.id} q={q} idx={i} />
          ))}
        </div>
      </div>
    </section>
  );
}
