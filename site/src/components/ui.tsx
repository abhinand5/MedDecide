import { useState, type ReactNode } from 'react';
import { Check, Copy, Star, ShieldCheck, CircleDashed, Globe, AlertTriangle } from 'lucide-react';
import { TIER_LABEL } from '../lib/data';

export function SectionHeader({
  id,
  eyebrow,
  title,
  lede,
  right,
}: {
  id?: string;
  eyebrow: string;
  title: ReactNode;
  lede?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div id={id} className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
      <div className="max-w-3xl">
        <div className="eyebrow mb-3">{eyebrow}</div>
        <h2 className="h-display text-[2.15rem] leading-[1.08] sm:text-[2.75rem]">{title}</h2>
        {lede && <p className="mt-3 text-[15px] leading-relaxed text-muted">{lede}</p>}
      </div>
      {right}
    </div>
  );
}

export function Pills<T extends string>({
  options,
  value,
  onChange,
  size = 'md',
  label,
}: {
  options: { value: T; label: ReactNode }[];
  value: T;
  onChange: (v: T) => void;
  size?: 'sm' | 'md';
  label?: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="flex flex-wrap gap-1.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={o.value === value}
          onClick={() => onChange(o.value)}
          className={`pill ${size === 'sm' ? '!px-2.5 !py-1 !text-xs' : ''} ${o.value === value ? 'pill-on' : ''}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function CopyButton({ text, className = '' }: { text: string; className?: string }) {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setDone(true);
    window.setTimeout(() => setDone(false), 1600);
  };
  return (
    <button
      onClick={copy}
      aria-label="Copy to clipboard"
      className={`inline-flex items-center gap-1.5 rounded-md border border-white/10 bg-white/5 px-2.5 py-1 font-mono text-[11px] text-slate-300 transition hover:bg-white/10 hover:text-white ${className}`}
    >
      {done ? <Check size={13} /> : <Copy size={13} />}
      {done ? 'Copied' : 'Copy'}
    </button>
  );
}

export function CodeBlock({ code, lang }: { code: string; lang: string }) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-800 bg-[#0B1120]">
      <div className="flex items-center justify-between border-b border-white/5 px-4 py-2">
        <span className="font-mono text-[11px] uppercase tracking-wider text-slate-500">{lang}</span>
        <CopyButton text={code} />
      </div>
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[12.5px] leading-[1.7] text-slate-200">
        <code>{code}</code>
      </pre>
    </div>
  );
}

export function TierBadge({ tier }: { tier: string }) {
  const label = TIER_LABEL[tier] ?? tier;
  if (tier === 'verified_full')
    return (
      <span className="badge bg-accent/10 text-accent" title="Measured on all 9 frozen sets locally">
        <ShieldCheck size={11} /> {label}
      </span>
    );
  if (tier === 'public_only')
    return (
      <span className="badge bg-subtle text-muted" title="Public tier (confirm_general) only">
        <Globe size={11} /> {label}
      </span>
    );
  return (
    <span className="badge bg-warn/10 text-warn" title="Some sets NOT MEASURED">
      <CircleDashed size={11} /> {label}
    </span>
  );
}

export function ParetoBadge() {
  return (
    <span className="badge bg-ink text-canvas" title="On the global 2-axis Pareto front">
      <Star size={10} fill="currentColor" /> Pareto
    </span>
  );
}

export function ReadoutFailMark({ reason }: { reason: string }) {
  return (
    <span className="ml-1 inline-flex align-middle text-warn" title={reason} aria-label={reason}>
      <AlertTriangle size={12} />
    </span>
  );
}

export function TypeBadge({ type }: { type: string }) {
  return (
    <span className="badge border border-line bg-subtle font-mono uppercase text-muted">{type}</span>
  );
}
