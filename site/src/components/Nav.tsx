import { useEffect, useState } from 'react';
import { Moon, Sun, Menu, X, ArrowLeft } from 'lucide-react';
import { links } from '../lib/data';
import { GithubIcon, HfIcon } from './Icons';
import type { Theme } from '../lib/theme';

const FULL_LINKS = [
  ['overview', 'Overview'],
  ['examples', 'Examples'],
  ['leaderboard', 'Leaderboard'],
  ['analytics', 'Analytics'],
  ['quickstart', 'Quickstart'],
  ['submit', 'Submit'],
] as const;
const LB_LINKS = [
  ['pareto', 'Pareto'],
  ['leaderboard', 'Leaderboard'],
  ['analytics', 'Analytics'],
  ['submit', 'Submit'],
] as const;

export function Brand() {
  return (
    <span className="flex items-center gap-2.5">
      <span className="grid h-7 w-7 place-items-center rounded-lg bg-accent text-canvas">
        <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
          <path d="M8 2.5v11M2.5 8h11" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
        </svg>
      </span>
      <span className="font-serif text-[1.35rem] leading-none tracking-tight">
        Osler <span className="text-muted">·</span> <span className="italic">MedDecide</span>
      </span>
    </span>
  );
}

export function Nav({ theme, onToggle, compact }: { theme: Theme; onToggle: () => void; compact: boolean }) {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 8);
    on();
    window.addEventListener('scroll', on, { passive: true });
    return () => window.removeEventListener('scroll', on);
  }, []);
  const items = compact ? LB_LINKS : FULL_LINKS;
  return (
    <header
      className={`sticky top-0 z-40 border-b backdrop-blur-md transition-colors ${
        scrolled ? 'border-line bg-canvas/85' : 'border-transparent bg-canvas/60'
      }`}
    >
      <nav className="section flex h-16 items-center justify-between gap-4" aria-label="Primary">
        <a href={compact ? links.landing_page : '#top'} className="shrink-0" aria-label="Osler home">
          <Brand />
        </a>
        <div className="hidden items-center gap-1 lg:flex">
          {compact && (
            <a href={links.landing_page} className="mr-2 inline-flex items-center gap-1 rounded-full px-3 py-1.5 text-[13px] font-medium text-accent hover:bg-accent/10">
              <ArrowLeft size={14} /> Full site
            </a>
          )}
          {items.map(([id, label]) => (
            <a key={id} href={`#${id}`} className="rounded-full px-3 py-1.5 text-[13px] font-medium text-muted transition hover:bg-subtle hover:text-ink">
              {label}
            </a>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <button onClick={onToggle} className="grid h-9 w-9 place-items-center rounded-full border border-line bg-card text-muted transition hover:text-ink" aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}>
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a href={links.github} target="_blank" rel="noreferrer" className="hidden h-9 items-center gap-2 rounded-full border border-line bg-card px-3 text-[13px] font-medium text-ink transition hover:border-ink/40 sm:inline-flex">
            <GithubIcon size={15} /> GitHub
          </a>
          <a href={links.hf_collection} target="_blank" rel="noreferrer" className="hidden h-9 items-center gap-2 rounded-full bg-ink px-3 text-[13px] font-medium text-canvas transition hover:opacity-90 sm:inline-flex">
            <HfIcon size={15} /> Hugging Face
          </a>
          <button className="grid h-9 w-9 place-items-center rounded-full border border-line bg-card lg:hidden" onClick={() => setOpen((o) => !o)} aria-label="Toggle menu" aria-expanded={open}>
            {open ? <X size={16} /> : <Menu size={16} />}
          </button>
        </div>
      </nav>
      {open && (
        <div className="section flex flex-col gap-1 border-t border-line pb-4 pt-2 lg:hidden">
          {compact && (
            <a href={links.landing_page} className="rounded-lg px-3 py-2 text-sm font-medium text-accent">← Full site</a>
          )}
          {items.map(([id, label]) => (
            <a key={id} href={`#${id}`} onClick={() => setOpen(false)} className="rounded-lg px-3 py-2 text-sm text-muted hover:bg-subtle hover:text-ink">
              {label}
            </a>
          ))}
          <div className="mt-2 flex gap-2 px-3">
            <a href={links.github} className="btn-ghost !py-1.5 text-xs"><GithubIcon size={14} /> GitHub</a>
            <a href={links.hf_collection} className="btn-ghost !py-1.5 text-xs"><HfIcon size={14} /> Hugging Face</a>
          </div>
        </div>
      )}
    </header>
  );
}
