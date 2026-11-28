// Series colours: CSS variables so light/dark both resolve correctly.
export const C = {
  o9: 'rgb(var(--accent))',
  o4: 'rgb(var(--c-4b))',
  base: 'rgb(var(--c-base))',
  p27: 'rgb(var(--c-27b))',
  other: 'rgb(var(--c-other))',
  warn: 'rgb(var(--warn))',
  ink: 'rgb(var(--ink))',
  muted: 'rgb(var(--muted))',
  line: 'rgb(var(--line))',
  card: 'rgb(var(--card))',
};

const MAP: Record<string, string> = {
  'Osler-9B': C.o9,
  'Osler-9B (raw)': C.o9,
  'Osler-4B': C.o4,
  'Osler-4B (raw)': C.o4,
  'd3-flash': C.base,
  'pplx-flash': 'rgb(var(--c-pf))',
  'pplx-27B': C.p27,
  'MedDecider-9B': 'rgb(var(--c-md))',
};

export function seriesColor(id: string): string {
  return MAP[id] ?? C.other;
}
