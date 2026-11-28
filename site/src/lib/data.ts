// Typed access layer over the generated leaderboard JSON.
// Every number on the site comes from this file (or is a commented constant
// pointing at the committed audit report). Missing cells are never imputed.
import raw from '../../data/leaderboard.json';

export type SetId =
  | 'confirm_general'
  | 'knowledge'
  | 'mimic_suite'
  | 'mr_verdict'
  | 'panel'
  | 'mednli_test'
  | 'mimic_heldout'
  | 'mr_heldout_test'
  | 'pairs_test';

export interface EvalSet {
  id: SetId;
  role: string;
  is_headline: boolean;
  is_public_tier: boolean;
  n_items: number;
  n_templates: number;
  sha256_prefix: string;
  description: string;
}

export interface SetCell {
  status: string;
  n?: number | null;
  scored?: number | null;
  coverage?: number | null;
  accuracy?: number | null;
  ci_lo?: number | null;
  ci_hi?: number | null;
  macro?: number | null;
  brier?: number | null;
  ece?: number | null;
  readout_health?: string | null;
}

export type VerificationTier = 'verified_full' | 'verified_partial' | 'public_only';

export interface Model {
  id: string;
  raw_id: string;
  is_osler: boolean;
  verification_tier: VerificationTier | string;
  params_b: number;
  size_tier: string;
  role: string;
  base_repo: string;
  hf_repo: string;
  hf_url: string;
  readout_type: string;
  non_generative: boolean;
  license: string;
  p50_ms_b1: number | null;
  peak_vram_gb: number | null;
  general_medical_avg: number | null;
  ehr_docs_avg: number | null;
  five_set_macro: number | null;
  on_pareto_front: boolean;
  undominated_le10b: boolean;
  undominated_le4b: boolean;
  sets: Partial<Record<SetId, SetCell>>;
}

export interface ReliabilityBin {
  bin: number;
  mid: number;
  n: number;
  conf: number;
  acc: number;
}

export interface RiskPoint {
  coverage: number;
  accuracy: number;
  error: number;
  min_conf: number;
}

export interface BucketScore {
  accuracy: number;
  brier: number;
}

export interface OptionBucket {
  n: number;
  models: Record<string, BucketScore | undefined>;
}

export interface PhiAuditRow {
  model: string;
  auc: number;
  ci_lo: number;
  ci_hi: number;
  control: string;
  control_auc: number;
  delta_auc: number;
  clear: boolean;
}

export interface ChoiceOption {
  key: string;
  label: string;
  p9b: number;
  p4b: number;
}

export interface ExampleQuestion {
  id: string;
  type: 'noul' | 'choice' | 'score' | string;
  prompt: string;
  expected: string;
  options?: ChoiceOption[];
  osler_9b?: { yes: number; no: number };
  osler_4b?: { yes: number; no: number };
}

export interface ClinicalExample {
  id: string;
  tag: string;
  title: string;
  state: string;
  questions: ExampleQuestion[];
}

export interface Links {
  github: string;
  landing_page: string;
  hf_osler_9b: string;
  hf_osler_4b: string;
  hf_bench: string;
  hf_space: string;
  hf_collection: string;
  private_eval_issue: string;
}

export interface LeaderboardData {
  generated_utc: string;
  links: Links;
  eval_sets: EvalSet[];
  models: Model[];
  per_source: Record<string, Record<string, { n: number; models: Record<string, unknown> }>>;
  reliability: Record<string, Record<string, ReliabilityBin[] | undefined>>;
  selective_prediction: Record<string, Record<string, RiskPoint[] | undefined>>;
  option_scaling: Record<string, Record<string, OptionBucket | undefined>>;
  phi_audit: PhiAuditRow[];
  clinical_examples: ClinicalExample[];
}

export const data = raw as unknown as LeaderboardData;
export const links = data.links;
export const models = data.models;
export const evalSets = data.eval_sets;

export const HEADLINE_SETS: SetId[] = ['confirm_general', 'knowledge', 'panel', 'mr_verdict', 'mimic_suite'];
export const GENERAL_SETS: SetId[] = ['confirm_general', 'knowledge', 'panel'];
export const EHR_SETS: SetId[] = ['mr_verdict', 'mimic_suite'];
export const ADDITIONAL_SETS: SetId[] = ['mednli_test', 'pairs_test', 'mimic_heldout', 'mr_heldout_test'];

export const SET_LABEL: Record<SetId, string> = {
  confirm_general: 'Fresh General',
  knowledge: 'Knowledge',
  panel: 'Ext. Panel',
  mr_verdict: 'MR Verdict',
  mimic_suite: 'MIMIC Suite',
  mednli_test: 'MedNLI',
  pairs_test: 'Min. Pairs',
  mimic_heldout: 'MIMIC Held-out',
  mr_heldout_test: 'MR Held-out',
};

/** Source: loops/osler_rc0/AUDIT.md "Verbatim extraction probe (Osler-4B-rc0)" —
 *  matches >= 50 tokens: 0 of 300 train-split MIMIC notes. Not present in leaderboard.json. */
export const VERBATIM_PROBE_4B = { hits: 0, notes: 300, minRunTokens: 50 } as const;

export function getModel(id: string): Model | undefined {
  return models.find((m) => m.id === id);
}

export function evalSet(id: SetId): EvalSet | undefined {
  return evalSets.find((e) => e.id === id);
}

/** A cell is measured only if its status says so and the metric is a number. */
export function cell(m: Model, set: SetId): SetCell | undefined {
  return m.sets[set];
}

export function isMeasured(c: SetCell | undefined): c is SetCell {
  return !!c && c.status === 'MEASURED';
}

export function metric(m: Model, set: SetId, key: 'accuracy' | 'brier' | 'ece' | 'macro'): number | null {
  const c = m.sets[set];
  if (!isMeasured(c)) return null;
  const v = c[key];
  return typeof v === 'number' ? v : null;
}

export function readoutFail(c: SetCell | undefined): string | null {
  if (!c || !c.readout_health) return null;
  return c.readout_health.startsWith('READOUT_FAIL') ? c.readout_health : null;
}

/** Human-readable reason a cell has no number. */
export function missingReason(m: Model, set: SetId): string {
  const c = m.sets[set];
  if (!c) return 'NOT MEASURED — no evaluation run';
  if (c.status !== 'MEASURED') return c.status;
  return 'NOT MEASURED';
}

export function coverageOf(m: Model, sets: SetId[]): number {
  return sets.filter((s) => metric(m, s, 'accuracy') !== null).length;
}

export const fmt = (v: number | null | undefined, d = 4): string =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : v.toFixed(d);
export const pct = (v: number | null | undefined, d = 1): string =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(d)}%`;

export const TIER_LABEL: Record<string, string> = {
  verified_full: 'Verified Full',
  verified_partial: 'Partial',
  public_only: 'Public Only',
};

export function sizeOk(m: Model, size: 'all' | 'le10' | 'le4'): boolean {
  if (size === 'all') return true;
  if (size === 'le10') return m.size_tier === '<=10B' || m.size_tier === '<=4B';
  return m.size_tier === '<=4B';
}

/** 2-axis frontier among the given models, using only full-coverage points. */
export function frontier2d(ms: Model[]): Set<string> {
  const pts = ms.filter(
    (m) =>
      m.general_medical_avg !== null &&
      m.ehr_docs_avg !== null &&
      coverageOf(m, GENERAL_SETS) === GENERAL_SETS.length &&
      coverageOf(m, EHR_SETS) === EHR_SETS.length,
  );
  const out = new Set<string>();
  for (const a of pts) {
    const dominated = pts.some(
      (b) =>
        b !== a &&
        b.general_medical_avg! >= a.general_medical_avg! &&
        b.ehr_docs_avg! >= a.ehr_docs_avg! &&
        (b.general_medical_avg! > a.general_medical_avg! || b.ehr_docs_avg! > a.ehr_docs_avg!),
    );
    if (!dominated) out.add(a.id);
  }
  return out;
}

// ---------- Derived headline figures (all computed, never typed in) ----------
const o9 = getModel('Osler-9B');
const o4 = getModel('Osler-4B');
const p27 = getModel('pplx-27B');

const macroRanked = models
  .filter((m) => m.five_set_macro !== null)
  .sort((a, b) => b.five_set_macro! - a.five_set_macro!);
const panelEceRanked = models
  .filter((m) => metric(m, 'panel', 'ece') !== null)
  .sort((a, b) => metric(a, 'panel', 'ece')! - metric(b, 'panel', 'ece')!);

function riskAt(set: string, model: string, cov: number): number | null {
  const pts = data.selective_prediction[set]?.[model];
  if (!pts) return null;
  const p = pts.find((x) => Math.abs(x.coverage - cov) < 0.01);
  return p ? p.accuracy : null;
}

export const derived = {
  o9,
  o4,
  p27,
  macroRank: o9 ? macroRanked.findIndex((m) => m.id === o9.id) + 1 : null,
  macroRankedCount: macroRanked.length,
  panelEceRank: o9 ? panelEceRanked.findIndex((m) => m.id === o9.id) + 1 : null,
  panelEceCount: panelEceRanked.length,
  speedup: o9?.p50_ms_b1 && p27?.p50_ms_b1 ? p27.p50_ms_b1 / o9.p50_ms_b1 : null,
  vramRatio: o9?.peak_vram_gb && p27?.peak_vram_gb ? p27.peak_vram_gb / o9.peak_vram_gb : null,
  mr80: riskAt('mr_verdict', 'Osler-9B', 0.8),
  mr50: riskAt('mr_verdict', 'Osler-9B', 0.5),
  mimic50: riskAt('mimic_suite', 'Osler-9B', 0.5),
  freshRank: (() => {
    const r = models
      .filter((m) => metric(m, 'confirm_general', 'accuracy') !== null)
      .sort((a, b) => metric(b, 'confirm_general', 'accuracy')! - metric(a, 'confirm_general', 'accuracy')!);
    return { rank: o9 ? r.findIndex((m) => m.id === o9.id) + 1 : null, of: r.length };
  })(),
};

export { riskAt };

export const phiAuditAllClear = data.phi_audit.length > 0 && data.phi_audit.every((r) => r.clear);
