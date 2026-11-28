import { useState } from 'react';
import { ExternalLink, GitPullRequest, Lock, MessageSquarePlus, ShieldAlert } from 'lucide-react';
import { evalSet, links } from '../lib/data';
import { CodeBlock, Pills, SectionHeader } from './ui';
import { Brand } from './Nav';
import { GithubIcon, HfIcon } from './Icons';

// ---------------------------------------------------------------- Quickstart
const SNIP_9B = `from transformers import AutoModel

# Standalone merged weights: no meddecide library needed.
model = AutoModel.from_pretrained("abhinand/Osler-9B", trust_remote_code=True)

result = model.system_one(
    state=(
        "68yo M with fever 38.9C, BP 84/52, lactate 4.2 mmol/L, "
        "on norepinephrine for E. coli bacteremia."
    ),
    questions={
        "septic_shock": {
            "type": "noul",
            "instructions": "Does the note support septic shock?",
        },
    },
)
print(result)  # calibrated p(yes) from one prefill pass, no generation`;

const SNIP_4B = `from transformers import AutoModel

model = AutoModel.from_pretrained("abhinand/Osler-4B", trust_remote_code=True)

result = model.system_one(
    state=(
        "71yo M, ischemic cardiomyopathy (EF 25%), 4 kg weight gain, "
        "JVP 14 cm, BNP 2,650 pg/mL, CXR vascular congestion."
    ),
    questions={
        "diuresis": {
            "type": "noul",
            "instructions": "Is IV loop diuretic therapy clinically indicated?",
        },
    },
)
print(result)`;

const SNIP_VLLM = `# vLLM Semantic Router (Engine mode) for Osler-9B
curl -fsSL https://vllm-sr.ai/install.sh | bash -s -- --channel dev --no-launch
export PATH="$HOME/.local/bin:$PATH"
vllm-sr serve -e abhinand/Osler-9B

# Query the live System One endpoint
curl -s http://localhost:8899/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "model": "abhinand/Osler-9B",
    "state": "Serum creatinine rose from 0.9 mg/dL to 2.4 mg/dL over 36 hours with oliguria.",
    "questions": {
      "aki": {"type": "noul", "instructions": "Do the documented lab values meet KDIGO AKI criteria?"}
    }
  }'

# Note: vllm-sr applies the scalar temperature from decision_config.json (exact for choice).
# For exact per-type noul/score temperatures over HTTP, use the bundled d3_server.py.`;

const SNIP_BENCH = `from datasets import load_dataset

# Public tier: post-cutoff, structured-gold decisions (CT.gov, openFDA, PubMed)
ds = load_dataset("abhinand/MedDecide-Bench", "confirm_general", split="test")
print(len(ds), ds[0].keys())`;

type QTab = '9b' | '4b' | 'vllm' | 'bench';

export function Quickstart() {
  const [tab, setTab] = useState<QTab>('9b');
  const map: Record<QTab, { code: string; lang: string }> = {
    '9b': { code: SNIP_9B, lang: 'python' },
    '4b': { code: SNIP_4B, lang: 'python' },
    vllm: { code: SNIP_VLLM, lang: 'bash' },
    bench: { code: SNIP_BENCH, lang: 'python' },
  };
  return (
    <section id="quickstart" className="section py-20">
      <SectionHeader
        eyebrow="06 · Quickstart"
        title={
          <>
            Three lines to a calibrated <span className="italic text-muted">decision.</span>
          </>
        }
        lede="Merged safetensors with a bundled runtime and built-in dev-only per-type calibration."
      />
      <div className="mb-4 overflow-x-auto pb-1">
        <Pills
          label="Snippet"
          value={tab}
          onChange={setTab}
          options={[
            { value: '9b', label: 'Osler-9B (HF AutoModel)' },
            { value: '4b', label: 'Osler-4B (HF AutoModel)' },
            { value: 'vllm', label: 'vLLM Semantic Router' },
            { value: 'bench', label: 'MedDecide-Bench' },
          ]}
        />
      </div>
      <CodeBlock key={tab} code={map[tab].code} lang={map[tab].lang} />
    </section>
  );
}

// ---------------------------------------------------------------- Submit
const SUBMIT_CMD = `python scripts/bench/eval_public_bench.py \\
  --model <your-hf-repo> \\
  --out predictions.jsonl

python scripts/bench/verify_submission.py \\
  --predictions predictions.jsonl \\
  --model-name MyModel-8B \\
  --hf-repo myorg/MyModel-8B \\
  --params-b 8.0 \\
  --license Apache-2.0 \\
  --out-submission site/submissions/myorg__MyModel-8B.json`;

export function Submit() {
  const cg = evalSet('confirm_general');
  const prUrl = `${links.github}/compare`;
  return (
    <section id="submit" className="section py-20">
      <SectionHeader
        eyebrow="07 · Submit & request evaluation"
        title={
          <>
            Put your model <span className="italic text-muted">on the board.</span>
          </>
        }
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="card flex flex-col p-6">
          <div className="mb-1 flex items-center gap-2">
            <GitPullRequest size={17} className="text-accent" />
            <h3 className="text-[17px] font-semibold text-ink">Submit a Public Tier score (PR)</h3>
          </div>
          <p className="mb-4 text-[13.5px] leading-relaxed text-muted">
            Run your model on <code className="num">confirm_general</code> ({cg?.n_items.toLocaleString()} public-domain items). Any open or
            closed model is eligible. Verify the per-item probabilities, then open a pull request that adds{' '}
            <code className="num">site/submissions/&lt;org&gt;__&lt;model&gt;.json</code> and links your{' '}
            <code className="num">predictions.jsonl</code> trace.
          </p>
          <CodeBlock code={SUBMIT_CMD} lang="bash" />
          <div className="mt-5 flex flex-wrap gap-2">
            <a href={prUrl} target="_blank" rel="noreferrer" className="btn-primary">
              <GithubIcon size={15} /> Open a Pull Request
            </a>
            <a href={`${links.github}/tree/main/site/submissions`} target="_blank" rel="noreferrer" className="btn-ghost">
              Submission guide <ExternalLink size={13} />
            </a>
          </div>
        </div>

        <div className="card flex flex-col p-6">
          <div className="mb-1 flex items-center gap-2">
            <Lock size={17} className="text-accent" />
            <h3 className="text-[17px] font-semibold text-ink">Request a Private EHR evaluation (issue)</h3>
          </div>
          <p className="mb-4 text-[13.5px] leading-relaxed text-muted">
            The EHR & Clinical Documentation suite (<code className="num">mimic_suite</code> + <code className="num">mr_verdict</code>) is
            built on credentialed PhysioNet data and is never published. The maintainers score requested models locally and publish only aggregate
            numbers.
          </p>
          <div className="mb-5 rounded-xl border border-warn/30 bg-warn/[0.06] p-4">
            <div className="mb-2 flex items-center gap-2 text-[13px] font-semibold text-warn">
              <ShieldAlert size={15} /> PhysioNet DUA compliance
            </div>
            <ul className="space-y-1.5 text-[13px] leading-relaxed text-ink/85">
              <li className="flex gap-2">
                <span className="text-accent">✓</span>
                <span>
                  <strong className="font-semibold">Open-weight models</strong> run locally, on PhysioNet-authorised hardware. No MIMIC text ever
                  leaves that machine.
                </span>
              </li>
              <li className="flex gap-2">
                <span className="text-bad">✕</span>
                <span>
                  <strong className="font-semibold">Closed / hosted APIs</strong> can enter the Public Tier only. Sending MIMIC text to a
                  third-party API breaks the DUA.
                </span>
              </li>
            </ul>
          </div>
          <ol className="mb-5 space-y-2 text-[13px] leading-relaxed text-muted">
            {[
              'Open the issue with your Hugging Face repo, revision and licence.',
              'The maintainers run the frozen private sets locally on PhysioNet-authorised hardware.',
              'Only aggregate accuracy, Brier, ECE and 95% CIs are added to this board.',
            ].map((t, i) => (
              <li key={i} className="flex gap-3">
                <span className="num grid h-5 w-5 shrink-0 place-items-center rounded-full border border-line text-[10.5px] text-ink">{i + 1}</span>
                <span>{t}</span>
              </li>
            ))}
          </ol>
          <div className="mt-auto">
            <a href={links.private_eval_issue} target="_blank" rel="noreferrer" className="btn-primary">
              <MessageSquarePlus size={15} /> Request Private Eval
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------- Citation & footer
const BIBTEX = `@misc{abhinand2026osler,
  author = {Abhinand},
  title  = {Osler: Calibrated Single-Pass Medical Decision Models},
  year   = {2026},
  note   = {Technical report / arXiv forthcoming},
  url    = {https://github.com/abhinand5/MedDecide}
}`;

export function Footer({ generated }: { generated: string }) {
  const cols: { h: string; items: [string, string][] }[] = [
    {
      h: 'Models',
      items: [
        ['Osler-9B', links.hf_osler_9b],
        ['Osler-4B', links.hf_osler_4b],
        ['HF Collection', links.hf_collection],
      ],
    },
    {
      h: 'Benchmark',
      items: [
        ['MedDecide-Bench dataset', links.hf_bench],
        ['Leaderboard Space', links.hf_space],
        ['Landing page', links.landing_page],
      ],
    },
    {
      h: 'Code',
      items: [
        ['GitHub repository', links.github],
        ['Request private eval', links.private_eval_issue],
        ['Apache-2.0 license', 'https://www.apache.org/licenses/LICENSE-2.0'],
      ],
    },
  ];
  return (
    <footer className="mt-8 border-t border-line">
      <section id="citation" className="section py-16">
        <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <div>
            <div className="eyebrow mb-3">08 · Citation</div>
            <h2 className="h-display text-[2.2rem] leading-tight">Cite Osler</h2>
            <p className="mt-3 text-[14px] leading-relaxed text-muted">
              If you use Osler or MedDecide-Bench, please cite the technical report. The weights, the benchmark (public-source items only) and
              the code are open; the training data stays private.
            </p>
          </div>
          <CodeBlock code={BIBTEX} lang="bibtex" />
        </div>
      </section>
      <div className="section grid gap-10 border-t border-line py-12 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr_1fr]">
        <div>
          <Brand />
          <p className="mt-4 max-w-xs text-[13px] leading-relaxed text-muted">
            Open, calibrated single-pass medical decision models with an open benchmark. Not a medical device and not for unsupervised
            clinical use.
          </p>
          <div className="mt-4 flex gap-2">
            <a href={links.github} target="_blank" rel="noreferrer" aria-label="GitHub" className="grid h-9 w-9 place-items-center rounded-full border border-line text-muted hover:text-ink">
              <GithubIcon size={15} />
            </a>
            <a href={links.hf_collection} target="_blank" rel="noreferrer" aria-label="Hugging Face" className="grid h-9 w-9 place-items-center rounded-full border border-line">
              <HfIcon size={15} />
            </a>
          </div>
        </div>
        {cols.map((c) => (
          <div key={c.h}>
            <div className="eyebrow mb-3">{c.h}</div>
            <ul className="space-y-2">
              {c.items.map(([label, href]) => (
                <li key={label}>
                  <a href={href} target="_blank" rel="noreferrer" className="text-[13px] text-ink/80 hover:text-accent">
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      <div className="section flex flex-col gap-2 border-t border-line py-6 text-[11.5px] text-muted sm:flex-row sm:justify-between">
        <span>© 2026 Abhinand · Apache-2.0</span>
        <span className="num">data generated {generated}</span>
      </div>
    </footer>
  );
}
