# RUNBOOK — running a MedDecide loop on the pod

For the operator. The loop agent never needs this file; it reads `AGENTS.md`.

## Machines

| role | machine | notes |
|---|---|---|
| loop pod | 1× RTX PRO 6000 Blackwell (96 GB), 128 cores, ~2 TB RAM, CUDA 13, Python 3.12, `uv` | root disk 20 GB; `/workspace` is a large network volume |
| teacher | 4× RTX PRO 6000 serving `deepseek-ai/DeepSeek-V4.1-Flash` | only when a task needs it (bench_v0: T10) |
| agent LLM | DeepSeek via the official API or OpenRouter | called by the agent harness on the loop pod |

Verified on 2026-10-06 for pod `0x1dieouykoxg2`: GPU, CUDA 13.0, Python 3.12.3, `uv`,
`git`, `node`; outbound HTTP 200 from clinicaltrials.gov, api.fda.gov,
ftp.ncbi.nlm.nih.gov, huggingface.co, openrouter.ai, github.com. Not set: `HF_TOKEN`,
git identity, SSH keys.

## Prerequisites (do these before starting a loop)

1. **Secrets file** at `/workspace/.secrets.env` (outside the repo):
   ```bash
   export HF_TOKEN=hf_...                 # account that has accepted google/medgemma-1.5-4b-it terms
   export DEEPSEEK_API_KEY=...            # or OPENROUTER_API_KEY — whatever your harness reads
   # export TEACHER_BASE_URL=http://<teacher-host>:<port>/v1   # when the teacher is up
   # export TEACHER_API_KEY=...
   ```
   `chmod 600 /workspace/.secrets.env`.
2. **MedGemma access:** on huggingface.co, open `google/medgemma-1.5-4b-it` with the
   account behind `HF_TOKEN` and accept the Health AI Developer Foundations terms.
3. **GitHub push access from the pod** (the loop commits its progress so you can review
   it remotely). Simplest: a deploy key with write access, scoped to this repo only:
   ```bash
   ssh-keygen -t ed25519 -N '' -f ~/.ssh/meddecide_deploy
   cat ~/.ssh/meddecide_deploy.pub   # add in GitHub → MedDecide → Settings → Deploy keys → "Allow write access"
   cat >> ~/.ssh/config <<'CFG'
   Host github.com
     IdentityFile ~/.ssh/meddecide_deploy
     IdentitiesOnly yes
   CFG
   git config --global user.name "MedDecide loop agent"
   git config --global user.email "<your email or a noreply address>"
   ```
   Note: `~/.ssh` lives on the root disk; if the pod is recreated, redo this step.
4. **Clone and environment:**
   ```bash
   cd /workspace && git clone git@github.com:abhinand5/MedDecide.git
   cd /workspace/MedDecide && source scripts/pod_env.sh
   apt-get update && apt-get install -y tmux     # if missing
   ```
5. **Agent harness** installed on the pod, configured with the DeepSeek model, able to
   run non-interactively with shell + file-edit tools auto-approved, and taking the
   prompt as its final command-line argument. That invocation is `AGENT_CMD` below.
   Test it once with a trivial prompt before the overnight run.
6. **Teacher (only for T10):** DeepSeek-V4.1-Flash behind an OpenAI-compatible server
   (vLLM or SGLang) that returns log-probabilities, with the per-token logprob limit set
   high enough to cover every option (e.g. vLLM `--max-logprobs 64`). Export
   `TEACHER_BASE_URL` / `TEACHER_API_KEY` into the secrets file when it is up. If it is
   not up when the loop reaches T10, T10 is marked BLOCKED and the loop continues.

## Start the loop

**With the DeepSeek harness's goal loop (web UI on the pod, the default):**

1. Make sure the harness process was started from a shell that ran
   `source /workspace/MedDecide/scripts/pod_env.sh` — the agent's commands inherit those
   cache paths and secrets. If it was started otherwise, restart it that way.
2. Working directory: `/workspace/MedDecide`.
3. Goal prompt: paste the block under "Prompt" in `loops/bench_v0/KICKOFF.md` verbatim.
4. Stop / completion condition (if the harness asks for one): the loop is finished when
   `loops/bench_v0/STATE.md` contains ``Loop status: `STOPPED` ``.
5. Allow shell and file-edit tools without per-call approval; allow long sessions.

Reaching the web UI from your laptop: `ssh.runpod.io` (the RunPod SSH proxy) does not
support port forwarding. Use the pod's **direct TCP SSH** from the RunPod console
("SSH over exposed TCP"; expose TCP port 22 on the pod if it is missing):
`ssh -i ~/Documents/g14_ssh_dir/runpod -p <port> -N -L 8888:127.0.0.1:8888 root@<ip>`,
or Tailscale. Do not expose the UI through RunPod's public HTTP proxy unless the
harness has its own authentication — it drives an agent with shell access.

**Without a goal-loop harness (fallback):** any CLI agent that takes the prompt as its
last argument can be driven by the outer loop script:

```bash
tmux new -s loop
cd /workspace/MedDecide && source scripts/pod_env.sh
AGENT_CMD='<cli agent command>' bash scripts/run_loop.sh bench_v0
```

It re-invokes the agent with the KICKOFF prompt until STATE.md says `STOPPED`.

## Review progress (from your laptop)

```bash
cd ~/dev/projects/MedDecide && git fetch && git checkout loop/bench_v0 && git pull
less loops/bench_v0/STATE.md        # board, iteration log, blocked items, questions
less loops/bench_v0/CLAIMS.md
ls loops/bench_v0/                  # committed reports (baselines.md, teacher_gate.md, ...)
```

At the hard stop, bring `FINDINGS.md`, `STATE.md` and `NEXT.md` to the advisor session
for review and planning of the next loop.

## The audit (T11)

When STATE.md asks for it: copy `outputs/bench_v0/T11/audit_sample.jsonl` off the pod
(`scp` or `runpodctl send`), open `tools/audit/audit.html` in a browser, load the file,
press `A`/`R`/`N` per item, Export, and copy `audit_v0.jsonl` back to
`/workspace/MedDecide/outputs/bench_v0/T11/audit_v0.jsonl`. The sample is benchmark
data — keep it off git.

## Branches

Each loop runs on `loop/<loop-name>`; `main` holds reviewed state only. After the
advisor review at a hard stop, merge the loop branch into `main`
(`git checkout main && git merge --no-ff loop/bench_v0 && git push`), then start the
next loop's branch from `main`.

## Stopping early

`tmux attach -t loop`, `Ctrl-c` the runner. The agent's state is in STATE.md; restart
the runner the same way to resume.
