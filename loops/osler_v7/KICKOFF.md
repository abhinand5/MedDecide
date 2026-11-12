# KICKOFF — osler_v7

Paste this into a fresh Sonnet 5.5 session on branch `loop/osler_v7` (`claude --model claude-sonnet-5-5`), then start `/loop`.

---

You are running the osler_v7 loop in /workspace/MedDecide.

1. `git branch --show-current` must print `loop/osler_v7`. If not, `git switch loop/osler_v7`.
2. Read, in order: `AGENTS.md` (Current loop section and the universal rules), `loops/osler_v7/GOAL.md`,
   `loops/osler_v7/ADVISORY.md`, `loops/osler_v7/STATE.md`. Where STATE.md and your memory disagree, STATE.md wins.
3. Take the first PENDING task in STATE §1 whose dependencies are DONE and whose approval is recorded in STATE §2.
   - No task is approved yet. T1, T2 and T3 need the operator's GO; GPU tasks need the GPU approval line.
   - If no task is approved, do nothing and report back in one message.
4. ADVISORY §0 lists five open decisions. Do not start T6 until §0.1–§0.4 are answered in STATE §2.
5. Never touch the external panel, its copies, the MIMIC audit suite or held-out items for training, selection or calibration.
   Never send MIMIC text to a hosted service. Never commit raw items, predictions, labels, checkpoints or keys.
6. Commit each completed task with `loop(osler_v7): T<n> <DONE|BLOCKED> — <headline>` and push `loop/osler_v7`. Run `uv run
   ruff check .` and `uv run pytest` before any commit that changes code. A loops-only commit skips them.
7. Do not start any GPU job without the operator's approval recorded in STATE §2. Log every GPU job under
   `outputs/osler_v7/<task>/logs/` and check its outputs, not only its exit code.
8. Stop when GOAL.md's "Done when" holds, and write the final report.
