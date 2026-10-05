# MedDecide

A family of open, calibrated **medical decision models** ("System One" / Jev-style):
given a state and typed questions (`noul` yes/no, `choice`, ordered `score`), they
return a probability for every allowed option from a single forward pass — no text
generation, bounded and threshold-able outputs.

Alongside the models: **MedDecide-Bench**, a public and regenerable benchmark whose
headline tier is built from records published after the evaluated models' training
cutoffs, with gold labels taken from structured source fields.

**Status:** loop 0 (`bench_v0`) — building the benchmark, harness, and baselines. No
model is released yet; "MedDecide" is a working name.

- Program plan: [docs/plans/PROGRAM.md](docs/plans/PROGRAM.md)
- Agent operating guide: [AGENTS.md](AGENTS.md)
- Running a loop: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Current loop: [loops/bench_v0/](loops/bench_v0/)
