# student_v1 — what the next loop should take up

Written at closure of loop 1b (student_v1). Each item names what is open, why, and the decision or measurement that
would close it. The operator decisions come first, because the next loop's scope depends on them.

## 1. Decisions only the operator can make (from STATE.md §6)

1. **V2 acceptance item 1 (padded-batch tolerance).** The no-padding batch-1 path is the evaluation path for all arms.
   Either accept it as satisfying item 1, or revise the tolerance to the measured fp32 noise floor. Nothing in this
   loop depends on the answer, but the record should say which.
2. **V4 record–claim consistency designs and catalog train-candidate sources** were not built (NOT MEASURED, D4). Build
   them in the next loop, or drop them from the plan.
3. **Arm B's tripwire rule.** The verdict uses step 5,000 (rule as written). Step 14,000 is the unrestricted best, reported
   as additional. Confirm the rule, or treat the single 6,022.8 spike as clipped noise.
4. **Arm C's batch order and option order (D15, D16).** Arm C now uses arms A/B's chunked bucketing with fixed batches of
   8 and their option-order permutation. The alternative is Unsloth's default random batches and no permutation, which
   carries the train file's position prior (V085). Confirm the matched design.
5. **D12 on the trained letter readout.** Arm B fails the readout-health gate on 8 of 11 fresh templates and all held-out
   cells, because its probability sits partly on non-offered letters (V089, V100, V101). Options: report as the protocol
   states (the loop's choice); redefine the readout (renormalise over all letters, or include the space variants) as a
   new, labelled measurement; or amend D12 for trained letter readouts, which is a settled decision.
6. **Which scope the head comparison reads.** A minus C is -2.78 pts on the 7 fresh seen templates but +1.14 pts on all 15
   seen templates; on held-out, C leads in both scopes (V109, V110). The program decision (keep the head the evidence
   favours) needs this choice. The loop did not pick a head.

## 2. Measurements that were not made (R3)

- **HLE (supplementary).** `data/bench/v0.2/supplementary/hle_med.jsonl` has not been scored by any arm. It needs three
  scoring runs (arm A, arm B with D12, arm C) and the same report tables. Low cost; do it first.
- **Option-shuffle flip rate.** Needs re-scoring with permuted options on the GPU, for each arm. Not run.
- **Long-record slice.** The v0.2 definition is in `long_record.json` as counts only. The counts predate the screen's
  495-item drop (24,263 against 23,768 fresh test items), and the item membership is not stored. Regenerate the
  counts from the current items, store the membership, and then report the slice per arm.
- **Per-record latency.** Only per-item latency was recorded. Record per-record timing if the program needs it.
- **Arm B held-out G1.** All four held-out cells fail D12, so there is no verdict. Only a new readout measurement (item 5
  above) could give one.
- **Arm C latency.** Not like-for-like: the Unsloth environment runs the reference causal-conv kernel because
  `causal_conv1d` is not installed. Install it there (or measure all three arms in one environment) before comparing
  latency.

## 3. Defects and limits to fix or carry forward

- **Prompt cap truncates 5.9 % of test items** (2,213 items, all in three openFDA templates; V102). The scoring path cuts
  the leading tokens. Decide whether a longer cap is needed for these templates. The D12 greedy check uses untruncated
  prompts for these cells, so it should be made like-for-like before its agreement numbers are used.
- **JEV-9B coverage** (2,192 items over the 8,192 cap, from loop 1) is still a gap in the comparisons against JEV.
- **Seen-template score dev count** is 33 against a target of 200 (D9).
- **Dev selection margins are small.** Arm C's selected step 5,000 beats step 7,500 by 0.0009 (V099). Consider selecting
  on a larger dev set, or reporting the selection's sensitivity, before the next loop picks a checkpoint.
- **Option order for noul.** Arm C does not permute noul items (its Unsloth noul question has no option list); arms A and
  B do. If option order is part of the matched design, extend the converter to permute the two answers.
- **Scripts.** The eval path had three `relative_to` bugs (now fixed). Audit every script that takes a relative path
  argument before the next loop runs it.

## 4. Scope the next loop should not reopen

- The model ladder, the teacher, the benchmark tiers, the D12 thresholds, G1 and its baselines, and the release policy
  are settled (ADVISORY, PROGRAM). Only the decisions in §1 above are open.
- Training data stays private. No test split is used to fit a temperature, a threshold or a template.

## 5. Process lessons from this loop

- Never overwrite a chain script while a shell is still running it; write a new file and stop the old PID first
  (a duplicate training run came from this; the memory note records it).
- Gate every chain step on its exit code, and check the output files, not only the log's "done" line.
- Give every evaluation script's relative-path arguments a resolved, printable form before the first GPU run.
