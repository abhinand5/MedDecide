# Community Leaderboard Submissions

To submit a **Public Tier** score on [`abhinand/MedDecide-Bench`](https://huggingface.co/datasets/abhinand/MedDecide-Bench) (`confirm_general`, 5,988 public-domain items):

1. Run your model on `confirm_general` and save per-item probabilities (`predictions.jsonl`):
   ```bash
   python scripts/bench/eval_public_bench.py \
     --model <your-hf-repo> \
     --out predictions.jsonl
   ```
2. Verify your predictions and generate a submission file:
   ```bash
   python scripts/bench/verify_submission.py \
     --predictions predictions.jsonl \
     --model-name MyModel-8B \
     --hf-repo myorg/MyModel-8B \
     --params-b 8.0 \
     --license Apache-2.0 \
     --out-submission site/submissions/myorg__MyModel-8B.json
   ```
3. Open a Pull Request adding `site/submissions/<org>__<model>.json` (and linking your `predictions.jsonl` trace).
4. Want your **open-weight** model scored on the **Private EHR & Clinical Documentation Suite** (`mimic_suite` + `mr_verdict`)? Open a [Private Eval Request Issue](https://github.com/abhinand5/MedDecide/issues/new?template=private_eval_request.yml).
