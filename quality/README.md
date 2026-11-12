# Phase 2: capability benchmark

`run_quality.py` checks whether the edits cost general capability.

```bash
cd quality && python run_quality.py
```

* **MMLU**: 200 questions sampled (seed 0) from `cais/mmlu` all/test.
  Zero-shot, chat template, the answer is the argmax of the next-token
  log-probs of `A`/`B`/`C`/`D` (no generation, no parsing).
* **GSM8K**: 48 problems (the first 48 of a seed-0 sample of 100) from
  `openai/gsm8k` main/test. Zero-shot chain of thought, greedy,
  `max_new_tokens=320`, last number (or `Answer: N`) vs the reference.
  **Downscaled from the planned 100**: on the shared CPU box, generating
  320-token chains of thought ran at up to ~10 min per batch of 16, so the
  run was restarted at 48 items. The original model's first 48 items come
  from the 100-item run (identical batches).

## Results (`results.json`)

| Model | MMLU (200) | GSM8K (48) | MMLU flips vs original (gained / lost) | GSM8K flips (gained / lost) |
|-------|-----------:|-----------:|------------:|------------:|
| original | 85 (42.5%) | 20 (41.7%) | – | – |
| manual (layer 12) | 87 (43.5%) | 17 (35.4%) | 17 / 15 | 4 / 7 |
| Heretic Abliteration | 85 (42.5%) | 15 (31.2%) | 4 / 4 | 1 / 6 |
| Heretic ARA | 88 (44.0%) | 15 (31.2%) | 17 / 14 | 1 / 6 |

95% intervals are about ±7 points (MMLU, n=200) and ±13 points (GSM8K,
n=48), so **no difference here is statistically significant**. For example
Heretic's 1 gained / 6 lost GSM8K items gives an exact McNemar p ≈ 0.13.

What the numbers do say:

* **MMLU is untouched by all three edits.** Accuracy moves by at most
  1.5 points. The flip counts show how different the edits are inside:
  Heretic Abliteration changes only 8 answers, while manual and ARA change
  ~30 each, in both directions, and land at the same total.
* **GSM8K is the one place all three edits lean the same way, down 3–5
  items out of 48.** Multi-step generation compounds small per-token drift,
  so it is a more sensitive probe than a single-token MMLU letter. It is a
  hint worth re-testing with all 1,319 items on a GPU, not a result.
* For scale: Qwen reports ~47.5% MMLU (5-shot) and ~49.6% GSM8K (4-shot)
  for this model, and our zero-shot numbers sit a little below that.
