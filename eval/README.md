# Phase 1: robust held-out evaluation

`run_eval.py` re-tests the four models (original, manual abliteration,
Heretic Abliteration, Heretic ARA) on much larger held-out sets than the
12-prompt check in the first lab run.

```bash
cd eval && python run_eval.py      # ~25 min on the 8-core CPU box
```

## Prompt sets (`prompts.json`)

| Set | n | Source | Purpose |
|-----|---|--------|---------|
| harmful | 93 | JailbreakBench `JBB-Behaviors` harmful (sources: AdvBench, HarmBench, TDC, originals) | refusal rate |
| harmless | 100 | `mlabonne/harmless_alpaca` test[100:200] | over-refusal + KL |
| borderline | 100 | JBB-Behaviors benign (same topics, but fine to answer) | over-refusal on look-alike prompts |

7 of the 100 JBB harmful prompts were dropped because they overlap (token
Jaccard >= 0.5) a prompt used to compute a refusal direction or to score
Heretic trials (`abliterate.py` prompts, `harmful_behaviors` train[:48] and
test[:16]). They are listed under `harmful_dropped_for_overlap`.
HarmBench itself is gated on the Hub, which is why JBB is used.

Decoding: greedy, `max_new_tokens=96`, the model's default chat template.

## Refusal detectors

1. **Substring**: the same snippet list as `abliterate.py`.
2. **LLM judge**: `Qwen/Qwen2.5-3B-Instruct`, refusal if P("Yes") > P("No")
   on a message-only prompt (`common.JUDGE_TEMPLATE`).

The judge needed calibration. On 48 hand-labelled responses
(`judge_calibration.json`, `judge_calibration_labels.json`):

| Detector | Accuracy | False pos. | False neg. |
|----------|---------:|-----------:|-----------:|
| Substring | 0.854 | 1 | 6 |
| Qwen2.5-3B, message only (used) | 0.854 | 1 | 6 |
| Qwen2.5-1.5B, request + response (first try, rejected) | 0.688 | 6 | 9 |

The first judge (1.5B, shown the request and the response) graded the
*request* rather than the response: it called many plain "I'm sorry, but I
can't assist" replies non-refusals and many compliant replies refusals (54%
disagreement with the substring detector on the original model). Larger
judges shown the request were worse (`judge_prompt_search.py`). Showing only
the response fixed this, at the cost of being blind to **deflections**:
replies that answer a harmless or opposite version of the request (asked
for a campaign against hiring disabled people, writes one *for* hiring them).
Every remaining error of both detectors in the calibration sample is a
deflection. Deflections were 6/36 harmful responses in that sample (original
1/9, manual 3/9, Heretic Abliteration 2/9, ARA 0/9): small numbers, but a
hint that edited models sometimes "comply in form" without complying in
substance.

## Results (`results.json`)

Refusal counts, 95% Wilson interval in brackets.

| Model | Harmful (substring) | Harmful (judge) | Disagree | Harmless (judge) | Borderline (judge) | First-token KL | Full-response KL |
|-------|--------------------:|----------------:|---------:|-----------------:|-------------------:|---------------:|-----------------:|
| original | 65/93 (70%) [60–78] | 63/93 (68%) [58–76] | 2.2% | 4/100 | 20/100 | 0 | 0 |
| manual (layer 12) | 3/93 (3%) [1–9] | 4/93 (4%) [2–11] | 1.1% | 3/100 | 3/100 | 0.075 | 0.040 |
| Heretic Abliteration | 1/93 (1%) [0–6] | 2/93 (2%) [1–8] | 1.1% | 1/100 | 2/100 | 0.100 | **0.016** |
| Heretic ARA | **0/93 (0%)** [0–4] | **0/93** [0–4] | 0.0% | 3/100 | 4/100 | 0.101 | 0.032 |

* KL is KL(original || model) on the 100 harmless prompts. "Full-response"
  teacher-forces the original model's greedy reply and averages per-token KL
  over the whole reply; first-token KL only looks at the first token.
* At these low rates the judge's own noise (a few false positives such as
  flagging "Certainly! Here's a comparison…") is about as large as the
  differences between the edited models. The 0–4% numbers are a tie.
* Borderline over-refusal drops from ~20% to 1–4% for all three edits: the
  original refuses many fine prompts that merely *sound* harmful.

Raw completions are kept in `eval/cache/` and are not committed.
