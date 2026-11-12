#!/usr/bin/env python3
"""Phase 2 - short capability benchmark for the four models.

  * MMLU: 200 questions sampled (seed 0) from cais/mmlu "all" test.
    Zero-shot, chat template, score = argmax over the next-token log-probs
    of "A"/"B"/"C"/"D" (no generation, so no answer-parsing noise).
  * GSM8K: 48 problems (the first 48 of a seed-0 sample of 100) from
    openai/gsm8k "main" test. Downscaled from 100 because CPU generation
    of 320-token chains of thought was too slow on the shared box.
    Zero-shot chain of thought, greedy, max_new_tokens=320, the last number
    in the reply is compared to the reference answer.

Both are small samples: a 200-question MMLU accuracy has a 95% CI of about
+/-7 points, a 48-item GSM8K one about +/-13 points. Use them to spot
damage, not to rank close models.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from common import MODELS, chat, dump, generate_batch, load, log  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
N_MMLU = 200
N_GSM8K = 48   # downscaled from 100: GSM8K CoT generation was ~10 min per batch of 16 on the shared CPU box
GSM_MAX_NEW = 320
LETTERS = ["A", "B", "C", "D"]


def mmlu_items():
    ds = load_dataset("cais/mmlu", "all")["test"]
    idx = random.Random(0).sample(range(len(ds)), N_MMLU)
    return [ds[i] for i in idx]


def gsm_items():
    ds = load_dataset("openai/gsm8k", "main")["test"]
    idx = random.Random(0).sample(range(len(ds)), 100)[:N_GSM8K]
    return [ds[i] for i in idx]


def mmlu_prompt(item) -> str:
    subj = item["subject"].replace("_", " ")
    opts = "\n".join(f"{l}. {c}" for l, c in zip(LETTERS, item["choices"]))
    return (f"The following is a multiple choice question about {subj}.\n\n"
            f"{item['question']}\n{opts}\n\n"
            "Answer with the letter of the correct option only.")


def gsm_prompt(item) -> str:
    return (f"{item['question']}\n\nSolve the problem step by step. "
            "On the last line, write the final answer as 'Answer: <number>'.")


def gsm_ref(item) -> str:
    return item["answer"].split("####")[-1].strip().replace(",", "")


def last_number(text: str) -> str | None:
    m = re.search(r"Answer:\s*\$?(-?[\d,]*\.?\d+)", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return None


def run_mmlu(model, tok, items):
    letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
    correct = []
    for start in range(0, len(items), 8):
        chunk = items[start:start + 8]
        enc = tok([chat(tok, mmlu_prompt(it)) for it in chunk], return_tensors="pt", padding=True)
        lp = F.log_softmax(model(**enc, use_cache=False).logits[:, -1, :].float(), -1)
        pred = lp[:, letter_ids].argmax(-1).tolist()
        correct += [p == it["answer"] for p, it in zip(pred, chunk)]
    return correct


def main() -> None:
    t0 = time.perf_counter()
    mm, gs = mmlu_items(), gsm_items()
    results = {"config": {"n_mmlu": N_MMLU, "n_gsm8k": N_GSM8K, "gsm_max_new_tokens": GSM_MAX_NEW,
                          "seed": 0, "mmlu": "cais/mmlu all/test, zero-shot letter log-prob",
                          "gsm8k": "openai/gsm8k main/test, zero-shot CoT, greedy"},
               "models": {}}
    for name, path in MODELS.items():
        cache = CACHE / f"{name}.json"
        if cache.exists():
            results["models"][name] = json.loads(cache.read_text())
            continue
        log(f"{name}: loading")
        model, tok = load(path)
        t1 = time.perf_counter()
        mc = run_mmlu(model, tok, mm)
        log(f"  MMLU {sum(mc)}/{len(mc)}")
        outs = generate_batch(model, tok, [gsm_prompt(it) for it in gs], GSM_MAX_NEW, 16, f"{name}/gsm8k")
        gc = [last_number(o) == gsm_ref(it) for o, it in zip(outs, gs)]
        log(f"  GSM8K {sum(gc)}/{len(gc)}")
        entry = {
            "mmlu_correct": sum(mc), "mmlu_n": len(mc), "mmlu_acc": round(sum(mc) / len(mc), 3),
            "gsm8k_correct": sum(gc), "gsm8k_n": len(gc), "gsm8k_acc": round(sum(gc) / len(gc), 3),
            "mmlu_flags": mc, "gsm8k_flags": gc,
            "gsm8k_examples": [{"question": it["question"][:160], "reply_tail": o[-160:],
                                "ref": gsm_ref(it)} for o, it in list(zip(outs, gs))[:3]],
            "seconds": round(time.perf_counter() - t1, 1),
        }
        dump(cache, entry)
        results["models"][name] = entry
        del model
    # paired comparison vs original (how many items flip)
    base = results["models"]["original"]
    for name, e in results["models"].items():
        e["mmlu_flips_vs_original"] = {
            "gained": sum(b is False and x for b, x in zip(base["mmlu_flags"], e["mmlu_flags"])),
            "lost": sum(b and not x for b, x in zip(base["mmlu_flags"], e["mmlu_flags"])),
        }
        e["gsm8k_flips_vs_original"] = {
            "gained": sum(b is False and x for b, x in zip(base["gsm8k_flags"], e["gsm8k_flags"])),
            "lost": sum(b and not x for b, x in zip(base["gsm8k_flags"], e["gsm8k_flags"])),
        }
    results["runtime_seconds_this_invocation"] = round(time.perf_counter() - t0, 1)
    dump(HERE / "results.json", results)
    for name, e in results["models"].items():
        log(f"{name:22s} MMLU {e['mmlu_acc']:.1%}  GSM8K {e['gsm8k_acc']:.1%}  "
            f"flips mmlu {e['mmlu_flips_vs_original']} gsm {e['gsm8k_flips_vs_original']}")


if __name__ == "__main__":
    main()
