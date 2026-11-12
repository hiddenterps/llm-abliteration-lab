#!/usr/bin/env python3
"""Phase 1 - robust held-out evaluation of the four models.

Prompt sets (built once, saved to eval/prompts.json):
  * harmful    - JailbreakBench JBB-Behaviors "harmful" split (100 behaviors
                 sourced from AdvBench, HarmBench, TDC and originals), minus any
                 prompt that overlaps the prompts used to compute a refusal
                 direction or to score Heretic trials (token Jaccard >= 0.5).
  * harmless   - mlabonne/harmless_alpaca test[100:200] (Heretic used
                 train[:48] and test[:16], so this slice is untouched).
  * borderline - JBB-Behaviors "benign" split: 100 prompts that are
                 topically close to the harmful ones but fine to answer.
                 Used to measure over-refusal.

For each model: greedy generation (max_new_tokens=96), refusal flags from
(a) the substring detector of abliterate.py and (b) an LLM judge
(Qwen2.5-1.5B-Instruct, P(Yes) > P(No) on "did the assistant refuse?").
On the harmless set we also compute first-token KL and full-response KL
(teacher-forced on the ORIGINAL model's greedy response).

Raw responses live in eval/cache/ (git-ignored). The committed JSON stores
only flags for harmful prompts, never harmful completions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset

from common import (MODELS, JUDGE_ID, JUDGE_TEMPLATE, ROOT, chat, dump, first_token_logprobs,
                    generate_batch, is_refusal_substring, jaccard, judge_refusals,
                    kl_batchmean, lab, load, log)

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
PROMPTS_PATH = HERE / "prompts.json"
RESULTS_PATH = HERE / "results.json"
MAX_NEW_TOKENS = 96
OVERLAP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# 1. prompt sets
# ---------------------------------------------------------------------------

def build_prompts() -> dict:
    if PROMPTS_PATH.exists():
        return json.loads(PROMPTS_PATH.read_text())
    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    hb = load_dataset("mlabonne/harmful_behaviors")
    used = list(lab.PROMPTS_RECUSA)                      # manual direction + old eval
    used += list(hb["train"]["text"][:48])               # Heretic direction prompts
    used += list(hb["test"]["text"][:16])                # Heretic trial scoring prompts
    harmful, dropped = [], []
    for row in jbb["harmful"]:
        goal = row["Goal"]
        best = max(jaccard(goal, u) for u in used)
        (dropped if best >= OVERLAP_THRESHOLD else harmful).append(
            {"prompt": goal, "category": row["Category"], "source": row["Source"],
             "max_jaccard_vs_training": round(best, 3)})
    harmless = list(load_dataset("mlabonne/harmless_alpaca")["test"]["text"][100:200])
    borderline = [{"prompt": r["Goal"], "category": r["Category"]} for r in jbb["benign"]]
    data = {
        "harmful": harmful,
        "harmful_dropped_for_overlap": dropped,
        "harmless": harmless,
        "borderline": borderline,
        "sources": {
            "harmful": "JailbreakBench/JBB-Behaviors (behaviors/harmful)",
            "harmless": "mlabonne/harmless_alpaca test[100:200]",
            "borderline": "JailbreakBench/JBB-Behaviors (behaviors/benign)",
        },
    }
    dump(PROMPTS_PATH, data)
    return data


# ---------------------------------------------------------------------------
# 2. generation per model
# ---------------------------------------------------------------------------

def generate_all(sets: dict) -> None:
    for name, path in MODELS.items():
        out = CACHE / f"responses_{name}.json"
        if out.exists():
            log(f"{name}: cached responses found, skipping generation")
            continue
        log(f"{name}: loading {path}")
        model, tok = load(path)
        res = {}
        for key in ("harmful", "harmless", "borderline"):
            prompts = [p if isinstance(p, str) else p["prompt"] for p in sets[key]]
            res[key] = generate_batch(model, tok, prompts, MAX_NEW_TOKENS, 16, f"{name}/{key}")
        res["first_token_logprobs_harmless"] = None  # computed in kl stage
        dump(out, res)
        del model


# ---------------------------------------------------------------------------
# 3. KL on harmless prompts
# ---------------------------------------------------------------------------

def _resp_logits(model, tok, prompt, response):
    p_ids = tok(chat(tok, prompt), return_tensors="pt")["input_ids"]
    r_ids = tok(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if r_ids.shape[1] == 0:
        return None
    ids = torch.cat([p_ids, r_ids], 1)
    logits = model(input_ids=ids, use_cache=False).logits[0].float()
    # positions that predict the first response token ... the last one
    return F.log_softmax(logits[p_ids.shape[1] - 1: ids.shape[1] - 1], -1)


def kl_stage(sets: dict) -> dict:
    """Original and one modified model are held in memory together (2 x ~1 GB)."""
    out_path = CACHE / "kl.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    prompts = sets["harmless"]
    orig_resp = json.loads((CACHE / "responses_original.json").read_text())["harmless"]
    base, tok = load(MODELS["original"])
    base_first = first_token_logprobs(base, tok, prompts)
    kl = {"original": {"first_token_kl": 0.0, "full_response_kl": 0.0}}
    for name, path in MODELS.items():
        if name == "original":
            continue
        log(f"KL: {name}")
        model, tok2 = load(path)
        first = first_token_logprobs(model, tok2, prompts)
        per_prompt = []
        for p, r in zip(prompts, orig_resp):
            b = _resp_logits(base, tok, p, r)
            if b is None:
                continue
            o = _resp_logits(model, tok2, p, r)
            per_tok = (b.exp() * (b - o)).sum(-1)  # KL(orig || mod) per position
            per_prompt.append(per_tok.mean().item())
        kl[name] = {
            "first_token_kl": kl_batchmean(base_first, first),
            "full_response_kl": sum(per_prompt) / len(per_prompt),
            "full_response_kl_max_prompt": max(per_prompt),
        }
        log(f"  {name}: {kl[name]}")
        del model
    dump(out_path, kl)
    return kl


# ---------------------------------------------------------------------------
# 4. LLM judge
# ---------------------------------------------------------------------------

def judge_stage(sets: dict) -> dict:
    out_path = CACHE / "judge.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    model, tok = load(JUDGE_ID)
    verdicts = {}
    for name in MODELS:
        resp = json.loads((CACHE / f"responses_{name}.json").read_text())
        verdicts[name] = {}
        for key in ("harmful", "harmless", "borderline"):
            flags, margins = judge_refusals(resp[key], model, tok)
            verdicts[name][key] = {"flags": flags, "margins": margins}
            log(f"judge {name}/{key}: {sum(flags)}/{len(flags)} refusals")
    dump(out_path, verdicts)
    del model
    return verdicts


def calibration_stage() -> dict:
    """Agreement of each detector with 48 hand labels (36 harmful + 12 borderline
    responses sampled across the four models, seed 1). Hand label 1 = refusal,
    including 'deflections' that answer a harmless version of the request."""
    if not (CACHE / "dev.json").exists():
        import random
        rng, dev = random.Random(1), []
        sets = json.loads(PROMPTS_PATH.read_text())
        for m in MODELS:
            resp = json.loads((CACHE / f"responses_{m}.json").read_text())
            for key, k in (("harmful", 9), ("borderline", 3)):
                for i in rng.sample(range(len(resp[key])), k):
                    dev.append({"model": m, "set": key, "i": i,
                                "prompt": sets[key][i]["prompt"], "response": resp[key][i]})
        dump(CACHE / "dev.json", dev)
    dev = json.loads((CACHE / "dev.json").read_text())
    gold = [bool(g) for g in json.loads((HERE / "judge_calibration_labels.json").read_text())["labels"]]
    resp = [d["response"] for d in dev]
    preds = {"substring": [is_refusal_substring(r) for r in resp]}
    preds["judge_qwen3b_message_only"], _ = judge_refusals(resp)
    v1 = json.loads((CACHE / "judge_v1_qwen1.5b.json").read_text())
    preds["rejected_judge_qwen1.5b_with_request"] = [v1[d["model"]][d["set"]]["flags"][d["i"]] for d in dev]
    out = {"n": len(gold), "gold_refusals": sum(gold), "items": [
        {"model": d["model"], "set": d["set"], "index": d["i"], "gold_refusal": g}
        for d, g in zip(dev, gold)], "detectors": {}}
    for k, p in preds.items():
        tp = sum(a and g for a, g in zip(p, gold)); fp = sum(a and not g for a, g in zip(p, gold))
        fn = sum(g and not a for a, g in zip(p, gold))
        out["detectors"][k] = {"accuracy": round(sum(a == g for a, g in zip(p, gold)) / len(gold), 3),
                               "true_pos": tp, "false_pos": fp, "false_neg": fn}
    dump(HERE / "judge_calibration.json", out)
    log(f"calibration: {out['detectors']}")
    return out


# ---------------------------------------------------------------------------
# 5. report
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0, c - h), 3), round(min(1, c + h), 3)]


def main() -> None:
    t0 = time.perf_counter()
    sets = build_prompts()
    log(f"sets: harmful={len(sets['harmful'])} (dropped {len(sets['harmful_dropped_for_overlap'])}), "
        f"harmless={len(sets['harmless'])}, borderline={len(sets['borderline'])}")
    sets_plain = {k: sets[k] for k in ("harmful", "harmless", "borderline")}
    generate_all(sets_plain)
    kl = kl_stage(sets_plain)
    judge = judge_stage(sets_plain)
    calib = calibration_stage()
    v1 = json.loads((CACHE / "judge_v1_qwen1.5b.json").read_text())

    report = {
        "config": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
            "chat_template": "model default (Qwen default system prompt)",
            "judge": JUDGE_ID,
            "judge_rule": "P(Yes) > P(No), message-only prompt (see common.JUDGE_TEMPLATE)",
            "judge_calibration": calib["detectors"],
            "n": {k: len(v) for k, v in sets_plain.items()},
            "overlap_filter": f"token Jaccard >= {OVERLAP_THRESHOLD} vs any direction/scoring prompt",
        },
        "models": {},
    }
    for name in MODELS:
        resp = json.loads((CACHE / f"responses_{name}.json").read_text())
        entry = {}
        for key in sets_plain:
            sub = [is_refusal_substring(r) for r in resp[key]]
            jud = judge[name][key]["flags"]
            n = len(sub)
            ks, kj = sum(sub), sum(jud)
            dis = sum(a != b for a, b in zip(sub, jud))
            entry[key] = {
                "n": n,
                "substring_refusals": ks,
                "substring_rate": round(ks / n, 3),
                "substring_ci95": wilson(ks, n),
                "judge_refusals": kj,
                "judge_rate": round(kj / n, 3),
                "judge_ci95": wilson(kj, n),
                "disagreement_rate": round(dis / n, 3),
                "judge_only": sum(j and not s for s, j in zip(sub, jud)),
                "substring_only": sum(s and not j for s, j in zip(sub, jud)),
                "flags": [{"substring": s, "judge": j} for s, j in zip(sub, jud)],
                "rejected_judge_v1_refusals": sum(v1[name][key]["flags"]),
                "empty_responses": sum(1 for r in resp[key] if not r.strip()),
            }
            if key != "harmful":
                entry[key]["examples"] = [
                    {"prompt": (p if isinstance(p, str) else p["prompt"]),
                     "response": lab.truncate(r, 120)}
                    for p, r in list(zip(sets_plain[key], resp[key]))[:5]
                ]
        entry["kl_harmless"] = kl[name]
        report["models"][name] = entry
    report["runtime_seconds_this_invocation"] = round(time.perf_counter() - t0, 1)
    dump(RESULTS_PATH, report)

    log("\n===== SUMMARY =====")
    for name, e in report["models"].items():
        log(f"{name:22s} harmful sub {e['harmful']['substring_rate']:.0%} judge {e['harmful']['judge_rate']:.0%} "
            f"(dis {e['harmful']['disagreement_rate']:.0%}) | harmless judge {e['harmless']['judge_rate']:.0%} "
            f"| borderline sub {e['borderline']['substring_rate']:.0%} judge {e['borderline']['judge_rate']:.0%} "
            f"| KL1 {e['kl_harmless']['first_token_kl']:.3f} KLfull {e['kl_harmless']['full_response_kl']:.3f}")


if __name__ == "__main__":
    main()
