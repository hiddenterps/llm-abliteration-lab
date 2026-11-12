#!/usr/bin/env python3
"""
Compare the original model, the manually abliterated model, and the Heretic
model on the same held-out prompts used by abliterate.py.

Reports:
  * refusal rate (same substring detector as the manual script)
  * first-token KL divergence vs. the original, on harmless prompts
    (same convention as Heretic: KL(original || modified), batch mean)
  * short harmless sample outputs

Harmful completions are not written to the output JSON, only yes/no flags.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import abliterate as lab

ROOT = Path(__file__).resolve().parent
ORIGINAL_ID = lab.MODEL_ID
MANUAL_DIR = ROOT / "model-abliterated"
HERETIC_DIR = ROOT / "model-heretic"
OUT_PATH = ROOT / "heretic" / "comparison.json"


def load(path_or_id: str):
    tokenizer = AutoTokenizer.from_pretrained(path_or_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path_or_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def first_token_logprobs(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """Log-probs of the first generated token for each prompt, shape (n, vocab)."""
    rows = []
    for prompt in prompts:
        inputs = lab.tokenize(tokenizer, prompt, next(model.parameters()).device)
        out = model(**inputs, use_cache=False)
        logits = out.logits[0, -1, :].float()
        rows.append(F.log_softmax(logits, dim=-1).cpu())
    return torch.stack(rows, dim=0)


def kl_original_vs_modified(base_logprobs: torch.Tensor, other_logprobs: torch.Tensor) -> float:
    # F.kl_div(input, target, log_target=True) = KL(target || input)
    # so input=modified, target=original → KL(original || modified).
    return F.kl_div(
        other_logprobs, base_logprobs, reduction="batchmean", log_target=True
    ).item()


def unload(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    t0 = time.perf_counter()
    torch.set_grad_enabled(False)

    eval_r, eval_i = lab.get_eval_splits()
    # A slightly larger harmless set makes the KL number less noisy, without
    # changing the held-out refusal evaluation.
    harmless_for_kl = list(lab.PROMPTS_HARMLESS)

    if not MANUAL_DIR.exists():
        raise SystemExit(f"Missing {MANUAL_DIR}; run abliterate.py first.")
    if not HERETIC_DIR.exists():
        raise SystemExit(f"Missing {HERETIC_DIR}; run heretic/run_heretic.sh first.")

    results = {"eval_refusal_n": len(eval_r), "harmless_kl_n": len(harmless_for_kl)}

    lab.log("Loading ORIGINAL model…")
    model, tokenizer = load(ORIGINAL_ID)
    lab.log("  refusal eval:")
    orig_r = lab.evaluate_prompts(model, tokenizer, eval_r, label="orig-r")
    lab.log("  harmless samples:")
    orig_i = lab.evaluate_prompts(model, tokenizer, eval_i, label="orig-i")
    lab.log("  first-token logprobs on harmless prompts (KL baseline)…")
    base_lp = first_token_logprobs(model, tokenizer, harmless_for_kl)
    unload(model)
    results["original"] = {
        "refusal_rate": orig_r["rate"],
        "refusal_count": f"{orig_r['n_refusals']}/{orig_r['n']}",
        "refusal_flags": [d["refused"] for d in orig_r["details"]],
        "kl_vs_original": 0.0,
    }

    def run_edited(tag: str, path: Path) -> dict:
        lab.log(f"\nLoading {tag} from {path}…")
        model, tokenizer = load(str(path))
        lab.log("  refusal eval:")
        ev_r = lab.evaluate_prompts(model, tokenizer, eval_r, label=f"{tag}-r")
        lab.log("  harmless samples:")
        ev_i = lab.evaluate_prompts(model, tokenizer, eval_i, label=f"{tag}-i")
        lab.log("  first-token KL vs original…")
        lp = first_token_logprobs(model, tokenizer, harmless_for_kl)
        kl = kl_original_vs_modified(base_lp, lp)
        unload(model)
        lab.log(f"  {tag}: refusal={ev_r['rate']:.0%}  KL={kl:.4f}")
        return {
            "refusal_rate": ev_r["rate"],
            "refusal_count": f"{ev_r['n_refusals']}/{ev_r['n']}",
            "refusal_flags": [d["refused"] for d in ev_r["details"]],
            "kl_vs_original": kl,
            "harmless_examples": [
                {
                    "prompt": eval_i[k],
                    "original": orig_i["details"][k]["snippet"],
                    tag: ev_i["details"][k]["snippet"],
                }
                for k in range(len(eval_i))
            ],
        }

    results["manual"] = run_edited("manual", MANUAL_DIR)
    results["heretic"] = run_edited("heretic", HERETIC_DIR)
    results["runtime_seconds"] = round(time.perf_counter() - t0, 1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    lab.log("\n===== COMPARISON =====")
    lab.log(
        f"{'model':<12} {'refusal':<12} {'KL vs orig (harmless, first token)':<36}"
    )
    for name in ("original", "manual", "heretic"):
        r = results[name]
        lab.log(
            f"{name:<12} {r['refusal_count']:<12} {r['kl_vs_original']:.4f}"
        )
    lab.log(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
