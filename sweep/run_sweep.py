#!/usr/bin/env python3
"""Phase 3 - layer sweep for manual abliteration.

For every layer l of Qwen2.5-0.5B-Instruct:
  1. take the refusal direction r_l = normalize(mean(harmful) - mean(harmless))
     from the output of layer l at the last prompt token, using the SAME 40+40
     training prompts as abliterate.py;
  2. orthogonalize embedding, o_proj and down_proj against r_l (exactly as
     abliterate.py does), on a fresh copy of the original weights;
  3. measure refusals on a 30-prompt held-out harmful subset (from
     eval/prompts.json, greedy, 64 new tokens; substring detector + the
     Qwen2.5-3B message-only judge) and first-token KL vs the original on the 100
     held-out harmless prompts.

Outputs sweep/results.json and sweep/layer_sweep.png.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from common import (JUDGE_ID, MODELS, ROOT, dump, first_token_logprobs,  # noqa: E402
                    generate_batch, is_refusal_substring, judge_refusals, kl_batchmean,
                    lab, load, log)


HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
N_HARMFUL = 30
MAX_NEW = 64


def train_split():
    refusal, harmless = list(lab.PROMPTS_RECUSA), list(lab.PROMPTS_HARMLESS)
    g = torch.Generator().manual_seed(0)
    refusal = [refusal[i] for i in torch.randperm(len(refusal), generator=g).tolist()]
    harmless = [harmless[i] for i in torch.randperm(len(harmless), generator=g).tolist()]
    return refusal[:lab.N_TRAIN], harmless[:lab.N_TRAIN]


def snapshot(model):
    keep = {"embed": model.model.embed_tokens.weight.detach().clone()}
    for i, layer in enumerate(model.model.layers):
        keep[f"o{i}"] = layer.self_attn.o_proj.weight.detach().clone()
        keep[f"d{i}"] = layer.mlp.down_proj.weight.detach().clone()
    return keep


@torch.no_grad()
def restore(model, keep):
    model.model.embed_tokens.weight.copy_(keep["embed"])
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.weight.copy_(keep[f"o{i}"])
        layer.mlp.down_proj.weight.copy_(keep[f"d{i}"])


def main() -> None:
    t0 = time.perf_counter()
    sets = json.loads((ROOT / "eval" / "prompts.json").read_text())
    harmful = [p["prompt"] for p in sets["harmful"]]
    # spread the 30 prompts across the whole set (and so across categories)
    step = len(harmful) / N_HARMFUL
    harmful = [harmful[int(i * step)] for i in range(N_HARMFUL)]
    harmless = sets["harmless"]

    cache_path = CACHE / "sweep_raw.json"
    raw = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    model, tok = load(MODELS["original"])
    n_layers = model.config.num_hidden_layers
    train_r, train_i = train_split()
    acts_r = lab.collect_activations(model, tok, train_r)
    acts_i = lab.collect_activations(model, tok, train_i)
    dirs = lab.refusal_directions(acts_r, acts_i)
    norms = {l: (acts_r[l].mean(0) - acts_i[l].mean(0)).norm().item() for l in dirs}
    del acts_r, acts_i
    saved = torch.load(ROOT / "refusal_direction.pt")
    saved_vec = saved["direction"] if isinstance(saved, dict) else saved
    cos12 = F.cosine_similarity(dirs[12].float(), saved_vec.float().flatten(), dim=0).item()
    log(f"cosine(sweep layer-12 direction, refusal_direction.pt) = {cos12:.4f}")

    base_first = first_token_logprobs(model, tok, harmless)
    keep = snapshot(model)

    if "baseline" not in raw:
        raw["baseline"] = {"responses": generate_batch(model, tok, harmful, MAX_NEW, 16, "baseline")}
        dump(cache_path, raw)
    for l in range(n_layers):
        key = str(l)
        if key in raw:
            continue
        restore(model, keep)
        lab.orthogonalize_weights(model, dirs[l])
        resp = generate_batch(model, tok, harmful, MAX_NEW, 16, f"layer {l}")
        kl = kl_batchmean(base_first, first_token_logprobs(model, tok, harmless))
        raw[key] = {"responses": resp, "first_token_kl": kl}
        log(f"layer {l:2d}: substring refusals {sum(map(is_refusal_substring, resp))}/{len(resp)}  KL {kl:.4f}")
        dump(cache_path, raw)
    del model, keep

    order = ["baseline"] + [str(l) for l in range(n_layers)]
    jflags, _ = judge_refusals([r for k in order for r in raw[k]["responses"]])

    rows = []
    for i, k in enumerate(order):
        resp = raw[k]["responses"]
        sub = [is_refusal_substring(r) for r in resp]
        jud = jflags[i * N_HARMFUL:(i + 1) * N_HARMFUL]
        rows.append({
            "layer": None if k == "baseline" else int(k),
            "substring_refusal_rate": round(sum(sub) / len(sub), 3),
            "judge_refusal_rate": round(sum(jud) / len(jud), 3),
            "first_token_kl": 0.0 if k == "baseline" else round(raw[k]["first_token_kl"], 4),
            "direction_norm": None if k == "baseline" else round(norms[int(k)], 3),
        })
    out = {"config": {"n_harmful": N_HARMFUL, "n_harmless_kl": len(harmless), "max_new_tokens": MAX_NEW,
                      "train_prompts": "abliterate.py 40 + 40", "judge": JUDGE_ID,
                      "cos_layer12_vs_saved_direction": round(cos12, 4)},
           "rows": rows, "runtime_seconds_this_invocation": round(time.perf_counter() - t0, 1)}
    dump(HERE / "results.json", out)
    plot(rows)
    for r in rows:
        log(str(r))


def plot(rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = rows[0]
    layers = [r["layer"] for r in rows[1:]]
    fig, ax1 = plt.subplots(figsize=(10, 5.2), dpi=140)
    ax1.plot(layers, [100 * r["judge_refusal_rate"] for r in rows[1:]], "o-", color="#c0392b",
             label="refusal % (LLM judge)")
    ax1.plot(layers, [100 * r["substring_refusal_rate"] for r in rows[1:]], "s--", color="#e67e22",
             alpha=0.8, label="refusal % (substring)")
    ax1.axhline(100 * base["judge_refusal_rate"], color="#c0392b", ls=":", lw=1,
                label=f"original, judge ({100 * base['judge_refusal_rate']:.0f}%)")
    ax1.set_xlabel("layer the refusal direction is taken from")
    ax1.set_ylabel("refusal rate on 30 held-out harmful prompts (%)")
    ax1.set_ylim(-3, 103)
    ax1.set_xticks(layers)
    ax2 = ax1.twinx()
    ax2.plot(layers, [r["first_token_kl"] for r in rows[1:]], "^-", color="#2c3e50",
             label="first-token KL vs original (harmless)")
    ax2.set_yscale("log")
    ax2.set_ylabel("first-token KL (log scale, lower = less drift)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax1.axvline(12, color="gray", lw=0.8, alpha=0.5)
    ax1.text(12.15, 2, "layer used by\nabliterate.py", fontsize=7, color="gray")
    plt.title("Qwen2.5-0.5B-Instruct: manual abliteration, one layer's direction at a time")
    fig.tight_layout()
    fig.savefig(HERE / "layer_sweep.png")


if __name__ == "__main__":
    main()
