# Heretic ARA run (CPU, reduced budget)

This folder reproduces the Heretic 2.1 **ARA** (Arbitrary-Rank Ablation)
run — the default modifier in Heretic 2.x — and compares it with the
manual script and the Abliteration modifier (parent folder).

```bash
# from the repo root, with the same venv
pip install -r heretic/requirements.txt
pip uninstall -y torchvision   # see the parent heretic/README.md

./heretic/ara/run_ara.sh
python compare_models.py
```

- `config.toml` — ARA modifier, 12 trials / 5 startup, LoRA rank 16
  (default 50), 3 L-BFGS steps (default 5), 16+16 scorer prompts, 48+48
  behaviour prompts, batch size 1. Every departure from the package
  default is marked `CHANGED` in the config comments.
- `run_ara.sh` — non-interactive CLI (`--trial-index=0`, merge + save).
- `results.json` — Heretic's own study scores on the mlabonne slices.
- The four-way held-out comparison lives in `../comparison.json`.

The merged model is written to `../../model-ara/` and is gitignored.
