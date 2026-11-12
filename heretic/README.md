# Heretic run (CPU, reduced budget)

This folder reproduces the Heretic 2.1 comparison against the manual
abliteration in the parent directory.

```bash
# from the repo root, with the same venv
pip install -r heretic/requirements.txt
pip uninstall -y torchvision   # see note below
./heretic/run_heretic.sh
python compare_models.py
```

- `config.toml` — Abliteration modifier (not the default ARA), 12 trials /
  5 startup, 16+16 scorer prompts, 48+48 direction prompts, batch size 1.
- `run_heretic.sh` — non-interactive CLI (`--trial-index=0`, merge + save).
- `results.json` — Heretic's own study scores (keyword refusals + first-token
  KL on the mlabonne slices).
- `comparison.json` — held-out comparison vs. the manual script, using the
  same detector and prompt split. No harmful completions are stored.

The merged model is written to `../model-heretic/` and is gitignored.

**torchvision note.** `heretic-llm` 2.1.0 may install a `torchvision` wheel
that does not match the official CPU build of PyTorch 2.15, and then fails
with `operator torchvision::nms does not exist`. Uninstalling `torchvision`
is enough for this text-only model.
