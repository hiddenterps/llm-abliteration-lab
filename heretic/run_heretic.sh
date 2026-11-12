#!/usr/bin/env bash
# Run Heretic (classic Abliteration modifier) on Qwen2.5-0.5B-Instruct.
# Non-interactive: pick the first Pareto-front trial and save a merged model.
#
# Note: heretic-llm 2.1.0 pulls a torchvision wheel that is incompatible with
# torch 2.15+cpu ("operator torchvision::nms does not exist"). Uninstall
# torchvision before running, or install a CPU wheel built for the same torch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$(dirname "$0")"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

exec "$ROOT/.venv/bin/heretic" \
  --model=Qwen/Qwen2.5-0.5B-Instruct \
  --n-trials=12 \
  --n-startup-trials=5 \
  --seed=0 \
  --batch-size=1 \
  --max-batch-size=2 \
  --max-response-length=48 \
  --device-map=cpu \
  --checkpoint-action=restart \
  --trial-index=0 \
  --model-action=save \
  --export-strategy=merge \
  --save-directory="$ROOT/model-heretic"
