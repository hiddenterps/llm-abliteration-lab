#!/usr/bin/env bash
# Run Heretic 2.1 with the default ARA modifier on Qwen2.5-0.5B-Instruct.
# Non-interactive: pick the first Pareto-front trial and save a merged model.
#
# Note: heretic-llm 2.1.0 may pull a torchvision wheel incompatible with
# torch 2.15+cpu. Uninstall torchvision if import fails with nms errors.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
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
  --save-directory="$ROOT/model-ara"
