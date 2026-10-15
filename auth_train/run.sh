#!/usr/bin/env bash
# Full workflow: precompute frozen-BERT hidden states -> fine-tune last layer -> evaluate on test.
# Run from anywhere; uses the repo's uv environment.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="${MODEL_DIR:-/Users/sam/models/bert-base-uncased}"
EPOCHS="${EPOCHS:-50}"

echo "== 1/4 precompute embeddings (frozen layers) =="
uv run python auth_train/precompute_embeddings.py --model-dir "$MODEL_DIR" "$@"

echo "== 2/4 fine-tune last layer + classifier for $EPOCHS epochs =="
uv run python auth_train/train.py --epochs "$EPOCHS"

echo "== 3/4 evaluate best checkpoint on test split =="
uv run python auth_train/evaluate.py

echo "== 4/4 plot train / validation loss per epoch =="
uv run python auth_train/plot_losses.py
