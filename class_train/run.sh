#!/usr/bin/env bash
# Full workflow: precompute frozen RoBERTa hidden states -> fine-tune last layer + head (BCEWithLogits)
# -> evaluate on test -> plot losses. Results go to class_results/, checkpoints to models/roberta-call-classifier/.
#
#   ./class_train/run.sh                    # full data, 20 epochs
#   EPOCHS=50 ./class_train/run.sh          # longer run
#   LIMIT=20000 ./class_train/run.sh        # first 20k calls per split (smaller cache, faster)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL_DIR="${MODEL_DIR:-/Users/sam/models/roberta-base}"
EPOCHS="${EPOCHS:-20}"
LIMIT="${LIMIT:-0}"

echo "== 1/4 precompute embeddings (frozen layers) =="
uv run python class_train/precompute_embeddings.py --model-dir "$MODEL_DIR" --limit "$LIMIT"

echo "== 2/4 fine-tune last layer + head for $EPOCHS epochs =="
uv run python class_train/train.py --epochs "$EPOCHS"

echo "== 3/4 evaluate best checkpoint on validation (threshold tuning) and test =="
uv run python class_train/evaluate.py

echo "== 4/4 plot train / validation loss per epoch =="
uv run python auth_train/plot_losses.py --metrics class_results/epoch_metrics.json \
    --args class_results/training_args.json --out class_results/loss_curve.png \
    --title "roberta-base call classifier: loss per epoch" --loss-name "BCE-with-logits loss"
