# class_train: multi-label call classification with RoBERTa

Fine-tunes the local `roberta-base` checkpoint (`/Users/sam/models/roberta-base`,
loaded with `local_files_only=True`, nothing is downloaded) on
[`../class_data/`](../class_data/README.md) to predict the seven call-center
buckets as a **multi-label** problem. Results are written to
[`../class_results/`](../class_results/), checkpoints to
`../models/roberta-call-classifier/`.

The recipe mirrors [`../auth_train/`](../auth_train/README.md): frozen lower
layers whose outputs are precomputed once, a trainable last layer + head, Adam,
cosine with hard restarts, gradient clipping, per-epoch train/validation loss.
The differences are the backbone (RoBERTa), the loss (`BCEWithLogitsLoss`, one
logit per bucket) and multi-label metrics.

## CLI

Everything runs from the repo root through the project's `uv` environment.

```bash
# one shot: precompute -> train -> evaluate -> plot
./class_train/run.sh                 # full 100k calls, 20 epochs  (~6 min precompute, ~1 h training on MPS)
EPOCHS=50 ./class_train/run.sh       # longer schedule
LIMIT=20000 ./class_train/run.sh     # first 20k calls per split: ~6 GB cache, ~40 s/epoch
```

Step by step:

```bash
# 1. cache the frozen stack (embeddings + layers 1-11) as float16 memmaps in class_train/cache/
uv run python class_train/precompute_embeddings.py                     # all 100k calls, ~32 GB
uv run python class_train/precompute_embeddings.py --limit 20000       # subset
uv run python class_train/precompute_embeddings.py --trainable-layers 2  # leave the last 2 layers trainable

# 2. train the last layer + classification head on the cache
uv run python class_train/train.py                                     # 20 epochs, 4 hard restarts
uv run python class_train/train.py --epochs 50 --restart-cycles 5      # same schedule as the BERT run
uv run python class_train/train.py --pos-weight                        # weight positives per label (neg/pos)
uv run python class_train/train.py --lr 5e-5 --batch-size 64 --max-tokens-per-batch 16384
uv run python class_train/train.py --max-tokens-per-batch 4096       # smaller activations if MPS memory is tight
uv run python class_train/train.py --length-bucket 128               # even fewer distinct tensor shapes
uv run python class_train/train.py --epochs 2 --limit 2000             # quick sanity run on the cache

# 3. evaluate the saved checkpoint end to end on raw transcripts
uv run python class_train/evaluate.py                                  # best checkpoint, tunes thresholds on validation
uv run python class_train/evaluate.py --model models/roberta-call-classifier/final
uv run python class_train/evaluate.py --limit 2000                     # quicker

# 4. chart train / validation loss per epoch
uv run python auth_train/plot_losses.py --metrics class_results/epoch_metrics.json \
    --args class_results/training_args.json --out class_results/loss_curve.png \
    --title "roberta-base call classifier: loss per epoch" --loss-name "BCE-with-logits loss"
```

Every script accepts `--help`, `--device {auto,mps,cuda,cpu}` and the
`--cache` / `--results` / `--model-out` paths if you want to keep several runs
apart. Watch GPU load with `macmon` in a second terminal.

## Model (`model.py`)

`CallRobertaClassifier` wraps `RobertaForSequenceClassification` with
`problem_type="multi_label_classification"` and 7 outputs:

| Part | Status | Parameters |
|------|--------|-----------:|
| embeddings + encoder layers 1–11 | frozen, precomputed | 117.0 M |
| encoder layer 12 + `RobertaClassificationHead` (dense + tanh + out_proj on the `<s>` token) | trainable | 7.7 M |

`frozen_forward` and `head_forward` split the pass at layer 12 so the cache
holds exactly the input of the trainable part. Frozen layers run in `eval()`
mode during precompute, so no dropout noise is baked in. The checkpoint is
written with `save_pretrained` and also loads via
`RobertaForSequenceClassification.from_pretrained(...)`; apply `sigmoid` to the
logits and threshold per label (see `class_results/thresholds.json`).

## Training recipe (`train.py` defaults)

| Setting | Value |
|---------|-------|
| Loss | `torch.nn.BCEWithLogitsLoss` (optional `--pos-weight` = negatives/positives per label) |
| Optimizer | `torch.optim.Adam`, peak lr `3e-5` |
| LR schedule | cosine with hard restarts, 4 cycles, 5% linear warmup, stepped per batch |
| Gradient clipping | global norm 1.0 |
| Epochs | 20 (each epoch is ~16.6 M tokens, about 10x the BERT run) |
| Batch size | fixed shapes: lengths padded to `--length-bucket` 64, rows per bucket = min(32, `--max-tokens-per-batch` 8,192 // padded length); 64-row eval buckets |
| Max length | 512 (no call in the data exceeds it) |
| Checkpointing | best = lowest validation BCE loss; final = last epoch |

Per epoch the trainer records train loss, validation loss, micro-F1 and
macro-F1 at a 0.5 threshold, subset accuracy (exact label-set match), Hamming
loss, primary-label accuracy (argmax of the 7 logits against `primary_label`,
the multi-class view), learning rate, gradient norm and clipping rate.

## Outputs (`../class_results/`)

| File | Content |
|------|---------|
| `epoch_metrics.csv` / `.json` | one row per epoch with the metrics above |
| `train.log`, `training_args.json`, `summary.json` | run log, hyperparameters, best/final numbers |
| `test_metrics.json` | test micro/macro F1, subset accuracy, Hamming loss, per-label precision/recall/F1 at 0.5 and at tuned thresholds, primary-label accuracy with a 7x7 confusion matrix, breakdown by noise level |
| `thresholds.json` | per-label thresholds tuned on the validation split |
| `test_errors.jsonl` | every test call whose predicted label set differs from the truth, with probabilities |
| `loss_curve.png` | train / validation loss per epoch |

## GPU utilization

Measured with `macmon` during the full-data run on an M5 Max: GPU 95–99% busy at 1.6 GHz, about 62 W, CPU 3–6%, no swap. The "GPU utilization during training" section of the [repo README](../README.md) has the samples and the five design choices that keep the GPU saturated (MPS device, one-time frozen-stack precompute, cache-only training steps, length-bucketed batching, no host round-trips inside the step).

`class_train/cache/` and `../models/` are git-ignored.
