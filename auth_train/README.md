# auth_train: fine-tuning bert-base-uncased for authentication-chunk classification

Fine-tunes the local `bert-base-uncased` checkpoint on the chunks in
[`../auth_data/`](../auth_data/README.md) to predict `ID_AUTH` / `OTHER_AUTH` /
`NO_AUTH`, and saves the resulting model under `../models/`.

## Workflow

```bash
./auth_train/run.sh                 # all three steps, 50 epochs
EPOCHS=5 ./auth_train/run.sh        # shorter run
```

or step by step:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `precompute_embeddings.py` | Runs the **frozen** part of BERT (token embeddings + encoder layers 1–11) once over every chunk and caches the per-token hidden states to `cache/` as packed float16 memmaps (~3.1 GB for all splits). |
| 2 | `train.py` | Trains the **last** encoder layer + pooler + classifier for 50 epochs on the cached states. Records train and validation loss per epoch to `results/`. Saves the best (lowest validation loss) and final checkpoints to `../models/bert-auth-classifier/{best,final}/`. |
| 3 | `evaluate.py` | Loads the saved checkpoint and runs the **full** model on raw text from `auth_data/test.jsonl`, writing `results/test_metrics.json` and `results/test_errors.jsonl`. |
| 4 | `plot_losses.py` | Draws train / validation loss per epoch from `results/epoch_metrics.json` to `results/loss_curve.png`. |

All scripts run from the repo's `uv` environment (`uv run python auth_train/<script>.py`)
and take `--help`. Device is auto-selected (CUDA, then Apple MPS, then CPU).

## Model (`model.py`)

`AuthBertClassifier` wraps Hugging Face `BertForSequenceClassification`
(bert-base-uncased, 12 layers, pooler, linear head, 3 labels) and exposes the
forward pass in two halves:

* `frozen_forward(input_ids, attention_mask)`: embeddings + encoder layers 1–11.
  These parameters have `requires_grad=False`.
* `head_forward(hidden, attention_mask)`: encoder layer 12 + pooler + dropout +
  classifier. These are the only trainable parameters (7.68 M of 109.5 M).
* `forward(...)`: both halves, for inference.

Because layers 1–11 never change during training, their output is computed once
in step 1 (in `eval()` mode, so no dropout noise is baked in) and the training
loop only runs `head_forward` on the cache. Cached-path and full-model losses
were verified to match to six decimal places. `--trainable-layers N` on the
precompute script unfreezes the last N layers instead of just one.

The checkpoint is written with `save_pretrained`, so it also loads directly with
`BertForSequenceClassification.from_pretrained("models/bert-auth-classifier/best")`;
label ids are in `config.json` (`ID_AUTH=0, NO_AUTH=1, OTHER_AUTH=2`).

## Training recipe (`train.py` defaults)

| Setting | Value |
|---------|-------|
| Optimizer | `torch.optim.Adam`, peak lr `3e-5` |
| Loss | `torch.nn.CrossEntropyLoss` |
| LR schedule | cosine with hard restarts (`transformers.get_cosine_with_hard_restarts_schedule_with_warmup`), 5 cycles over 50 epochs, 5% linear warmup, stepped per batch |
| Gradient clipping | global norm clipped to 1.0 (`clip_grad_norm_`) |
| Epochs | 50 |
| Batch size | 32 train / 128 eval, length-bucketed to minimise padding |
| Max sequence length | 200 tokens incl. `[CLS]`/`[SEP]` (nothing in the data is truncated) |
| Seed | 42 |

## Results (`results/`)

* `epoch_metrics.csv` / `epoch_metrics.json`: one row per epoch with
  `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1`, learning rate at
  the end of the epoch, mean gradient norm, fraction of steps that were
  clipped, and wall time. Written after every epoch, so a partial run is still
  recorded.
* `train.log`: the same information as a log, plus device and parameter counts.
* `training_args.json`: the exact hyperparameters used.
* `summary.json`: best epoch, best validation loss, final numbers, total time.
* `test_metrics.json`: loss, accuracy, macro-F1, per-class precision / recall /
  F1 and the confusion matrix on the test split, from the best checkpoint.
* `test_errors.jsonl`: every misclassified test chunk with its probabilities.
* `loss_curve.png`: train / validation loss per epoch (log scale), from `plot_losses.py`.

`cache/` and `../models/` are git-ignored (3.1 GB and ~440 MB per checkpoint);
rerun the workflow to recreate them.

### Outcome of the shipped run (2026-10-14, Apple MPS, 831 s for 50 epochs)

| | train loss | val loss | val acc | val macro-F1 |
|---|---:|---:|---:|---:|
| epoch 1 | 0.507 | 0.0236 | 0.994 | 0.995 |
| epoch 2 | 0.0049 | 0.0004 | 1.000 | 1.000 |
| epoch 10 | 1.6e-5 | 8e-6 | 1.000 | 1.000 |
| epoch 50 | ~0 | ~0 | 1.000 | 1.000 |

Best checkpoint: epoch 33. Test split (best checkpoint, full model on raw
text): accuracy 1.000, macro-F1 1.000, zero errors across 2,539 chunks.
The synthetic data is templated, so the task saturates within two epochs; the
50-epoch recipe is in place for harder or real transcripts, and the remaining
epochs mainly show the hard-restart cycles in `lr_end_of_epoch`.
