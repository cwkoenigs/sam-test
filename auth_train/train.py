#!/usr/bin/env python3
"""
Step 2: fine-tune the last BERT layer + classifier on the cached hidden states.

* model        auth_train/model.py (AuthBertClassifier: bert-base-uncased, frozen
               embeddings + first 11 encoder layers; last layer, pooler and
               classifier trainable)
* input        auth_train/cache/*.hidden.npy produced by precompute_embeddings.py
* optimizer    torch.optim.Adam
* loss         cross-entropy
* schedule     cosine with hard restarts (+ linear warmup), stepped per batch
* clipping     global gradient-norm clipping
* epochs       50 (default); train loss and validation loss recorded per epoch
* outputs      auth_train/results/epoch_metrics.{csv,json}, train.log,
               models/bert-auth-classifier/{best,final}/

Usage:
    uv run python auth_train/train.py            # full run, defaults below
    uv run python auth_train/train.py --epochs 2 --limit 512   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
from torch import nn
from transformers import BertTokenizerFast, get_cosine_with_hard_restarts_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LABELS, AuthBertClassifier, pick_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# --------------------------------------------------------------------------- #
# cached-hidden-state dataset
# --------------------------------------------------------------------------- #


class CachedHiddenSplit:
    """Packed float16 hidden states on disk + per-example offsets/lengths/labels."""

    def __init__(self, cache_dir: str, split: str, limit: int = 0) -> None:
        idx = np.load(os.path.join(cache_dir, f"{split}.index.npz"), allow_pickle=False)
        self.offsets = idx["offsets"]
        self.lengths = idx["lengths"]
        self.labels = idx["labels"]
        self.ids = idx["ids"]
        if limit:
            self.offsets, self.lengths = self.offsets[:limit], self.lengths[:limit]
            self.labels, self.ids = self.labels[:limit], self.ids[:limit]
        self.hidden = np.load(os.path.join(cache_dir, f"{split}.hidden.npy"), mmap_mode="r")
        self.hidden_size = self.hidden.shape[1]

    def __len__(self) -> int:
        return len(self.labels)

    def batch(self, indices: Sequence[int]) -> Dict[str, torch.Tensor]:
        max_len = int(self.lengths[indices].max())
        h = np.zeros((len(indices), max_len, self.hidden_size), dtype=np.float32)
        mask = np.zeros((len(indices), max_len), dtype=np.int64)
        for j, i in enumerate(indices):
            n = int(self.lengths[i])
            o = int(self.offsets[i])
            h[j, :n] = self.hidden[o:o + n]
            mask[j, :n] = 1
        return {
            "hidden": torch.from_numpy(h),
            "attention_mask": torch.from_numpy(mask),
            "labels": torch.from_numpy(self.labels[indices].astype(np.int64)),
        }

    def batches(self, batch_size: int, shuffle: bool, rng: random.Random) -> Iterator[Dict[str, torch.Tensor]]:
        """Length-bucketed batching: shuffle, sort within mega-buckets, shuffle batches."""
        order = list(range(len(self)))
        if shuffle:
            rng.shuffle(order)
            mega = batch_size * 32
            buckets = [sorted(order[i:i + mega], key=lambda k: self.lengths[k]) for i in range(0, len(order), mega)]
            batches = [b[i:i + batch_size] for b in buckets for i in range(0, len(b), batch_size)]
            rng.shuffle(batches)
        else:
            order.sort(key=lambda k: self.lengths[k])
            batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        for b in batches:
            yield self.batch(np.array(b))

    def num_batches(self, batch_size: int) -> int:
        return math.ceil(len(self) / batch_size)


# --------------------------------------------------------------------------- #
# train / eval loops
# --------------------------------------------------------------------------- #


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def evaluate(model: AuthBertClassifier, data: CachedHiddenSplit, loss_fn: nn.Module,
             device: torch.device, batch_size: int) -> Dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for b in data.batches(batch_size, shuffle=False, rng=random.Random(0)):
        logits = model.head_forward(b["hidden"].to(device), b["attention_mask"].to(device))
        labels = b["labels"].to(device)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        n += len(labels)
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(labels.cpu().numpy())
    y_pred, y_true = np.concatenate(preds), np.concatenate(trues)
    return {
        "loss": total_loss / n,
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": macro_f1(y_true, y_pred, len(LABELS)),
    }


def write_metrics(results_dir: str, history: List[Dict]) -> None:
    with open(os.path.join(results_dir, "epoch_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(results_dir, "epoch_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(HERE, "cache"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--model-out", default=os.path.join(REPO, "models", "bert-auth-classifier"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5, help="peak learning rate for Adam")
    ap.add_argument("--warmup-ratio", type=float, default=0.05, help="fraction of total steps for linear warmup")
    ap.add_argument("--restart-cycles", type=int, default=5, help="number of cosine hard restarts over training")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only the first N examples of each split")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.results, exist_ok=True)
    log_path = os.path.join(args.results, "train.log")
    log_f = open(log_path, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    with open(os.path.join(args.cache, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["labels"] != LABELS:
        raise SystemExit(f"label order mismatch between cache {meta['labels']} and model.py {LABELS}")

    train_data = CachedHiddenSplit(args.cache, "train", args.limit)
    val_data = CachedHiddenSplit(args.cache, "validation", args.limit)
    log(f"device={device} train={len(train_data)} validation={len(val_data)} "
        f"cache_model={meta['model_dir']} frozen_layers={meta['n_frozen_layers']}")

    model = AuthBertClassifier(meta["model_dir"], trainable_layers=meta["trainable_layers"]).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(meta["model_dir"])
    counts = model.count_parameters()
    log(f"parameters total={counts['total']:,} trainable={counts['trainable']:,} frozen={counts['frozen']:,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    steps_per_epoch = train_data.num_batches(args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=args.restart_cycles)
    log(f"optimizer=Adam lr={args.lr} schedule=cosine_hard_restarts cycles={args.restart_cycles} "
        f"warmup_steps={warmup_steps} total_steps={total_steps} max_grad_norm={args.max_grad_norm}")

    train_args = vars(args) | {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
                               "warmup_steps": warmup_steps, "device": str(device), "optimizer": "Adam",
                               "loss": "CrossEntropyLoss", "scheduler": "cosine_with_hard_restarts_with_warmup"}
    with open(os.path.join(args.results, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(train_args, f, indent=2)

    history: List[Dict] = []
    best_val = float("inf")
    best_epoch = -1
    rng = random.Random(args.seed)
    step = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen, grad_norms = 0.0, 0, []
        for b in train_data.batches(args.batch_size, shuffle=True, rng=rng):
            logits = model.head_forward(b["hidden"].to(device), b["attention_mask"].to(device))
            labels = b["labels"].to(device)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
            grad_norms.append(float(gn))
            optimizer.step()
            scheduler.step()
            step += 1
            running += loss.item() * len(labels)
            n_seen += len(labels)
        train_loss = running / n_seen
        val = evaluate(model, val_data, loss_fn, device, args.eval_batch_size)
        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val["loss"], 6),
            "val_accuracy": round(val["accuracy"], 6),
            "val_macro_f1": round(val["macro_f1"], 6),
            "lr_end_of_epoch": scheduler.get_last_lr()[0],
            "mean_grad_norm": round(float(np.mean(grad_norms)), 4),
            "clipped_frac": round(float(np.mean([g > args.max_grad_norm for g in grad_norms])), 4),
            "epoch_seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        write_metrics(args.results, history)
        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch = val["loss"], epoch
            model.save(os.path.join(args.model_out, "best"), tokenizer,
                       {"epoch": epoch, "val_loss": val["loss"], "val_accuracy": val["accuracy"],
                        "val_macro_f1": val["macro_f1"]})
        log(f"epoch {epoch:3d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} lr={rec['lr_end_of_epoch']:.2e} "
            f"clipped={rec['clipped_frac']:.0%} {rec['epoch_seconds']}s{' *best*' if improved else ''}")

    model.save(os.path.join(args.model_out, "final"), tokenizer,
               {"epoch": args.epochs, "val_loss": history[-1]["val_loss"], "best_epoch": best_epoch})
    summary = {
        "epochs": args.epochs, "best_epoch": best_epoch, "best_val_loss": best_val,
        "final_train_loss": history[-1]["train_loss"], "final_val_loss": history[-1]["val_loss"],
        "final_val_accuracy": history[-1]["val_accuracy"], "final_val_macro_f1": history[-1]["val_macro_f1"],
        "total_seconds": round(time.time() - t_start, 1),
        "model_best": os.path.join(args.model_out, "best"), "model_final": os.path.join(args.model_out, "final"),
    }
    with open(os.path.join(args.results, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"done best_epoch={best_epoch} best_val_loss={best_val:.4f} total={summary['total_seconds']}s")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
