#!/usr/bin/env python3
"""
Step 2: fine-tune the last RoBERTa layer + classification head for multi-label
call classification on the cached hidden states.

Same recipe as auth_train/train.py, with a multi-label head:
* model        class_train/model.py (roberta-base, embeddings + layers 1-11 frozen)
* loss         torch.nn.BCEWithLogitsLoss (one logit per bucket)
* optimizer    torch.optim.Adam
* schedule     cosine with hard restarts + linear warmup, stepped per batch
* clipping     global gradient-norm clipping
* per epoch    train loss, validation loss, micro/macro F1 @0.5, subset accuracy,
               primary-label accuracy (argmax of the logits vs primary_label)
* outputs      class_results/epoch_metrics.{csv,json}, train.log, training_args.json,
               summary.json; models/roberta-call-classifier/{best,final}/

Usage:
    uv run python class_train/train.py                        # defaults (20 epochs)
    uv run python class_train/train.py --epochs 50 --restart-cycles 5
    uv run python class_train/train.py --epochs 2 --limit 2000   # smoke test
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
from typing import Dict, Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from transformers import RobertaTokenizerFast, get_cosine_with_hard_restarts_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LABELS, CallRobertaClassifier, pick_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


class CachedHiddenSplit:
    """
    Packed float16 hidden states on disk + per-example offsets/lengths/labels.

    The cache is read with positional reads (os.pread), NOT a memory map. A
    memmap leaves every page touched during an epoch resident in the process,
    and on Apple Silicon the MPS allocator counts that against its memory
    limit ("other allocations: 51 GiB" in the OOM message), so a 25 GB train
    cache kills the run partway through the first epoch. pread keeps the
    process footprint at one batch.
    """

    def __init__(self, cache_dir: str, split: str, limit: int = 0) -> None:
        idx = np.load(os.path.join(cache_dir, f"{split}.index.npz"), allow_pickle=False)
        self.offsets, self.lengths = idx["offsets"], idx["lengths"]
        self.labels, self.primary, self.ids = idx["label_vectors"], idx["primary_ids"], idx["ids"]
        if limit:
            self.offsets, self.lengths = self.offsets[:limit], self.lengths[:limit]
            self.labels, self.primary, self.ids = self.labels[:limit], self.primary[:limit], self.ids[:limit]
        path = os.path.join(cache_dir, f"{split}.hidden.npy")
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            shape, _, dtype = np.lib.format._read_array_header(f, version)
            self.data_offset = f.tell()
        self.hidden_size = int(shape[1])
        self.row_bytes = self.hidden_size * np.dtype(dtype).itemsize
        self.dtype = dtype
        self.fd = os.open(path, os.O_RDONLY)

    def __len__(self) -> int:
        return len(self.labels)

    def _read(self, i: int) -> np.ndarray:
        n, o = int(self.lengths[i]), int(self.offsets[i])
        buf = os.pread(self.fd, n * self.row_bytes, self.data_offset + o * self.row_bytes)
        return np.frombuffer(buf, dtype=self.dtype).reshape(n, self.hidden_size)

    def batch(self, indices: Sequence[int], pad_len: int, pad_rows: int) -> Dict[str, torch.Tensor]:
        """
        One batch with a FIXED shape (pad_rows, pad_len, hidden). Rows beyond
        len(indices) are filled with repeated real examples and flagged invalid
        so they contribute nothing to the loss or the metrics.
        """
        h = np.zeros((pad_rows, pad_len, self.hidden_size), dtype=np.float32)
        mask = np.zeros((pad_rows, pad_len), dtype=np.int64)
        valid = np.zeros(pad_rows, dtype=np.float32)
        rows = list(indices) + [indices[j % len(indices)] for j in range(pad_rows - len(indices))]
        for j, i in enumerate(rows):
            n = int(self.lengths[i])
            h[j, :n] = self._read(i)
            mask[j, :n] = 1
        valid[: len(indices)] = 1.0
        rows_np = np.array(rows)
        return {"hidden": torch.from_numpy(h), "attention_mask": torch.from_numpy(mask),
                "labels": torch.from_numpy(self.labels[rows_np]), "primary": torch.from_numpy(self.primary[rows_np]),
                "valid": torch.from_numpy(valid)}

    def make_batches(self, batch_size: int, max_tokens: int, bucket: int, shuffle: bool,
                     rng: random.Random) -> List[Tuple[List[int], int, int]]:
        """
        Fixed-shape, length-bucketed batches: (indices, pad_len, pad_rows).

        Sequence lengths are rounded up to a multiple of `bucket` tokens and
        every batch in a bucket has the same number of rows
        min(batch_size, max_tokens // pad_len), so an epoch uses at most
        512 / bucket distinct tensor shapes. This matters on Apple MPS, where
        every distinct shape compiles and caches a new kernel graph in Metal
        memory that is never released: with per-batch padding the driver grows
        by ~0.2 GB per unique shape until "MPS backend out of memory ... other
        allocations: 57 GiB". Partial batches are padded with masked rows.
        """
        by_bucket: Dict[int, List[int]] = {}
        for k in range(len(self)):
            pad_len = int(math.ceil(int(self.lengths[k]) / bucket) * bucket)
            by_bucket.setdefault(pad_len, []).append(k)
        batches: List[Tuple[List[int], int, int]] = []
        for pad_len, idxs in sorted(by_bucket.items()):
            rows = max(1, min(batch_size, max_tokens // pad_len))
            if shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), rows):
                batches.append((idxs[i:i + rows], pad_len, rows))
        if shuffle:
            rng.shuffle(batches)
        return batches

    def batches(self, batch_size: int, max_tokens: int, bucket: int, shuffle: bool,
                rng: random.Random) -> Iterator[Dict[str, torch.Tensor]]:
        for idxs, pad_len, rows in self.make_batches(batch_size, max_tokens, bucket, shuffle, rng):
            yield self.batch(idxs, pad_len, rows)

    def num_batches(self, batch_size: int, max_tokens: int, bucket: int) -> int:
        return len(self.make_batches(batch_size, max_tokens, bucket, shuffle=False, rng=random.Random(0)))

    def shapes(self, batch_size: int, max_tokens: int, bucket: int) -> List[Tuple[int, int]]:
        return sorted({(rows, pad_len) for _, pad_len, rows in
                       self.make_batches(batch_size, max_tokens, bucket, shuffle=False, rng=random.Random(0))})


def masked_loss(loss_fn: nn.Module, logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """BCEWithLogits per element, averaged over labels, then over valid rows only."""
    per_row = loss_fn(logits, labels).mean(1)
    return (per_row * valid).sum() / valid.sum().clamp_min(1.0)


def device_memory_gb(device: torch.device) -> float:
    """Total memory the GPU driver holds for this process (what the MPS limit is checked against)."""
    if device.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 1e9, 2)
    if device.type == "cuda":
        return round(torch.cuda.memory_reserved() / 1e9, 2)
    return 0.0


def free_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, primary_true: np.ndarray,
                       threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = (y_pred & (y_true == 1)).sum(0)
    fp = (y_pred & (y_true == 0)).sum(0)
    fn = ((1 - y_pred) & (y_true == 1)).sum(0)
    per_f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    return {
        "micro_f1": float(micro),
        "macro_f1": float(per_f1.mean()),
        "subset_accuracy": float((y_pred == y_true).all(1).mean()),
        "hamming_loss": float((y_pred != y_true).mean()),
        "primary_accuracy": float((y_prob.argmax(1) == primary_true).mean()),
    }


@torch.no_grad()
def evaluate(model: CallRobertaClassifier, data: CachedHiddenSplit, loss_fn: nn.Module, device: torch.device,
             batch_size: int, max_tokens: int, bucket: int) -> Dict[str, float]:
    model.eval()
    total_loss, n, probs, trues, prim = 0.0, 0, [], [], []
    for b in data.batches(batch_size, max_tokens, bucket, shuffle=False, rng=random.Random(0)):
        logits = model.head_forward(b["hidden"].to(device), b["attention_mask"].to(device))
        labels, valid = b["labels"].to(device), b["valid"].to(device)
        total_loss += masked_loss(loss_fn, logits, labels, valid).item() * float(valid.sum().item())
        k = int(valid.sum().item())
        n += k
        probs.append(torch.sigmoid(logits[:k]).cpu().numpy())
        trues.append(b["labels"][:k].numpy().astype(int))
        prim.append(b["primary"][:k].numpy())
    m = multilabel_metrics(np.concatenate(trues), np.concatenate(probs), np.concatenate(prim))
    m["loss"] = total_loss / n
    return m


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
    ap.add_argument("--results", default=os.path.join(REPO, "class_results"))
    ap.add_argument("--model-out", default=os.path.join(REPO, "models", "roberta-call-classifier"))
    ap.add_argument("--epochs", type=int, default=20, help="each epoch is ~10x the tokens of the BERT run; 20 is a sane default")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--max-tokens-per-batch", type=int, default=8192,
                    help="cap on rows x padded length per batch (bounds activation memory; 8192 = 32 x 256)")
    ap.add_argument("--length-bucket", type=int, default=64,
                    help="pad sequence lengths up to a multiple of this so an epoch uses few distinct tensor shapes "
                         "(MPS caches one compiled graph per shape); 64 -> at most 8 shapes")
    ap.add_argument("--lr", type=float, default=3e-5, help="peak learning rate for Adam")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--restart-cycles", type=int, default=4, help="number of cosine hard restarts over training")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--pos-weight", action="store_true",
                    help="weight positives per label by (neg/pos) in BCEWithLogitsLoss to counter label sparsity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N cached examples per split")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    os.makedirs(args.results, exist_ok=True)
    log_f = open(os.path.join(args.results, "train.log"), "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_f.write(line + "\n")
        log_f.flush()

    with open(os.path.join(args.cache, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    if meta["labels"] != LABELS:
        raise SystemExit(f"label order mismatch between cache {meta['labels']} and class_data/labels.json {LABELS}")

    train_data = CachedHiddenSplit(args.cache, "train", args.limit)
    val_data = CachedHiddenSplit(args.cache, "validation", args.limit)
    log(f"device={device} train={len(train_data)} validation={len(val_data)} cache_model={meta['model_dir']} "
        f"frozen_layers={meta['n_frozen_layers']} train_tokens={int(train_data.lengths.sum())}")

    model = CallRobertaClassifier(meta["model_dir"], trainable_layers=meta["trainable_layers"]).to(device)
    tokenizer = RobertaTokenizerFast.from_pretrained(meta["model_dir"], local_files_only=True)
    counts = model.count_parameters()
    log(f"parameters total={counts['total']:,} trainable={counts['trainable']:,} frozen={counts['frozen']:,}")

    pos_weight = None
    if args.pos_weight:
        pos = train_data.labels.sum(0)
        pos_weight = torch.tensor((len(train_data) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
        log(f"pos_weight={[round(float(x), 2) for x in pos_weight]}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")   # reduced by masked_loss()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    steps_per_epoch = train_data.num_batches(args.batch_size, args.max_tokens_per_batch, args.length_bucket)
    shapes = train_data.shapes(args.batch_size, args.max_tokens_per_batch, args.length_bucket)
    log(f"fixed batch shapes (rows x padded_len): {shapes}")
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, num_cycles=args.restart_cycles)
    log(f"loss=BCEWithLogitsLoss optimizer=Adam lr={args.lr} schedule=cosine_hard_restarts cycles={args.restart_cycles} "
        f"warmup_steps={warmup_steps} total_steps={total_steps} max_grad_norm={args.max_grad_norm} "
        f"batch_size={args.batch_size} max_tokens_per_batch={args.max_tokens_per_batch} length_bucket={args.length_bucket} "
        f"steps_per_epoch={steps_per_epoch}")
    with open(os.path.join(args.results, "training_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args) | {"steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "warmup_steps": warmup_steps,
                                "device": str(device), "optimizer": "Adam", "loss": "BCEWithLogitsLoss",
                                "scheduler": "cosine_with_hard_restarts_with_warmup", "labels": LABELS}, f, indent=2)

    history: List[Dict] = []
    best_val, best_epoch = float("inf"), -1
    rng = random.Random(args.seed)
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running, n_seen, grad_norms = 0.0, 0, []
        for b in train_data.batches(args.batch_size, args.max_tokens_per_batch, args.length_bucket, shuffle=True, rng=rng):
            logits = model.head_forward(b["hidden"].to(device), b["attention_mask"].to(device))
            labels, valid = b["labels"].to(device), b["valid"].to(device)
            loss = masked_loss(loss_fn, logits, labels, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)))
            optimizer.step()
            scheduler.step()
            k = float(valid.sum().item())
            running += loss.item() * k
            n_seen += k
            del logits, loss, labels, valid, b
        train_loss = running / n_seen
        free_device_cache(device)
        val = evaluate(model, val_data, loss_fn, device, args.eval_batch_size, 2 * args.max_tokens_per_batch, args.length_bucket)
        free_device_cache(device)
        rec = {"epoch": epoch, "train_loss": round(train_loss, 6), "val_loss": round(val["loss"], 6),
               "val_micro_f1": round(val["micro_f1"], 6), "val_macro_f1": round(val["macro_f1"], 6),
               "val_subset_accuracy": round(val["subset_accuracy"], 6), "val_hamming_loss": round(val["hamming_loss"], 6),
               "val_primary_accuracy": round(val["primary_accuracy"], 6), "lr_end_of_epoch": scheduler.get_last_lr()[0],
               "mean_grad_norm": round(float(np.mean(grad_norms)), 4),
               "clipped_frac": round(float(np.mean([g > args.max_grad_norm for g in grad_norms])), 4),
               "epoch_seconds": round(time.time() - t0, 1), "device_memory_gb": device_memory_gb(device)}
        history.append(rec)
        write_metrics(args.results, history)
        improved = val["loss"] < best_val
        if improved:
            best_val, best_epoch = val["loss"], epoch
            model.save(os.path.join(args.model_out, "best"), tokenizer, {"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}})
        log(f"epoch {epoch:3d}/{args.epochs} train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"micro_f1={val['micro_f1']:.4f} macro_f1={val['macro_f1']:.4f} subset_acc={val['subset_accuracy']:.4f} "
            f"primary_acc={val['primary_accuracy']:.4f} lr={rec['lr_end_of_epoch']:.2e} clipped={rec['clipped_frac']:.0%} "
            f"{rec['epoch_seconds']}s gpu_mem={rec['device_memory_gb']}GB{' *best*' if improved else ''}")

    model.save(os.path.join(args.model_out, "final"), tokenizer, {"epoch": args.epochs, "best_epoch": best_epoch})
    summary = {"epochs": args.epochs, "best_epoch": best_epoch, "best_val_loss": best_val, **{f"final_{k}": v for k, v in history[-1].items()},
               "total_seconds": round(time.time() - t_start, 1), "model_best": os.path.join(args.model_out, "best"),
               "model_final": os.path.join(args.model_out, "final")}
    with open(os.path.join(args.results, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"done best_epoch={best_epoch} best_val_loss={best_val:.4f} total={summary['total_seconds']}s")
    log_f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
