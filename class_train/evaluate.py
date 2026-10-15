#!/usr/bin/env python3
"""
Step 3: evaluate a saved checkpoint end-to-end on raw transcripts.

Runs the FULL model (tokenizer -> all 12 layers -> head) on class_data
validation and test splits. Per-label decision thresholds are tuned on the
validation split (maximising per-label F1), then test metrics are reported at
both the default 0.5 threshold and the tuned thresholds.

Outputs under class_results/:
    test_metrics.json   micro/macro F1, subset accuracy, Hamming loss, per-label
                        precision/recall/F1 (0.5 and tuned), primary-label accuracy
                        and 7x7 confusion matrix from the argmax
    thresholds.json     tuned per-label thresholds
    test_errors.jsonl   calls whose predicted label set differs from the truth

Usage:
    uv run python class_train/evaluate.py                                   # best checkpoint
    uv run python class_train/evaluate.py --model models/roberta-call-classifier/final
    uv run python class_train/evaluate.py --limit 2000                      # quick check
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from transformers import RobertaTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LABEL2ID, LABELS, CallRobertaClassifier, pick_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def read_jsonl(path: str, limit: int) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def predict(model: CallRobertaClassifier, tok: RobertaTokenizerFast, rows: List[Dict], device: torch.device,
            batch_size: int, max_length: int, bucket: int = 64) -> Tuple[np.ndarray, float]:
    """
    Full-model inference with FIXED tensor shapes: lengths are padded to a
    multiple of `bucket` and every batch in a bucket has `batch_size` rows
    (partial batches are padded with repeated rows that are discarded), so
    the MPS graph cache holds at most 512 / bucket shapes.
    """
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    out = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    total_loss = 0.0
    by_bucket: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        n = min(r["roberta_tokens"], max_length)
        by_bucket.setdefault(int(math.ceil(n / bucket) * bucket), []).append(i)
    for pad_len, idxs in sorted(by_bucket.items()):
        for i in range(0, len(idxs), batch_size):
            real = idxs[i:i + batch_size]
            filled = real + [real[j % len(real)] for j in range(batch_size - len(real))]
            enc = tok([rows[k]["transcript"] for k in filled], truncation=True, max_length=pad_len,
                      padding="max_length", return_tensors="pt")
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))[: len(real)]
            labels = torch.tensor([rows[k]["label_vector"] for k in real], dtype=torch.float32, device=device)
            total_loss += loss_fn(logits, labels).item()
            out[real] = torch.sigmoid(logits).cpu().numpy()
    return out, total_loss / (len(rows) * len(LABELS))


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> List[float]:
    grid = np.linspace(0.05, 0.95, 19)
    ths = []
    for k in range(y_true.shape[1]):
        best_t, best_f = 0.5, -1.0
        for t in grid:
            p = (y_prob[:, k] >= t).astype(int)
            tp = int((p & (y_true[:, k] == 1)).sum())
            fp = int((p & (y_true[:, k] == 0)).sum())
            fn = int(((1 - p) & (y_true[:, k] == 1)).sum())
            f = 2 * tp / max(2 * tp + fp + fn, 1)
            if f > best_f:
                best_f, best_t = f, float(t)
        ths.append(round(best_t, 2))
    return ths


def report(y_true: np.ndarray, y_prob: np.ndarray, ths: List[float]) -> Dict:
    y_pred = (y_prob >= np.array(ths)).astype(int)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    return {"micro_f1": float(mf), "micro_precision": float(mp), "micro_recall": float(mr), "macro_f1": float(f.mean()),
            "subset_accuracy": float((y_pred == y_true).all(1).mean()), "hamming_loss": float((y_pred != y_true).mean()),
            "per_label": {l: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k]),
                              "threshold": ths[k]} for k, l in enumerate(LABELS)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(REPO, "models", "roberta-call-classifier", "best"))
    ap.add_argument("--data-dir", default=os.path.join(REPO, "class_data"))
    ap.add_argument("--results", default=os.path.join(REPO, "class_results"))
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="rows per split (0 = all)")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = RobertaTokenizerFast.from_pretrained(args.model, local_files_only=True)
    model = CallRobertaClassifier.load(args.model).to(device).eval()

    t0 = time.time()
    val_rows = read_jsonl(os.path.join(args.data_dir, "validation.jsonl"), args.limit)
    val_prob, _ = predict(model, tok, val_rows, device, args.batch_size, args.max_length)
    val_true = np.array([r["label_vector"] for r in val_rows])
    ths = tune_thresholds(val_true, val_prob)

    test_rows = read_jsonl(os.path.join(args.data_dir, "test.jsonl"), args.limit)
    test_prob, test_loss = predict(model, tok, test_rows, device, args.batch_size, args.max_length)
    test_true = np.array([r["label_vector"] for r in test_rows])
    primary_true = np.array([LABEL2ID[r["primary_label"]] for r in test_rows])
    primary_pred = test_prob.argmax(1)

    metrics = {
        "model": os.path.abspath(args.model), "n_validation": len(val_rows), "n_test": len(test_rows),
        "test_bce_loss": test_loss,
        "at_threshold_0.5": report(test_true, test_prob, [0.5] * len(LABELS)),
        "at_tuned_thresholds": report(test_true, test_prob, ths),
        "primary_label": {"accuracy": float((primary_pred == primary_true).mean()), "labels": LABELS,
                          "confusion_rows_true_cols_pred": confusion_matrix(primary_true, primary_pred, labels=list(range(len(LABELS)))).tolist()},
        "by_noise_level": {},
        "seconds": round(time.time() - t0, 1),
    }
    tuned_pred = (test_prob >= np.array(ths)).astype(int)
    for lvl in ("low", "medium", "high"):
        m = np.array([r["noise_level"] == lvl for r in test_rows])
        if m.any():
            metrics["by_noise_level"][lvl] = {"n": int(m.sum()), "subset_accuracy": float((tuned_pred[m] == test_true[m]).all(1).mean()),
                                              "primary_accuracy": float((primary_pred[m] == primary_true[m]).mean())}

    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.results, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({l: t for l, t in zip(LABELS, ths)}, f, indent=2)
    with open(os.path.join(args.results, "test_errors.jsonl"), "w", encoding="utf-8") as f:
        for r, t, p, pr in zip(test_rows, test_true, tuned_pred, test_prob):
            if (t != p).any():
                f.write(json.dumps({"id": r["id"], "noise_level": r["noise_level"], "true": [l for l, x in zip(LABELS, t) if x],
                                    "pred": [l for l, x in zip(LABELS, p) if x],
                                    "prob": {l: round(float(x), 3) for l, x in zip(LABELS, pr)}, "transcript": r["transcript"]}) + "\n")

    a, b = metrics["at_threshold_0.5"], metrics["at_tuned_thresholds"]
    print(json.dumps({"n_test": len(test_rows), "bce_loss": round(test_loss, 5),
                      "micro_f1@0.5": round(a["micro_f1"], 4), "macro_f1@0.5": round(a["macro_f1"], 4), "subset_acc@0.5": round(a["subset_accuracy"], 4),
                      "micro_f1@tuned": round(b["micro_f1"], 4), "macro_f1@tuned": round(b["macro_f1"], 4), "subset_acc@tuned": round(b["subset_accuracy"], 4),
                      "primary_accuracy": round(metrics["primary_label"]["accuracy"], 4)}, indent=2))
    print(f"{'label':16s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'thr':>5s} {'n':>6s}")
    for l, m in b["per_label"].items():
        print(f"{l:16s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['threshold']:5.2f} {m['support']:6d}")
    print("by noise level:", metrics["by_noise_level"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
