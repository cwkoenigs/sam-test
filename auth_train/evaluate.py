#!/usr/bin/env python3
"""
Step 3: evaluate a saved checkpoint end-to-end on the held-out test split.

Unlike training, this runs the FULL model (tokenizer -> embeddings -> all 12
layers -> classifier) on raw text from auth_data/test.jsonl, so it also checks
that the saved checkpoint in models/ works standalone without the cache.

Outputs auth_train/results/test_metrics.json and test_errors.jsonl.

Usage:
    uv run python auth_train/evaluate.py                      # models/bert-auth-classifier/best
    uv run python auth_train/evaluate.py --model models/bert-auth-classifier/final
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LABEL2ID, LABELS, AuthBertClassifier, pick_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(REPO, "models", "bert-auth-classifier", "best"))
    ap.add_argument("--data", default=os.path.join(REPO, "auth_data", "test.jsonl"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--max-length", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    tokenizer = BertTokenizerFast.from_pretrained(args.model)
    model = AuthBertClassifier.load(args.model).to(device).eval()

    rows = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    y_true = np.array([LABEL2ID[r["label"]] for r in rows])
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    t0 = time.time()
    preds, probs, total_loss = [], [], 0.0
    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch = rows[i:i + args.batch_size]
            enc = tokenizer([r["text"] for r in batch], truncation=True, max_length=args.max_length,
                            padding=True, return_tensors="pt")
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            labels = torch.tensor(y_true[i:i + len(batch)], device=device)
            total_loss += loss_fn(logits, labels).item()
            p = torch.softmax(logits, -1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(-1))
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)

    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    metrics = {
        "model": os.path.abspath(args.model),
        "data": os.path.abspath(args.data),
        "n_examples": len(rows),
        "loss": total_loss / len(rows),
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": report["macro avg"]["f1-score"],
        "per_class": {l: report[l] for l in LABELS},
        "confusion_matrix": {"labels": LABELS, "rows_true_cols_pred": cm.tolist()},
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(args.results, exist_ok=True)
    with open(os.path.join(args.results, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.results, "test_errors.jsonl"), "w", encoding="utf-8") as f:
        for r, t, p, pr in zip(rows, y_true, y_pred, y_prob):
            if t != p:
                f.write(json.dumps({"id": r["id"], "true": LABELS[t], "pred": LABELS[p],
                                    "prob": {l: round(float(x), 4) for l, x in zip(LABELS, pr)},
                                    "text": r["text"]}) + "\n")

    print(json.dumps({k: metrics[k] for k in ("n_examples", "loss", "accuracy", "macro_f1")}, indent=2))
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))
    print("confusion (rows=true, cols=pred):", LABELS)
    print(cm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
