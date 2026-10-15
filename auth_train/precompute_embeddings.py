#!/usr/bin/env python3
"""
Step 1: precompute the frozen part of BERT once, before training.

For every chunk in auth_data/{train,validation,test}.jsonl this runs the
embedding layer and the frozen encoder layers (all but the last) and stores
the resulting per-token hidden states (float16) in a packed, memory-mapped
array under auth_train/cache/. Training then only has to run the trainable
last layer + classifier over the cached states, which is what makes 50 epochs
cheap.

Outputs per split:
    cache/<split>.hidden.npy   float16 (total_tokens, hidden)  packed, memmap
    cache/<split>.index.npz    offsets, lengths, labels, ids
    cache/meta.json            model dir, frozen layer count, label map, ...

Usage:
    uv run python auth_train/precompute_embeddings.py \
        --model-dir /Users/sam/models/bert-base-uncased
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from transformers import BertTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LABEL2ID, LABELS, AuthBertClassifier, pick_device  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def read_jsonl(path: str, limit: int = 0) -> List[Dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(REPO, "auth_data"))
    ap.add_argument("--model-dir", default="/Users/sam/models/bert-base-uncased")
    ap.add_argument("--out", default=os.path.join(HERE, "cache"))
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--max-length", type=int, default=200, help="incl. [CLS]/[SEP]; data max is 196")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--trainable-layers", type=int, default=1, help="layers left OUT of the cache (trained later)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N rows per split")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(0)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = AuthBertClassifier(args.model_dir, trainable_layers=args.trainable_layers).to(device).eval()
    print(f"device={device} frozen_layers={model.n_frozen_layers} trainable_layers={model.trainable_layers} "
          f"params={model.count_parameters()}")

    os.makedirs(args.out, exist_ok=True)
    meta = {
        "model_dir": os.path.abspath(args.model_dir),
        "hidden_size": model.hidden_size,
        "n_frozen_layers": model.n_frozen_layers,
        "trainable_layers": model.trainable_layers,
        "max_length": args.max_length,
        "dtype": "float16",
        "labels": LABELS,
        "label2id": LABEL2ID,
        "splits": {},
    }

    for split in args.splits.split(","):
        t0 = time.time()
        rows = read_jsonl(os.path.join(args.data_dir, f"{split}.jsonl"), args.limit)
        enc = tokenizer([r["text"] for r in rows], truncation=True, max_length=args.max_length,
                        padding=False, return_attention_mask=False)
        ids_list = enc["input_ids"]
        lengths = np.array([len(x) for x in ids_list], dtype=np.int32)
        offsets = np.zeros(len(rows), dtype=np.int64)
        offsets[1:] = np.cumsum(lengths[:-1])
        total = int(lengths.sum())
        labels = np.array([LABEL2ID[r["label"]] for r in rows], dtype=np.int64)
        chunk_ids = np.array([r["id"] for r in rows])
        truncated = int((lengths >= args.max_length).sum())

        hidden_path = os.path.join(args.out, f"{split}.hidden.npy")
        store = np.lib.format.open_memmap(hidden_path, mode="w+", dtype=np.float16,
                                          shape=(total, model.hidden_size))

        # process in length-sorted batches to minimise padding
        order = np.argsort(lengths, kind="stable")
        done = 0
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            max_len = int(lengths[idx].max())
            input_ids = torch.zeros((len(idx), max_len), dtype=torch.long)
            attn = torch.zeros((len(idx), max_len), dtype=torch.long)
            for j, i in enumerate(idx):
                n = lengths[i]
                input_ids[j, :n] = torch.tensor(ids_list[i], dtype=torch.long)
                attn[j, :n] = 1
            hidden = model.frozen_forward(input_ids.to(device), attn.to(device))
            hidden = hidden.to(torch.float16).cpu().numpy()
            for j, i in enumerate(idx):
                n = lengths[i]
                store[offsets[i]:offsets[i] + n] = hidden[j, :n]
            done += len(idx)
            if (b // args.batch_size) % 50 == 0:
                print(f"  [{split}] {done}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        store.flush()
        del store

        np.savez(os.path.join(args.out, f"{split}.index.npz"),
                 offsets=offsets, lengths=lengths, labels=labels, ids=chunk_ids)
        meta["splits"][split] = {
            "n_examples": len(rows), "total_tokens": total, "max_length_seen": int(lengths.max()),
            "truncated": truncated, "bytes": total * model.hidden_size * 2,
            "label_counts": {LABELS[k]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
        }
        print(f"[{split}] {len(rows)} examples, {total} tokens, {meta['splits'][split]['bytes'] / 1e9:.2f} GB, "
              f"truncated={truncated}, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote", os.path.join(args.out, "meta.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
