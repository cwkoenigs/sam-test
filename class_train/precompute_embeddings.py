#!/usr/bin/env python3
"""
Step 1: precompute the frozen part of RoBERTa once, before training.

Runs the embedding layer and the frozen encoder layers (all but the last)
over every call in class_data/{train,validation,test}.jsonl and stores the
per-token hidden states (float16) in a packed, memory-mapped array under
class_train/cache/. The full 100k-call dataset is ~20.7M tokens, so the cache
is ~32 GB on disk; use --limit to work with a subset.

Outputs per split:
    cache/<split>.hidden.npy   float16 (total_tokens, hidden)  packed, memmap
    cache/<split>.index.npz    offsets, lengths, label_vectors, primary_ids, ids
    cache/meta.json

Usage:
    uv run python class_train/precompute_embeddings.py                 # all splits, full data
    uv run python class_train/precompute_embeddings.py --limit 20000   # first 20k rows per split
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from transformers import RobertaTokenizerFast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LABEL2ID, LABELS, CallRobertaClassifier, pick_device  # noqa: E402

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
    ap.add_argument("--data-dir", default=os.path.join(REPO, "class_data"))
    ap.add_argument("--model-dir", default="/Users/sam/models/roberta-base", help="local roberta-base directory")
    ap.add_argument("--out", default=os.path.join(HERE, "cache"))
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--trainable-layers", type=int, default=1, help="layers left OUT of the cache (trained later)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows per split (0 = all)")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(0)
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir, local_files_only=True)
    model = CallRobertaClassifier(args.model_dir, trainable_layers=args.trainable_layers).to(device).eval()
    print(f"device={device} frozen_layers={model.n_frozen_layers} trainable_layers={model.trainable_layers} "
          f"params={model.count_parameters()}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    meta = {"model_dir": os.path.abspath(args.model_dir), "hidden_size": model.hidden_size,
            "n_frozen_layers": model.n_frozen_layers, "trainable_layers": model.trainable_layers,
            "max_length": args.max_length, "dtype": "float16", "labels": LABELS, "label2id": LABEL2ID,
            "limit": args.limit, "splits": {}}

    for split in args.splits.split(","):
        t0 = time.time()
        rows = read_jsonl(os.path.join(args.data_dir, f"{split}.jsonl"), args.limit)
        enc = tokenizer([r["transcript"] for r in rows], truncation=True, max_length=args.max_length,
                        padding=False, return_attention_mask=False)
        ids_list = enc["input_ids"]
        lengths = np.array([len(x) for x in ids_list], dtype=np.int32)
        offsets = np.zeros(len(rows), dtype=np.int64)
        offsets[1:] = np.cumsum(lengths[:-1])
        total = int(lengths.sum())
        label_vectors = np.array([r["label_vector"] for r in rows], dtype=np.float32)
        primary_ids = np.array([LABEL2ID[r["primary_label"]] for r in rows], dtype=np.int64)
        call_ids = np.array([r["id"] for r in rows])

        store = np.lib.format.open_memmap(os.path.join(args.out, f"{split}.hidden.npy"), mode="w+",
                                          dtype=np.float16, shape=(total, model.hidden_size))
        order = np.argsort(lengths, kind="stable")
        pad_id = tokenizer.pad_token_id
        done = 0
        for b in range(0, len(order), args.batch_size):
            idx = order[b:b + args.batch_size]
            # pad to a multiple of 64 tokens and keep the row count fixed so MPS compiles few graphs
            max_len = int(math.ceil(int(lengths[idx].max()) / 64) * 64)
            if len(idx) < args.batch_size:
                idx = np.concatenate([idx, idx[: args.batch_size - len(idx)]]) if len(idx) * 2 >= args.batch_size else \
                    np.resize(idx, args.batch_size)
            input_ids = torch.full((len(idx), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(idx), max_len), dtype=torch.long)
            for j, i in enumerate(idx):
                n = lengths[i]
                input_ids[j, :n] = torch.tensor(ids_list[i], dtype=torch.long)
                attn[j, :n] = 1
            hidden = model.frozen_forward(input_ids.to(device), attn.to(device)).to(torch.float16).cpu().numpy()
            for j, i in enumerate(idx):
                n = lengths[i]
                store[offsets[i]:offsets[i] + n] = hidden[j, :n]
            done += len(idx)
            if (b // args.batch_size) % 100 == 0:
                print(f"  [{split}] {done}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        store.flush()
        del store

        np.savez(os.path.join(args.out, f"{split}.index.npz"), offsets=offsets, lengths=lengths,
                 label_vectors=label_vectors, primary_ids=primary_ids, ids=call_ids)
        meta["splits"][split] = {"n_examples": len(rows), "total_tokens": total, "max_length_seen": int(lengths.max()),
                                 "truncated": int((lengths >= args.max_length).sum()),
                                 "bytes": total * model.hidden_size * 2,
                                 "label_counts": {l: int(label_vectors[:, k].sum()) for k, l in enumerate(LABELS)}}
        print(f"[{split}] {len(rows)} calls, {total} tokens, {meta['splits'][split]['bytes'] / 1e9:.2f} GB, "
              f"{time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote", os.path.join(args.out, "meta.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
