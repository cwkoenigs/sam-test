"""
PyTorch model definition for the authentication-chunk classifier.

`AuthBertClassifier` wraps a Hugging Face `BertForSequenceClassification`
(bert-base-uncased + pooler + linear head) and splits its forward pass into:

  * `frozen_forward`  embeddings + the first N-1 encoder layers (frozen, run
                      once in `precompute_embeddings.py` and cached to disk)
  * `head_forward`    the last encoder layer + pooler + dropout + classifier
                      (trainable, run every step in `train.py` on the cache)
  * `forward`         the full model, used for inference / evaluation

Only the last encoder layer(s), the pooler and the classifier receive
gradients; everything below is frozen (`requires_grad=False`). The model is
saved with `save_pretrained`, so the checkpoint in `models/` loads with the
standard `BertForSequenceClassification.from_pretrained` as well.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizerFast

LABELS: List[str] = ["ID_AUTH", "NO_AUTH", "OTHER_AUTH"]  # sorted, stable ids
LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: l for l, i in LABEL2ID.items()}

TRAIN_META_FILE = "auth_classifier_meta.json"


class AuthBertClassifier(nn.Module):
    def __init__(self, model_dir: str, trainable_layers: int = 1, dropout: Optional[float] = None,
                 num_labels: int = len(LABELS)) -> None:
        super().__init__()
        kwargs = {}
        if dropout is not None:
            kwargs["classifier_dropout"] = dropout
        self.hf = BertForSequenceClassification.from_pretrained(
            model_dir,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            attn_implementation="eager",   # BertLayer is called directly with an additive mask
            **kwargs,
        )
        n_layers = len(self.hf.bert.encoder.layer)
        if not (1 <= trainable_layers < n_layers):
            raise ValueError(f"trainable_layers must be in [1, {n_layers - 1}]")
        self.trainable_layers = trainable_layers
        self.n_frozen_layers = n_layers - trainable_layers
        self._apply_freezing()

    # ------------------------------------------------------------------ #
    # freezing
    # ------------------------------------------------------------------ #
    def _apply_freezing(self) -> None:
        for p in self.hf.parameters():
            p.requires_grad = False
        for layer in self.hf.bert.encoder.layer[self.n_frozen_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.hf.bert.pooler.parameters():
            p.requires_grad = True
        for p in self.hf.classifier.parameters():
            p.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.hf.parameters() if p.requires_grad)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.hf.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ------------------------------------------------------------------ #
    # forward pieces
    # ------------------------------------------------------------------ #
    @property
    def hidden_size(self) -> int:
        return self.hf.config.hidden_size

    def extended_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, T) 0/1 mask -> (B, 1, 1, T) additive mask in the model dtype."""
        return self.hf.bert.get_extended_attention_mask(attention_mask, attention_mask.shape)

    @torch.no_grad()
    def frozen_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Embeddings + frozen encoder layers. Returns (B, T, H) hidden states."""
        bert = self.hf.bert
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        hidden = bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
        ext = self.extended_mask(attention_mask)
        for layer in bert.encoder.layer[: self.n_frozen_layers]:
            hidden = layer(hidden, attention_mask=ext)[0]
        return hidden

    def head_forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Trainable encoder layer(s) + pooler + classifier. Returns (B, num_labels) logits."""
        bert = self.hf.bert
        ext = self.extended_mask(attention_mask)
        for layer in bert.encoder.layer[self.n_frozen_layers:]:
            hidden = layer(hidden, attention_mask=ext)[0]
        pooled = bert.pooler(hidden)
        pooled = self.hf.dropout(pooled)
        return self.hf.classifier(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.frozen_forward(input_ids, attention_mask, token_type_ids)
        return self.head_forward(hidden, attention_mask)

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, out_dir: str, tokenizer: Optional[BertTokenizerFast] = None,
             extra_meta: Optional[Dict] = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.hf.save_pretrained(out_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)
        meta = {
            "labels": LABELS,
            "trainable_layers": self.trainable_layers,
            "n_frozen_layers": self.n_frozen_layers,
            "parameters": self.count_parameters(),
        }
        if extra_meta:
            meta.update(extra_meta)
        with open(os.path.join(out_dir, TRAIN_META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, model_dir: str) -> "AuthBertClassifier":
        meta_path = os.path.join(model_dir, TRAIN_META_FILE)
        trainable_layers = 1
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                trainable_layers = int(json.load(f).get("trainable_layers", 1))
        return cls(model_dir, trainable_layers=trainable_layers)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
