#!/usr/bin/env python3
"""
Zero-shot entity extraction over NER_data with the local GLiNER model, scored
against the gold spans, plus PCI / PII call-level detection.

The model is never fine-tuned. Four accuracy fixes are available as switches
and can be compared on the same calls with --compare:

  --strip-speaker-tags   blank out "agent:" / "customer:" in the model input
                         (same-length replacement, offsets unchanged) so they
                         stop being predicted as person names
  --normalize-digits     rewrite spoken digit runs ("four four six five",
                         "double one", "oh") to digits before the model sees
                         the text; predictions are mapped back to the original
                         character offsets
  --rules                deterministic PCI/identifier layer on the normalised
                         text: Luhn-valid 15-16 digit runs -> CARD_NUMBER;
                         context + pattern -> CARD_CVV, CARD_EXPIRATION, SSN;
                         "npi" context + 80840-Luhn -> PROVIDER_NPI. Rule spans
                         override overlapping model spans.
  --label-set tuned      alternative natural-language prompt labels
  --rules-exclusive      card / cvv / expiration / ssn / npi come only from the
                         rules; the model's (noisy) predictions for those labels
                         are dropped

Transcripts are longer than GLiNER's window, so each call is split into
overlapping word windows and predictions are mapped back to absolute offsets.

Usage:
    uv run python NER_data/run_gliner.py --limit 500                      # baseline
    uv run python NER_data/run_gliner.py --limit 500 --strip-speaker-tags --normalize-digits --rules
    uv run python NER_data/run_gliner.py --limit 500 --compare            # full matrix -> results/comparison.{json,md}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))

LABEL_SETS = {
    "default": None,  # taken from labels.json
    "tuned": {
        "PERSON_NAME": "person's full name", "PROVIDER_NAME": "doctor or clinic name", "PROVIDER_NPI": "npi number",
        "DATE_OF_BIRTH": "date of birth", "ADDRESS": "street address with city", "SSN": "social security number",
        "CARD_NUMBER": "payment card number", "CARD_CVV": "cvv security code", "CARD_EXPIRATION": "card expiration date",
        "MEMBER_ID": "member id number", "PHONE_NUMBER": "telephone number",
    },
}

DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9"}
MULT = {"double": 2, "triple": 3}
FILLER_SEQS = [("um",), ("uh",), ("hmm",), ("hold", "on"), ("let", "me", "see")]
RULE_LABELS = {"CARD_NUMBER", "CARD_CVV", "CARD_EXPIRATION", "SSN", "PROVIDER_NPI"}
FILLER_RE = r"(?:um|uh|hmm|hold on|let me see)"
MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"


# --------------------------------------------------------------------------- #
# text preparation with offset maps
# --------------------------------------------------------------------------- #


def strip_speaker_tags(text: str) -> str:
    """Replace 'agent:' / 'customer:' with spaces of the same length (offsets unchanged)."""
    return re.sub(r"\b(agent|customer):", lambda m: " " * len(m.group(0)), text)


def normalize_digits(text: str) -> Tuple[str, List[int], List[int]]:
    """
    Convert runs of >= 3 spoken digits into digit strings. A filler ("um",
    "hold on", ...) inside a run is skipped when digits continue after it.
    Returns (normalised text, start_map, end_map) where start_map[i] / end_map[i]
    give the original start / end offset for normalised character i.
    """
    toks = []
    pos = 0
    for w in text.split(" "):
        toks.append((w, pos, pos + len(w)))
        pos += len(w) + 1
    out_parts: List[Tuple[str, int, int]] = []     # (text, orig_start, orig_end)
    i = 0
    while i < len(toks):
        w = toks[i][0]
        if w in DIGIT_WORDS or (w in MULT and i + 1 < len(toks) and toks[i + 1][0] in DIGIT_WORDS):
            j, digits = i, ""
            while j < len(toks):
                wj = toks[j][0]
                if wj in DIGIT_WORDS:
                    digits += DIGIT_WORDS[wj]
                    j += 1
                elif wj in MULT and j + 1 < len(toks) and toks[j + 1][0] in DIGIT_WORDS:
                    digits += DIGIT_WORDS[toks[j + 1][0]] * MULT[wj]
                    j += 2
                else:
                    skipped = False
                    for seq in FILLER_SEQS:
                        k = j + len(seq)
                        if tuple(t[0] for t in toks[j:k]) == seq and k < len(toks) and (toks[k][0] in DIGIT_WORDS or toks[k][0] in MULT):
                            j, skipped = k, True
                            break
                    if not skipped:
                        break
            if len(digits) >= 3:
                out_parts.append((digits, toks[i][1], toks[j - 1][2]))
                i = j
                continue
        out_parts.append((w, toks[i][1], toks[i][2]))
        i += 1
    norm, smap, emap = [], [], []
    for k, (t, s, e) in enumerate(out_parts):
        if k:
            norm.append(" ")
            smap.append(e_prev)  # type: ignore[name-defined]  # a space maps to the gap after the previous token
            emap.append(e_prev)  # type: ignore[name-defined]
        norm.append(t)
        for c in range(len(t)):
            # spread original span across normalised chars; start of first char = orig start, end of last = orig end
            smap.append(s if c == 0 else min(e, s + c))
            emap.append(e if c == len(t) - 1 else min(e, s + c + 1))
        e_prev = e
    return "".join(norm), smap, emap


def identity_maps(text: str) -> Tuple[List[int], List[int]]:
    return list(range(len(text))), list(range(1, len(text) + 1))


# --------------------------------------------------------------------------- #
# rule layer (runs on the normalised text)
# --------------------------------------------------------------------------- #


def luhn_valid(num: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(num)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


DIGIT_RUN = re.compile(r"\d+(?:(?:[ -]|\s" + FILLER_RE + r"\s)\d+)*")
CVV_CTX = re.compile(r"(?:cvv|cvc|security code|code on the back|code on the front|digit code)(?:\W+\w+){0,10}?\W+(\d{3,4})\b")
EXP_CTX = re.compile(r"(?:expiration|expires|expiry|exp date)(?:\W+\w+){0,8}?\W+("
                     r"\d{2}/\d{2}|\d{2} \d{2}\b|\d{4}\b|(?:" + MONTHS + r") (?:\d{4}|twenty \w+(?: \w+)?|two thousand(?: \w+){0,2}))")
SSN_CTX = re.compile(r"(?:social|ssn)(?:\W+\w+){0,10}?\W+(\d{3}[- ]?\d{2}[- ]?\d{4}|\d{9})\b")
NPI_CTX = re.compile(r"\bnpi\b(?:\W+\w+){0,8}?\W+(\d{3} \d{3} \d{4}|\d{10})\b")


def rule_spans(norm: str) -> List[Dict]:
    out = []
    for m in DIGIT_RUN.finditer(norm):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) in (15, 16) and luhn_valid(digits):
            out.append({"start": m.start(), "end": m.end(), "label": "CARD_NUMBER", "score": 0.99, "text": m.group(0), "rule": True})
    for rx, label in ((CVV_CTX, "CARD_CVV"), (EXP_CTX, "CARD_EXPIRATION"), (SSN_CTX, "SSN")):
        for m in rx.finditer(norm):
            out.append({"start": m.start(1), "end": m.end(1), "label": label, "score": 0.98, "text": m.group(1), "rule": True})
    for m in NPI_CTX.finditer(norm):
        digits = re.sub(r"\D", "", m.group(1))
        if luhn_valid("80840" + digits):
            out.append({"start": m.start(1), "end": m.end(1), "label": "PROVIDER_NPI", "score": 0.98, "text": m.group(1), "rule": True})
    return out


# --------------------------------------------------------------------------- #
# windows, matching
# --------------------------------------------------------------------------- #


def windows(text: str, max_words: int, stride: int) -> List[Tuple[int, str]]:
    spans, pos = [], 0
    for w in text.split(" "):
        spans.append((pos, pos + len(w)))
        pos += len(w) + 1
    out, i = [], 0
    while i < len(spans):
        j = min(i + max_words, len(spans))
        out.append((spans[i][0], text[spans[i][0]:spans[j - 1][1]]))
        if j == len(spans):
            break
        i += stride
    return out


def overlap(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1, max(a[1], b[1]) - min(a[0], b[0]))


def prf(tp: int, fp_: int, fn_: int) -> Dict[str, float]:
    p = tp / (tp + fp_) if tp + fp_ else 0.0
    r = tp / (tp + fn_) if tp + fn_ else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


# --------------------------------------------------------------------------- #
# one configuration
# --------------------------------------------------------------------------- #


def predict_call(model, transcript: str, gliner_labels: List[str], g2l: Dict[str, str], cfg: Dict) -> List[Dict]:
    text = strip_speaker_tags(transcript) if cfg["strip"] else transcript
    if cfg["normalize"]:
        norm, smap, emap = normalize_digits(text)
    else:
        norm, (smap, emap) = text, identity_maps(text)

    preds: List[Dict] = []
    wins = windows(norm, cfg["window_words"], cfg["stride_words"])
    texts = [w for _, w in wins]
    for b in range(0, len(texts), cfg["batch_size"]):
        outs = model.batch_predict_entities(texts[b:b + cfg["batch_size"]], gliner_labels, threshold=cfg["threshold"])
        for (off, _), ents in zip(wins[b:b + cfg["batch_size"]], outs):
            for e in ents:
                preds.append({"start": off + e["start"], "end": off + e["end"], "label": g2l.get(e["label"], e["label"]),
                              "score": round(float(e["score"]), 3), "rule": False})
    if cfg["rules"]:
        if cfg.get("rules_exclusive"):
            preds = [p for p in preds if p["label"] not in RULE_LABELS]
        preds.extend(rule_spans(norm))
    # map normalised offsets back to original offsets
    for p in preds:
        p["start"], p["end"] = smap[p["start"]], emap[p["end"] - 1]
        p["text"] = transcript[p["start"]:p["end"]]
    # rules first, then higher score; drop anything overlapping an already-kept span (any label)
    preds.sort(key=lambda p: (not p["rule"], -p["score"]))
    kept: List[Dict] = []
    for p in preds:
        if not any(overlap((p["start"], p["end"]), (q["start"], q["end"])) > 0.5 for q in kept):
            kept.append(p)
    return kept


def evaluate_config(model, rows: List[Dict], types: Dict, keep: set, cfg: Dict) -> Tuple[Dict, List[Dict]]:
    label_map = LABEL_SETS[cfg["label_set"]] or {k: v["gliner"] for k, v in types.items()}
    gliner_labels = [label_map[k] for k in keep]
    g2l = {label_map[k]: k for k in keep}
    pci_types = {k for k, v in types.items() if v["pci"]}
    tp_part, tp_exact, fp, fn = Counter(), Counter(), Counter(), Counter()
    fmt_hit, fmt_tot = Counter(), Counter()
    cc = Counter()
    errors = []
    t0 = time.time()
    for r in rows:
        gold = [e for e in r["entities"] if e["label"] in keep]
        preds = [p for p in predict_call(model, r["transcript"], gliner_labels, g2l, cfg) if p["label"] in keep]
        matched = set()
        for g in gold:
            fmt_tot[f"{g['label']}:{g['format']}"] += 1
            best, best_ov = None, 0.0
            for i, p in enumerate(preds):
                if i in matched or p["label"] != g["label"]:
                    continue
                ov = overlap((g["start"], g["end"]), (p["start"], p["end"]))
                if ov > best_ov:
                    best, best_ov = i, ov
            if best is not None and best_ov >= 0.5:
                matched.add(best)
                tp_part[g["label"]] += 1
                fmt_hit[f"{g['label']}:{g['format']}"] += 1
                if preds[best]["start"] == g["start"] and preds[best]["end"] == g["end"]:
                    tp_exact[g["label"]] += 1
            else:
                fn[g["label"]] += 1
        for i, p in enumerate(preds):
            if i not in matched:
                fp[p["label"]] += 1
        gp, pp = r["contains_pci"], any(p["label"] in pci_types for p in preds)
        cc["pci_tp"] += gp and pp; cc["pci_fp"] += (not gp) and pp; cc["pci_fn"] += gp and not pp; cc["pci_tn"] += (not gp) and not pp
        gi, pi = r["contains_pii"], any(types[p["label"]]["pii"] for p in preds)
        cc["pii_tp"] += gi and pi; cc["pii_fp"] += (not gi) and pi; cc["pii_fn"] += gi and not pi; cc["pii_tn"] += (not gi) and not pi
        if len(errors) < 300:
            missed = [g for g in gold if not any(p["label"] == g["label"] and overlap((g["start"], g["end"]), (p["start"], p["end"])) >= 0.5 for p in preds)]
            spurious = [p for i, p in enumerate(preds) if i not in matched]
            if missed or spurious:
                errors.append({"id": r["id"], "noise_level": r["noise_level"], "missed": missed, "spurious": spurious})
    per_label = {}
    for l in sorted(keep):
        per_label[l] = {"support": tp_part[l] + fn[l], "partial": prf(tp_part[l], fp[l], fn[l]),
                        "exact": prf(tp_exact[l], fp[l] + tp_part[l] - tp_exact[l], fn[l] + tp_part[l] - tp_exact[l])}
    metrics = {
        "config": cfg, "n_calls": len(rows), "labels": gliner_labels,
        "micro_partial_match": prf(sum(tp_part.values()), sum(fp.values()), sum(fn.values())),
        "macro_f1_partial": round(sum(v["partial"]["f1"] for v in per_label.values()) / len(per_label), 4),
        "per_label": per_label,
        "pci_call_detection": {**prf(cc["pci_tp"], cc["pci_fp"], cc["pci_fn"]), **{k[4:]: cc[k] for k in ("pci_tp", "pci_fp", "pci_fn", "pci_tn")}},
        "pii_call_detection": {**prf(cc["pii_tp"], cc["pii_fp"], cc["pii_fn"]), **{k[4:]: cc[k] for k in ("pii_tp", "pii_fp", "pii_fn", "pii_tn")}},
        "recall_by_format": {k: round(fmt_hit[k] / v, 4) for k, v in sorted(fmt_tot.items())},
        "seconds": round(time.time() - t0, 1),
    }
    return metrics, errors


COMPARE_MATRIX = [
    ("baseline", {}),
    ("strip_tags", {"strip": True}),
    ("normalize_digits", {"normalize": True}),
    ("rules", {"rules": True}),
    ("normalize+rules", {"normalize": True, "rules": True}),
    ("all_fixes", {"strip": True, "normalize": True, "rules": True}),
    ("all_fixes+tuned_labels", {"strip": True, "normalize": True, "rules": True, "label_set": "tuned"}),
    ("all_fixes+rules_exclusive", {"strip": True, "normalize": True, "rules": True, "rules_exclusive": True}),
    ("all_fixes+tuned+rules_exclusive", {"strip": True, "normalize": True, "rules": True, "rules_exclusive": True, "label_set": "tuned"}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="/Users/sam/models/gliner_base")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--window-words", type=int, default=120)
    ap.add_argument("--stride-words", type=int, default=90)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--labels-subset", choices=["all", "pci", "pii"], default="all")
    ap.add_argument("--label-set", choices=list(LABEL_SETS), default="default")
    ap.add_argument("--strip-speaker-tags", action="store_true")
    ap.add_argument("--normalize-digits", action="store_true")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--rules-exclusive", action="store_true",
                    help="with --rules: card number / cvv / expiration / ssn / npi come ONLY from the rules (model predictions for those labels dropped)")
    ap.add_argument("--compare", action="store_true", help="run the fix matrix on the same calls and write comparison.{json,md}")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    from gliner import GLiNER
    model = GLiNER.from_pretrained(args.model, local_files_only=True)
    try:
        import torch
        if torch.backends.mps.is_available():
            model = model.to("mps")
    except Exception:
        pass
    with open(os.path.join(HERE, "labels.json"), encoding="utf-8") as f:
        lab = json.load(f)
    types = lab["entity_types"]
    keep = {k for k, v in types.items() if args.labels_subset == "all" or v["pci"] or (args.labels_subset == "pii" and v["pii"])}
    rows = []
    with open(os.path.join(HERE, f"{args.split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if args.limit and len(rows) >= args.limit:
                break
    base_cfg = {"threshold": args.threshold, "window_words": args.window_words, "stride_words": args.stride_words,
                "batch_size": args.batch_size, "strip": args.strip_speaker_tags, "normalize": args.normalize_digits,
                "rules": args.rules, "rules_exclusive": args.rules_exclusive, "label_set": args.label_set}
    os.makedirs(args.results, exist_ok=True)

    if not args.compare:
        metrics, errors = evaluate_config(model, rows, types, keep, base_cfg)
        metrics.update({"model": args.model, "split": args.split})
        suffix = args.tag or ("" if args.labels_subset == "all" else f"_{args.labels_subset}")
        with open(os.path.join(args.results, f"gliner_metrics{suffix}.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(args.results, f"gliner_errors{suffix}.jsonl"), "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(json.dumps({k: metrics[k] for k in ("n_calls", "micro_partial_match", "macro_f1_partial", "pci_call_detection", "pii_call_detection", "seconds")}, indent=2))
        print(f"{'label':16s} {'n':>6s} {'P':>6s} {'R':>6s} {'F1':>6s}   exact-F1")
        for l, m in metrics["per_label"].items():
            print(f"{l:16s} {m['support']:6d} {m['partial']['precision']:6.3f} {m['partial']['recall']:6.3f} {m['partial']['f1']:6.3f}   {m['exact']['f1']:.3f}")
        return 0

    results = {}
    for name, over in COMPARE_MATRIX:
        cfg = {**base_cfg, "strip": False, "normalize": False, "rules": False, "rules_exclusive": False, "label_set": "default", **over}
        metrics, errors = evaluate_config(model, rows, types, keep, cfg)
        results[name] = metrics
        with open(os.path.join(args.results, f"gliner_errors_{name}.jsonl"), "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"{name:24s} micro_f1={metrics['micro_partial_match']['f1']:.3f} macro_f1={metrics['macro_f1_partial']:.3f} "
              f"pci_f1={metrics['pci_call_detection']['f1']:.3f} card_f1={metrics['per_label'].get('CARD_NUMBER', {}).get('partial', {}).get('f1', 0):.3f} "
              f"{metrics['seconds']}s", flush=True)
    with open(os.path.join(args.results, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "n_calls": len(rows), "threshold": args.threshold, "model": args.model, "results": results}, f, indent=2)
    labels = sorted(keep)
    lines = [f"GLiNER zero-shot fix comparison: {len(rows)} {args.split} calls, threshold {args.threshold}, partial-match F1\n",
             "| config | micro F1 | macro F1 | PCI call P / R / F1 | " + " | ".join(labels) + " |",
             "|---|---:|---:|---|" + "---:|" * len(labels)]
    for name, m in results.items():
        pc = m["pci_call_detection"]
        lines.append(f"| {name} | {m['micro_partial_match']['f1']:.3f} | {m['macro_f1_partial']:.3f} | "
                     f"{pc['precision']:.2f} / {pc['recall']:.2f} / {pc['f1']:.2f} | " +
                     " | ".join(f"{m['per_label'][l]['partial']['f1']:.2f}" for l in labels) + " |")
    with open(os.path.join(args.results, "comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
