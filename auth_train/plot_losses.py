#!/usr/bin/env python3
"""
Plot train / validation loss per epoch from auth_train/results/epoch_metrics.json.

Writes auth_train/results/loss_curve.png (referenced from the repo README).

Usage:
    uv run python auth_train/plot_losses.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Two-series categorical palette (validated: CVD dE 24.7, normal dE 33.6, contrast >= 3:1)
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
TEXT_PRIMARY = "#1f1f1d"
TEXT_SECONDARY = "#5c5b55"
GRID = "#e6e5e0"
SURFACE = "#fcfcfb"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=os.path.join(HERE, "results", "epoch_metrics.json"))
    ap.add_argument("--args", default=os.path.join(HERE, "results", "training_args.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "loss_curve.png"))
    ap.add_argument("--title", default="bert-base-uncased auth classifier: loss per epoch")
    ap.add_argument("--subtitle", default="last layer + classifier trained; Adam, cosine with {cycles} hard restarts (grey lines)")
    ap.add_argument("--loss-name", default="cross-entropy loss")
    ap.add_argument("--floor", type=float, default=1e-6,
                    help="losses are recorded to 6 decimals; values below this are drawn at the floor")
    a = ap.parse_args()

    with open(a.metrics, encoding="utf-8") as f:
        hist = json.load(f)
    restarts = []
    if os.path.exists(a.args):
        with open(a.args, encoding="utf-8") as f:
            targs = json.load(f)
        cycles, epochs = targs.get("restart_cycles", 0), targs.get("epochs", len(hist))
        if cycles > 1:
            restarts = [round(epochs * k / cycles) for k in range(1, cycles)]

    epochs = [h["epoch"] for h in hist]
    train = [max(h["train_loss"], a.floor) for h in hist]
    val = [max(h["val_loss"], a.floor) for h in hist]

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": TEXT_PRIMARY,
                         "axes.labelcolor": TEXT_SECONDARY, "xtick.color": TEXT_SECONDARY,
                         "ytick.color": TEXT_SECONDARY})
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for x in restarts:  # cosine hard-restart boundaries, as faint context
        ax.axvline(x + 0.5, color=GRID, linewidth=1, zorder=1)

    ax.plot(epochs, train, color=TRAIN_COLOR, linewidth=2, zorder=3, label="train loss")
    ax.plot(epochs, val, color=VAL_COLOR, linewidth=2, zorder=3, label="validation loss")
    for xs, ys, c in ((epochs, train, TRAIN_COLOR), (epochs, val, VAL_COLOR)):
        ax.scatter(xs[0], ys[0], s=36, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.annotate(f"train  {hist[0]['train_loss']:.3f}", (epochs[0], train[0]), xytext=(8, 4),
                textcoords="offset points", color=TEXT_PRIMARY, fontsize=9)
    ax.annotate(f"validation  {hist[0]['val_loss']:.4f}", (epochs[0], val[0]), xytext=(8, 4),
                textcoords="offset points", color=TEXT_PRIMARY, fontsize=9)

    ax.set_yscale("log")
    ax.set_ylim(a.floor * 0.6, 1.5)
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.5, len(epochs) + 0.5)
    ax.set_xticks([1] + list(range(10, len(epochs) + 1, 10)))
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{a.loss_name} (log scale)")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)

    ax.set_title(a.title, loc="left",
                 fontsize=12, color=TEXT_PRIMARY, pad=22)
    sub = a.subtitle.format(cycles=len(restarts) + 1)
    if any(h["train_loss"] < a.floor or h["val_loss"] < a.floor for h in hist):
        sub += f"; values below {a.floor:g} drawn at the floor"
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=8.5, color=TEXT_SECONDARY, va="bottom")
    ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT_PRIMARY)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, facecolor=SURFACE)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
