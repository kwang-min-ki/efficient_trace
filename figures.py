"""Redraw the paper's figures from the jsonl the other scripts wrote.

  python figures.py curves   --hacking runs/hacking.jsonl --nonhacking runs/nonhacking.jsonl --out fig7.png
  python figures.py trace    --hacking sweep_h.jsonl --nonhacking sweep_n.jsonl --out fig8.png
  python figures.py f1-bars  --results f1.jsonl --out fig10.png
  python figures.py f1-curve --results f1_by_step.jsonl --out fig11.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import read_jsonl
from trace import FRACS

RED, BLUE = "#d62728", "#1f77b4"


def save(out):
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"wrote {out}")


def curves(args):
    """Fig. 7 / 19 / 21: E[R-hat] vs CoT percentage."""
    plt.figure(figsize=(5, 4))
    for path, label, color in [(args.hacking, "Hacking Model", RED),
                               (args.nonhacking, "Non-Hacking Model", BLUE)]:
        if not path:
            continue
        records = list(read_jsonl(path))
        mean = [sum(c) / len(records) for c in zip(*(r["curve"] for r in records))]
        plt.plot([f * 100 for f in FRACS], mean, label=label, color=color, marker="o")
    plt.xlabel("CoT Percentage (%)")
    plt.ylabel("Avg Passing Rate")
    plt.legend()
    save(args.out)


def trace(args):
    """Fig. 8 / 18: TRACE score over training steps. Rows are {step, mean_auc}."""
    plt.figure(figsize=(5, 4))
    for path, label, color in [(args.hacking, "Hacking Model", RED),
                               (args.nonhacking, "Non-Hacking Model", BLUE)]:
        if not path:
            continue
        rows = sorted(read_jsonl(path), key=lambda r: r["step"])
        plt.plot([r["step"] for r in rows], [r["mean_auc"] for r in rows],
                 label=label, color=color, marker="o")
    plt.xlabel("Training Steps")
    plt.ylabel("TRACE Score")
    plt.legend()
    save(args.out)


def f1_bars(args):
    """Fig. 9 / 10: F1 per model, CoT monitor vs TRACE."""
    rows = [r for r in read_jsonl(args.results) if "monitor" in r]
    x = range(len(rows))
    plt.figure(figsize=(1.8 * len(rows) + 2, 4))
    for off, key, label, color in [(-0.2, "monitor", "CoT Monitor", BLUE),
                                   (0.2, "trace", "TRACE", RED)]:
        bars = plt.bar([i + off for i in x], [r[key]["f1"] for r in rows], 0.35,
                       label=label, color=color)
        plt.bar_label(bars, fmt="%.3f", fontsize=9)
    plt.xticks(list(x), [r["tag"] for r in rows])
    plt.ylim(0, 1.1)
    plt.ylabel("F1 Score")
    plt.legend()
    save(args.out)


def f1_curve(args):
    """Fig. 11 / 12: F1 over training steps."""
    rows = sorted((r for r in read_jsonl(args.results) if r.get("step") is not None),
                  key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    plt.figure(figsize=(5, 4))
    plt.plot(steps, [r["trace"]["f1"] for r in rows], label="TRACE", color=RED, marker="o")
    if all("monitor" in r for r in rows):
        plt.plot(steps, [r["monitor"]["f1"] for r in rows], label="CoT Monitoring",
                 color=BLUE, marker="s", linestyle="--")
    plt.xlabel("Training Steps")
    plt.ylabel("F1 Score")
    plt.legend()
    save(args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, needs in [("curves", curves, "pair"), ("trace", trace, "pair"),
                            ("f1-bars", f1_bars, "results"), ("f1-curve", f1_curve, "results")]:
        p = sub.add_parser(name)
        if needs == "pair":
            p.add_argument("--hacking", required=True)
            p.add_argument("--nonhacking")
        else:
            p.add_argument("--results", required=True)
        p.add_argument("--out", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
