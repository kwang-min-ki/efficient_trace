"""Assemble the TRACE-hf vs Likelihood-TRACE-hf comparison table from the .stats files
each script writes and the f1.jsonl detect.py f1 appends to.

Expects, per {tag} = "{impl}_{window}_{loophole}" (e.g. "trace_hf_none_ic",
"lhf_full_rm"), two run outputs from trace_hf.py / likelihood_trace_hf.py:
  runs/{tag}_hack.jsonl(.stats)     -- hacking checkpoint
  runs/{tag}_nonhack.jsonl(.stats)  -- non-hacking checkpoint
and one row in the f1 results file (detect.py f1 --tag {tag} --out runs/f1_<domain>.jsonl).

  python make_table.py --f1 runs/f1_math.jsonl --run-dir runs --out table.md
"""

import argparse
import json
from pathlib import Path

from data import read_jsonl


def load_stats(path):
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1", required=True, help="detect.py f1 output, one row per --tag")
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    f1_rows = {r["tag"]: r for r in read_jsonl(args.f1)}
    run_dir = Path(args.run_dir)

    cols = ["tag", "impl", "score_window", "hack_mean_auc", "nonhack_mean_auc",
            "separation", "f1", "precision", "recall",
            "hack_scoring_s", "nonhack_scoring_s", "hack_wall_s", "nonhack_wall_s"]
    rows = []
    for tag in sorted(f1_rows):
        hack = load_stats(run_dir / f"{tag}_hack.jsonl.stats")
        nonhack = load_stats(run_dir / f"{tag}_nonhack.jsonl.stats")
        f1 = f1_rows[tag]
        h_auc = hack["mean_auc"] if hack else None
        n_auc = nonhack["mean_auc"] if nonhack else None
        rows.append({
            "tag": tag,
            "impl": (hack or nonhack or {}).get("impl", ""),
            "score_window": (hack or nonhack or {}).get("score_window", ""),
            "hack_mean_auc": h_auc, "nonhack_mean_auc": n_auc,
            "separation": (h_auc - n_auc) if h_auc is not None and n_auc is not None else None,
            "f1": f1["trace"]["f1"],
            "precision": f1["trace"]["precision"],
            "recall": f1["trace"]["recall"],
            # scoring_s excludes the shared rollout-generation step -- the fair, apples-to-apples
            # TRACE-vs-likelihood-TRACE comparison; wall_s = rollout_s + scoring_s, for reference.
            "hack_scoring_s": (hack or {}).get("scoring_time_s"),
            "nonhack_scoring_s": (nonhack or {}).get("scoring_time_s"),
            "hack_wall_s": hack["wall_clock_s"] if hack else None,
            "nonhack_wall_s": nonhack["wall_clock_s"] if nonhack else None,
        })

    def fmt(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
