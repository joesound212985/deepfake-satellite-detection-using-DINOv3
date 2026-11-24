#!/usr/bin/env python3
"""
Collect every test_metrics.json written by dinov3_forensics_train_v2.py and
print mean +/- standard deviation per variant (full, noedge, nobound, ...).

Usage:
    python aggregate_runs.py                      # scans ~/results_dinov3
    python aggregate_runs.py /path/to/results     # scans another folder
    python aggregate_runs.py --csv table.csv      # also writes a CSV
"""
import sys, json, csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

COLUMNS = [
    ("accuracy",          lambda m: m["accuracy"]),
    ("macro_f1",          lambda m: m["macro_f1"]),
    ("balanced_acc",      lambda m: m["balanced_accuracy"]),
    ("tampered_recall",   lambda m: m["per_class"]["tampered"]["recall"]),
    ("tampered_auc",      lambda m: m["per_class"]["tampered"]["auc"]),
    ("mean_iou",          lambda m: m["localization_tampered"]["mean_iou"]),
    ("mean_dice",         lambda m: m["localization_tampered"]["mean_dice"]),
    ("mean_mcc",          lambda m: m["localization_tampered"]["mean_mcc"]),
    ("params_M",          lambda m: m["parameters"]["total"] / 1e6),
    ("ms_per_image",      lambda m: m["ms_per_test_image"]),
    ("train_s_per_epoch", lambda m: m["train_sec_per_epoch"]),
]


def main():
    args = [a for a in sys.argv[1:]]
    csv_path = None
    if "--csv" in args:
        i = args.index("--csv")
        csv_path = args[i + 1]
        del args[i:i + 2]
    root = Path(args[0]) if args else Path.home() / "results_dinov3"

    groups = defaultdict(list)
    for f in sorted(root.rglob("test_metrics.json")):
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)
        key = (Path(str(m.get("checkpoint", "?"))).name, m.get("variant", "?"))
        groups[key].append(m)

    if not groups:
        print(f"No test_metrics.json found under {root}")
        return

    rows = []
    for (ckpt, variant), runs in sorted(groups.items()):
        seeds = sorted(r.get("seed") for r in runs)
        print(f"\n{ckpt} | {variant} | n={len(runs)} | seeds={seeds}")
        row = {"checkpoint": ckpt, "variant": variant, "n_runs": len(runs), "seeds": seeds}
        for name, get in COLUMNS:
            vals = []
            for r in runs:
                try:
                    v = float(get(r))
                except (KeyError, TypeError, ValueError):
                    continue
                if v == v:  # skip NaN
                    vals.append(v)
            if not vals:
                continue
            mu = mean(vals)
            sd = stdev(vals) if len(vals) > 1 else float("nan")
            row[name + "_mean"] = mu
            row[name + "_std"] = sd
            sd_txt = f" +/- {sd:.4f}" if len(vals) > 1 else "  (single run, no std)"
            print(f"  {name:<18} {mu:.4f}{sd_txt}")
        rows.append(row)

    if csv_path:
        fields = sorted({k for r in rows for k in r})
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
