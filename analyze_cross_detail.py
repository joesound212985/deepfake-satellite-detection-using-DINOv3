#!/usr/bin/env python3
"""Added 2026-11-26 (Joe approved). CPU only; reads the saved per-image zero-shot scores, trains nothing.
For every cross-dataset file: bootstrap 95% CI of AUC, detection rate at 5% false alarms (TPR@5%FPR) with CI,
and per-group AUC (each manipulated group against all reals of that set).
Bootstrap: 2000 resamples, images resampled within class, numpy RandomState(0), percentile interval.
TPR@5%FPR threshold = 95th percentile of the real scores OF THE SAME SET (an oracle operating point that
needs real images from the target source; it is not a zero-shot number). Writes cross_detail.json and
CROSS_DETAIL.md; refuses to overwrite."""
import json, sys, glob
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "cross_detail.json"; MD = HERE / "CROSS_DETAIL.md"
if OUT.exists() or MD.exists():
    sys.exit("cross_detail.json / CROSS_DETAIL.md exist; not overwriting")
B = 2000


def tpr5(y, s):
    thr = np.quantile(s[y == 0], 0.95)
    return float((s[y == 1] > thr).mean())


def boot(y, s, fn):
    rs = np.random.RandomState(0); r = np.flatnonzero(y == 0); f = np.flatnonzero(y == 1); v = []
    for _ in range(B):
        i = np.concatenate([rs.choice(r, len(r)), rs.choice(f, len(f))]); v.append(fn(y[i], s[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


res = {"bootstrap": {"resamples": B, "seed": 0, "within_class": True, "interval": "percentile 95%"},
       "tpr_at_5pct_fpr": "threshold = 95th percentile of the same set's real scores (oracle operating point)", "files": {}}
lines = ["# Cross-dataset detail (zero-shot scores)", "", "Made by `analyze_cross_detail.py` from the saved per-image files. "
         "AUC and detection rate at 5% false alarms with bootstrap 95% intervals (2000 resamples within class, seed 0). "
         "The 5% false-alarm threshold is set on the same set's real images, so that column is an oracle operating point, not zero-shot.", "",
         "| file | n real | n manip. | AUC [95% CI] | detection @5% FA [95% CI] |", "|---|---|---|---|---|"]
groups_md = []
for f in sorted(glob.glob(str(HERE / "cross*_metrics.per_image.json"))):
    name = Path(f).name.replace("_metrics.per_image.json", "")
    d = json.load(open(f)); y = np.array([r["label"] for r in d]); s = np.array([r["p_manip"] for r in d]); g = np.array([r["group"] for r in d])
    R = {"n_real": int((y == 0).sum()), "n_manipulated": int((y == 1).sum()),
         "auc": float(roc_auc_score(y, s)), "auc_ci95": boot(y, s, roc_auc_score),
         "tpr_at_5pct_fpr": tpr5(y, s), "tpr_at_5pct_fpr_ci95": boot(y, s, tpr5), "per_group": {}}
    for gg in sorted(set(g[y == 1])):
        m = (y == 0) | (g == gg)
        R["per_group"][gg] = {"n": int((g == gg).sum()), "auc_vs_all_reals": float(roc_auc_score(y[m], s[m])),
                              "detection_rate_thr0.5": float((s[g == gg] >= 0.5).mean()), "tpr_at_5pct_fpr": tpr5(y[m], s[m])}
    res["files"][name] = R
    lines.append(f"| {name} | {R['n_real']} | {R['n_manipulated']} | {R['auc']:.3f} [{R['auc_ci95'][0]:.3f}, {R['auc_ci95'][1]:.3f}] | "
                 f"{R['tpr_at_5pct_fpr']:.3f} [{R['tpr_at_5pct_fpr_ci95'][0]:.3f}, {R['tpr_at_5pct_fpr_ci95'][1]:.3f}] |")
    groups_md += ["", f"### {name}", "", "| manipulated group | n | AUC vs all reals | detected @0.5 | detected @5% FA |", "|---|---|---|---|---|"]
    groups_md += [f"| {k} | {v['n']} | {v['auc_vs_all_reals']:.3f} | {v['detection_rate_thr0.5']:.3f} | {v['tpr_at_5pct_fpr']:.3f} |" for k, v in R["per_group"].items()]
OUT.write_text(json.dumps(res, indent=1)); MD.write_text("\n".join(lines + ["", "## Per manipulated group"] + groups_md) + "\n")
print("\n".join(lines[4:]))
