#!/usr/bin/env python3
"""Label-free transfer experiments A/B/C. Settings fixed in ZEROSHOT_EXTRAS_PROTOCOL.md. CPU only; refuses to overwrite."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
FEAT = Path.home() / "data" / "features" / "fewshot_cls384"
OUT = HERE / "zeroshot_extras_metrics.json"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwriting")
EXS = ["full_s42", "web_s42", "sat_orig", "web_orig"]
SETS = ["rsfake", "uw", "dmaer", "fmow", "gdd_indomain", "gdd_crossgenerator", "gdd_crossdomain", "gdd_localization"]
ZS = {"full_s42": "cross_{d}_metrics.per_image.json", "web_s42": "cross_web_{d}_metrics.per_image.json"}


def load(ex, d):
    meta = json.loads((FEAT / ex / f"{d}.ids.json").read_text())
    F = np.load(FEAT / ex / f"{d}.npy").astype(np.float64); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
    return F, np.array([m["label"] for m in meta]), meta


def rates(y, pred):
    tpr = float(pred[y == 1].mean()); fpr = float(pred[y == 0].mean())
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": (tpr + 1 - fpr) / 2, "detection_rate": tpr, "false_alarm_rate": fpr}


R = {"protocol": "ZEROSHOT_EXTRAS_PROTOCOL.md", "A_clustering": {}, "B_frozen_probe": {}, "C_one_class": {}}
# ---- A
for ex, pat in ZS.items():
    R["A_clustering"][ex] = {}
    for d in SETS:
        F, y, meta = load(ex, d)
        recs = {r["id"]: r["p_manip"] for r in json.loads((HERE / pat.format(d=d)).read_text())}
        s = np.array([recs[m["id"]] for m in meta])
        c = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(F)
        fake_c = int(s[c == 1].mean() > s[c == 0].mean()); pred = (c == fake_c).astype(int)
        r = rates(y, pred); r["oracle_naming_accuracy"] = max(r["accuracy"], 1 - r["accuracy"])
        r["cluster_sizes"] = [int((c == 0).sum()), int((c == 1).sum())]; r["zero_shot_auc_same_model"] = float(roc_auc_score(y, s))
        R["A_clustering"][ex][d] = r
# ---- B and C
for ex in EXS:
    Ftr, ytr, _ = load(ex, "own_train")
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Ftr, ytr)
    real = np.flatnonzero(ytr == 0); real = real[np.random.RandomState(0).permutation(len(real))]
    n_cal = len(real) // 10; cal, ref = real[:n_cal], real[n_cal:]
    nn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(Ftr[ref])
    oc = lambda X: nn.kneighbors(X)[0].mean(axis=1)
    thr = float(np.percentile(oc(Ftr[cal]), 95))
    R["B_frozen_probe"][ex] = {"n_train": int(len(ytr))}; R["C_one_class"][ex] = {"n_reference": int(len(ref)), "n_calibration": int(len(cal)), "threshold": thr}
    for d in ["own_test"] + SETS:
        F, y, _ = load(ex, d)
        p = clf.predict_proba(F)[:, 1]
        R["B_frozen_probe"][ex][d] = dict(rates(y, (p >= 0.5).astype(int)), auc=float(roc_auc_score(y, p)))
        s = oc(F)
        R["C_one_class"][ex][d] = dict(rates(y, (s > thr).astype(int)), auc=float(roc_auc_score(y, s)))
OUT.write_text(json.dumps(R, indent=1)); print("[save]", OUT)
for part in ("A_clustering", "B_frozen_probe", "C_one_class"):
    print("\n" + part)
    for ex, D in R[part].items():
        print(" ", ex, " | ".join(f"{d}: " + (f"AUC={v['auc']:.3f} " if 'auc' in v else "") + f"acc={v['accuracy']:.3f}" for d, v in D.items() if isinstance(v, dict)))
