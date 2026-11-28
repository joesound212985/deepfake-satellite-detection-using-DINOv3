#!/usr/bin/env python3
"""Held-out-generator few-shot probes. Protocol: FEWSHOT_PROTOCOL.md, addendum 2026-11-27 (evening).
CPU only, cached features; refuses to overwrite its output."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FEAT = Path.home() / "data" / "features" / "fewshot_cls384"
OUT = HERE / "fewshot_logo_metrics.json"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwriting")
SETS = ["rsfake", "uw", "fmow", "gdd_indomain", "gdd_crossgenerator", "gdd_crossdomain", "gdd_localization"]
EXS = ["full_s42", "full_s43", "full_s44", "web_s42", "web_s43", "web_s44", "sat_orig", "web_orig"]
KS = [4, 16, 64, 256]; DRAWS = 10
SPLITS = {**json.loads((HERE / "fewshot_splits.json").read_text()), **json.loads((HERE / "fewshot_gdd_splits.json").read_text())}


def agg(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "n_draws": len(v)}


def score(F, y, tr, te):
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(F[tr], y[tr])
    s = clf.predict_proba(F[te])[:, 1]; pred = s >= 0.5
    return {"auc": float(roc_auc_score(y[te], s)), "accuracy": float((pred == y[te]).mean()),
            "detection_rate": float(pred[y[te] == 1].mean()), "false_alarm_rate": float(pred[y[te] == 0].mean())}


def aggd(rows):
    return {k: agg([r[k] for r in rows]) for k in rows[0]}


R = {"protocol": "FEWSHOT_PROTOCOL.md, addendum 2026-11-27 (evening)", "ks": KS, "draws": DRAWS, "datasets": {}}
for d in SETS:
    meta = json.loads((FEAT / "full_s42" / f"{d}.ids.json").read_text())
    ids = [m["id"] for m in meta]; pos = {i: n for n, i in enumerate(ids)}
    y = np.array([m["label"] for m in meta]); grp = np.array([m["group"] for m in meta])
    pool = np.array(sorted(pos[i] for i in SPLITS[d]["pool"])); held = np.array(sorted(pos[i] for i in SPLITS[d]["held_out"]))
    assert len(pool) + len(held) == len(meta) and not set(pool) & set(held)
    pr, pm = pool[y[pool] == 0], pool[y[pool] == 1]
    D = {"groups": {}, "extractors": {}}
    mg = sorted(set(grp[y == 1]))
    for g in mg:
        D["groups"][g] = {"held_out_manipulated": int(((grp[held] == g)).sum()), "held_out_real": int((y[held] == 0).sum()),
                          "pool_manipulated_other_groups": int((grp[pm] != g).sum()), "pool_real": int(len(pr))}
    for ex in EXS:
        m2 = json.loads((FEAT / ex / f"{d}.ids.json").read_text())
        assert [m["id"] for m in m2] == ids, f"id order differs: {ex}/{d}"
        F = np.load(FEAT / ex / f"{d}.npy").astype(np.float64); F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        E = {}
        for g in mg:
            te = held[(y[held] == 0) | (grp[held] == g)]
            pmo = pm[grp[pm] != g]
            G = {"unseen": {}, "seen": {}}
            for k in KS:
                if k > min(len(pr), len(pmo)):
                    G["unseen"][f"k{k}"] = "skipped: fewer than k images in one class"
                else:
                    rows = []
                    for seed in range(DRAWS):
                        rs = np.random.RandomState(seed)
                        tr = np.concatenate([rs.choice(pr, k, replace=False), rs.choice(pmo, k, replace=False)])
                        rows.append(score(F, y, tr, te))
                    G["unseen"][f"k{k}"] = aggd(rows)
                if k > min(len(pr), len(pm)):
                    G["seen"][f"k{k}"] = "skipped: fewer than k images in one class"
                else:
                    rows = []
                    for seed in range(DRAWS):      # the shots of the original protocol (fewshot_adapt.draw)
                        rs = np.random.RandomState(seed)
                        tr = np.concatenate([rs.choice(pr, k, replace=False), rs.choice(pm, k, replace=False)])
                        rows.append(score(F, y, tr, te))
                    G["seen"][f"k{k}"] = aggd(rows)
            G["unseen"]["whole_pool"] = dict(score(F, y, np.concatenate([pr, pmo]), te), n_train=int(len(pr) + len(pmo)))
            G["seen"]["whole_pool"] = dict(score(F, y, pool, te), n_train=int(len(pool)))
            E[g] = G
        # mean over held-out groups of the draw-mean AUC
        S = {}
        for mode in ("unseen", "seen"):
            S[mode] = {}
            for n in [f"k{k}" for k in KS] + ["whole_pool"]:
                v = [E[g][mode][n] for g in mg]
                if any(isinstance(x, str) for x in v):
                    S[mode][n] = None; continue
                S[mode][n] = float(np.mean([x["auc"]["mean"] if isinstance(x["auc"], dict) else x["auc"] for x in v]))
        E["_mean_over_groups_auc"] = S
        D["extractors"][ex] = E
        print(d, ex, {m: {n: (None if v is None else round(v, 3)) for n, v in S[m].items()} for m in S}, flush=True)
    R["datasets"][d] = D
OUT.write_text(json.dumps(R, indent=1)); print("[save]", OUT)
