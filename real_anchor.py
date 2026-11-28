#!/usr/bin/env python3
"""Real-anchored matched filter: a manipulation signature learned on our own training data, applied to a
new source after estimating that source's mean and covariance from k authentic images (no manipulated image
of the new source is used). Settings fixed in REAL_ANCHOR_PROTOCOL.md.
CPU only, cached features; refuses to overwrite its output."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
FEAT = Path.home() / "data" / "features" / "fewshot_cls384"
OUT = HERE / "real_anchor_metrics.json"
if OUT.exists():
    sys.exit(f"{OUT} exists; not overwriting")
SETS = ["rsfake", "uw", "dmaer", "fmow", "gdd_indomain", "gdd_crossgenerator", "gdd_crossdomain", "gdd_localization"]
EXS = ["full_s42", "full_s43", "full_s44", "web_s42", "web_s43", "web_s44", "sat_orig", "web_orig"]
KS = [4, 16, 64, 256]; DRAWS = 10
SPLITS = {**json.loads((HERE / "fewshot_splits.json").read_text()), **json.loads((HERE / "fewshot_gdd_splits.json").read_text())}


def l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def load(ex, d):
    meta = json.loads((FEAT / ex / f"{d}.ids.json").read_text())
    F = l2(np.load(FEAT / ex / f"{d}.npy").astype(np.float64))
    return F, np.array([m["label"] for m in meta]), meta


def agg(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "n_draws": len(v)}


def aggd(rows):
    return {k: agg([r[k] for r in rows]) for k in rows[0]}


def rates(y, s, thr):
    pred = s >= thr
    return {"auc": float(roc_auc_score(y, s)), "accuracy": float((pred == y).mean()),
            "detection_rate": float(pred[y == 1].mean()), "false_alarm_rate": float(pred[y == 0].mean())}


def per_group(y, grp, s, thr, groups):
    """Each manipulated group alone against all held-out real images (as in fewshot_logo.py)."""
    out = {}
    for g in groups:
        m = (y == 0) | (grp == g)
        out[g] = rates(y[m], s[m], thr)
    out["_mean_over_groups_auc"] = float(np.mean([out[g]["auc"] for g in groups]))
    return out


Z95 = 1.6448536269514722      # one-sided 5% point of the standard normal: constant-false-alarm threshold
R = {"protocol": "REAL_ANCHOR_PROTOCOL.md", "ks": KS, "draws": DRAWS, "cfar_threshold": Z95, "extractors": {}}


def lw_cov(X):
    """Ledoit-Wolf covariance of the rows of X about their own mean (no free parameter)."""
    return LedoitWolf().fit(X - X.mean(axis=0)).covariance_


def amf(X, mu, cov, delta):
    """Adaptive matched filter: delta' C^-1 (x - mu) / sqrt(delta' C^-1 delta); N(0,1) for images drawn like the reference."""
    u = np.linalg.solve(cov, delta)
    return (X - mu) @ u / np.sqrt(delta @ u)


def rx(X, mu, cov):
    """Reed-Xiaoli one-class score: squared Mahalanobis distance to the reference real images."""
    Z = X - mu
    return np.einsum("ij,ij->i", Z, np.linalg.solve(cov, Z.T).T)


for ex in EXS:
    Ftr, ytr, _ = load(ex, "own_train")
    real_tr = Ftr[ytr == 0]
    mu_s = real_tr.mean(axis=0); cov_s = lw_cov(real_tr)
    delta = Ftr[ytr == 1].mean(axis=0) - mu_s          # manipulation signature, training folder only
    E = {"n_train": int(len(ytr)), "n_train_real": int(len(real_tr)), "sets": {}}
    for d in ["own_test"] + SETS:
        F, y, meta = load(ex, d)
        if d == "own_test":      # sanity row: source statistics on our own test folder
            E["sets"][d] = {"no_anchor": rates(y, amf(F, mu_s, cov_s, delta), Z95)}
            continue
        ids = [m["id"] for m in meta]; pos = {i: n for n, i in enumerate(ids)}
        grp = np.array([m["group"] for m in meta])
        pool = np.array(sorted(pos[i] for i in SPLITS[d]["pool"])); held = np.array(sorted(pos[i] for i in SPLITS[d]["held_out"]))
        assert len(pool) + len(held) == len(meta) and not set(pool) & set(held)
        pr = pool[y[pool] == 0]
        yh, gh, Fh = y[held], grp[held], F[held]
        groups = sorted(set(grp[y == 1]))
        D = {"held_out_real": int((yh == 0).sum()), "held_out_manipulated": int((yh == 1).sum()), "pool_real": int(len(pr))}
        s0 = amf(Fh, mu_s, cov_s, delta)      # reference: no image of the new source
        D["no_anchor"] = dict(rates(yh, s0, Z95), per_group=per_group(yh, gh, s0, Z95, groups))

        def run(shots):
            mu_t = F[shots].mean(axis=0); cov_t = lw_cov(F[shots])
            out = {}
            for name, s in (("anchor_mean", amf(Fh, mu_t, cov_s, delta)),          # same ranking as no_anchor; only the threshold moves
                            ("anchor_matched_filter", amf(Fh, mu_t, cov_t, delta)),  # primary
                            ("anchor_rx", rx(Fh, mu_t, cov_t))):                     # no signature at all
                thr = np.inf if name == "anchor_rx" else Z95
                out[name] = (rates(yh, s, thr), per_group(yh, gh, s, thr, groups))
            return out

        for k in KS:
            key = f"k{k}"
            if k > len(pr):
                D[key] = "skipped: pool has fewer than k real images"; continue
            draws = [run(np.random.RandomState(seed).choice(pr, k, replace=False)) for seed in range(DRAWS)]
            D[key] = {}
            for name in draws[0]:
                D[key][name] = aggd([dr[name][0] for dr in draws])
                D[key][name]["mean_over_groups_auc"] = agg([dr[name][1]["_mean_over_groups_auc"] for dr in draws])
                D[key][name]["per_group_auc"] = {g: agg([dr[name][1][g]["auc"] for dr in draws]) for g in groups}
        whole = run(pr)
        D["whole_pool"] = {name: dict(whole[name][0], mean_over_groups_auc=whole[name][1]["_mean_over_groups_auc"],
                                      per_group_auc={g: whole[name][1][g]["auc"] for g in groups}, n_real=int(len(pr)))
                           for name in whole}
        E["sets"][d] = D
        show = {key: {name: round(D[key][name]["auc"]["mean"] if isinstance(D[key][name]["auc"], dict) else D[key][name]["auc"], 3)
                      for name in D[key]} for key in [f"k{k}" for k in KS] + ["whole_pool"] if isinstance(D[key], dict)}
        print(ex, d, "no_anchor", round(D["no_anchor"]["auc"], 3), show, flush=True)
    R["extractors"][ex] = E
OUT.write_text(json.dumps(R, indent=1)); print("[save]", OUT)
