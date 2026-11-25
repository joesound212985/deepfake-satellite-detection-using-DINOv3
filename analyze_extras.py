#!/usr/bin/env python3
"""
Post-hoc analyses from saved per-image evaluation files only (no model is run here).
Writes analysis_extras.json. Sections are skipped (and listed under "missing") if their inputs do not exist yet.

  1. two_class_test       main test set scored real-vs-manipulated, next to the SimLBR baseline
  2. rsfake_per_generator detection rate and AUC (each generator's fakes vs all reals)
  3. robustness           test-set scores after JPEG re-compression / down-scaling
  4. bootstrap_ci         95% percentile intervals (1000 resamples of images, seed 42) for AUC / accuracy
  0. consistency          the clean re-evaluation must reproduce test_metrics.json of the same checkpoint
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

G = Path(__file__).resolve().parent
RES = Path.home() / "results_dinov3" / "vit"
MODELS = {"full_s42": ("", RES / "dinov3-vitl16-pretrain-sat493m/full_s42"),
          "web_s42": ("web_", RES / "dinov3-vitl16-pretrain-lvd1689m/web_s42")}
CROSS = ["fmow", "rsfake", "uw", "dmaer"]
DEGRADE = ["jpeg90", "jpeg70", "jpeg50", "jpeg30", "scale0.75", "scale0.5"]
B, SEED = 1000, 42
missing = []


def load(name):
    f = G / name
    if not f.exists():
        missing.append(name)
        return None
    return json.load(open(f))


def ci(fn, n, rng):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(idx))
        except ValueError:      # resample with a single class
            continue
    vals = [v for v in vals if v == v]      # drop NaN (single-class resamples)
    if len(vals) < B // 2:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def binary_stats(recs):
    y = np.array([r["label"] for r in recs]); p = np.array([r["p_manip"] for r in recs])
    rng = np.random.default_rng(SEED)
    out = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_manipulated": int((y == 1).sum())}
    hard = (p > 0.5).astype(int)
    if out["n_real"] and out["n_manipulated"]:
        out["auc"] = float(roc_auc_score(y, p)); out["auc_ci95"] = ci(lambda i: roc_auc_score(y[i], p[i]), len(y), rng)
        out["accuracy_thr0.5"] = float((hard == y).mean())
        out["accuracy_ci95"] = ci(lambda i: float((hard[i] == y[i]).mean()), len(y), rng)
        out["f1_manipulated_thr0.5"] = float(f1_score(y, hard))
        out["false_alarm_rate"] = float(hard[y == 0].mean()); out["detection_rate"] = float(hard[y == 1].mean())
    elif out["n_manipulated"]:
        out["detection_rate"] = float(hard.mean()); out["detection_rate_ci95"] = ci(lambda i: float(hard[i].mean()), len(y), rng)
    return out


res = {"bootstrap": {"resamples": B, "seed": SEED, "unit": "image", "interval": "percentile 95%"}}

# 0 + 1 ------------------------------------------------------------------ test set, two-class
res["two_class_test"] = {}; res["consistency"] = {}
for m, (pre, rdir) in MODELS.items():
    stem = f"overhead_{m.split('_')[0]}_metrics"
    d, recs = load(stem + ".json"), load(stem + ".per_image.json")
    if d is None or recs is None:
        continue
    res["two_class_test"][m] = dict(binary_stats(recs), **{"from_argmax": d["overall"].get("two_class_from_argmax")})
    tm = json.load(open(rdir / "test_metrics.json"))
    t3 = d["overall"]["three_class"]; loc = d["overall"].get("localization", {})
    res["consistency"][m] = {
        "accuracy": [t3["accuracy"], tm["accuracy"]], "macro_f1": [t3["macro_f1"], tm["macro_f1"]],
        "mean_iou": [loc.get("mean_iou"), tm["localization_tampered"]["mean_iou"]],
        "order": "[re-evaluation with eval_cross_dataset.py, test_metrics.json]",
        "agree_within_0.005": bool(abs(t3["accuracy"] - tm["accuracy"]) < 0.005 and abs(t3["macro_f1"] - tm["macro_f1"]) < 0.005)}
for name in ("simlbr_dmaer_metrics.json", "simlbr_nolbr_dmaer_metrics.json"):
    f = G / name
    if not f.exists():
        f = G / name.replace("_dmaer", "")
    if f.exists():
        res["two_class_test"]["simlbr" + ("_nolbr" if "nolbr" in name else "")] = dict(
            json.load(open(f))["mean_std_over_seeds"]["overhead_test"], source=f.name)

# 2 + 4 ------------------------------------------------------------------ external sets
res["cross_dataset"] = {}; res["rsfake_per_generator"] = {}
for m, (pre, _) in MODELS.items():
    res["cross_dataset"][m] = {}
    for c in CROSS:
        recs = load(f"cross_{pre}{c}_metrics.per_image.json")
        if recs is None:
            continue
        res["cross_dataset"][m][c] = binary_stats(recs)
        loc = [r for r in recs if "iou" in r]
        if loc:
            iou = np.array([r["iou"] for r in loc]); rng = np.random.default_rng(SEED)
            res["cross_dataset"][m][c]["mean_iou"] = float(iou.mean())
            res["cross_dataset"][m][c]["mean_iou_ci95"] = ci(lambda i: float(iou[i].mean()), len(iou), rng)
        if c == "rsfake":
            reals = [r for r in recs if r["label"] == 0]
            per = {}
            for g in sorted({r["group"] for r in recs if r["label"] == 1}):
                per[g] = binary_stats([r for r in recs if r["group"] == g] + reals)
            res["rsfake_per_generator"][m] = per
    for c in ("allreals_dmaer",):
        f = G / f"cross_{pre}{c}_metrics.per_image.json"
        if f.exists():
            res["cross_dataset"][m]["dmaer_allreals_sensitivity"] = binary_stats(json.load(open(f)))

# 3 ---------------------------------------------------------------------- robustness
res["robustness"] = {}
for m in MODELS:
    short = m.split("_")[0]; rows = {}
    for cond in ["clean"] + DEGRADE:
        d = load(f"overhead_{short}_metrics.json" if cond == "clean" else f"robust_{short}_{cond}_metrics.json")
        if d is None:
            continue
        o = d["overall"]
        rows[cond] = {"accuracy": o["three_class"]["accuracy"], "macro_f1": o["three_class"]["macro_f1"],
                      "tampered_recall": o["three_class"]["tampered_recall"],
                      "auc_real_vs_manipulated": o.get("auc_real_vs_manipulated"),
                      "false_alarm_rate": o.get("false_alarm_rate_real_thr0.5"),
                      "mean_iou_tampered": o.get("localization", {}).get("mean_iou")}
    res["robustness"][m] = rows

res["missing_inputs"] = sorted(set(missing))
(G / "analysis_extras.json").write_text(json.dumps(res, indent=2))
print(json.dumps({k: v for k, v in res.items() if k in ("consistency", "two_class_test", "robustness", "missing_inputs")}, indent=1)[:6000])
print("[save] analysis_extras.json")
