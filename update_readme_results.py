#!/usr/bin/env python3
"""Rewrites the results block of README.md (between the two marker lines) from result files only:
table.csv (aggregate_runs.py), cross_detail.json + cross_*_metrics.json, fewshot_metrics.json and
fewshot_gdd_metrics.json. No number is typed by hand. Run it again whenever those files change.

  python update_readme_results.py
"""
import csv, json, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
START, END = "<!-- results:start -->", "<!-- results:end -->"
J = lambda n: json.load(open(H / n)) if (H / n).exists() else None

ROWS = [("full", "Full (satellite pretraining)"), ("plain", "Plain decoder"), ("web", "Web pretraining"),
        ("frozen", "Frozen backbone"), ("resnet", "ResNet-50, backbone LR 2e-6"),
        ("resnetlr", "ResNet-50, backbone LR 1e-4"), ("radio", "C-RADIOv4")]
SETS = [("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER (size-matched)"),
        ("gdd_indomain", "GDD in-domain"), ("gdd_crossgenerator", "GDD cross-generator"),
        ("gdd_crossdomain", "GDD cross-domain"), ("gdd_localization", "GDD local edits"),
        ("fmow", "fmow-fake-small (120 images)")]


def ms(row, key, digits=3):
    m, s, n = float(row[key + "_mean"]), row[key + "_std"], int(row["n_runs"])
    return f"{m:.{digits}f} ± {float(s):.{digits}f}" if n > 1 else f"{m:.{digits}f}"


out = [START, "", f"*Generated {datetime.date.today().isoformat()} by `update_readme_results.py` from the result "
       "files in this repository. Rows with one seed are single runs; the others are mean ± standard deviation "
       "over the listed seeds.*", ""]

# ---- own test set
table = {r["variant"]: r for r in csv.DictReader(open(H / "table.csv"))} if (H / "table.csv").exists() else {}
out += ["### Own test set (3,150 images; three classes, tamper mask on the 888 tampered images)", "",
        "| Row | Seeds | Accuracy | Macro-F1 | Balanced acc. | Tampered recall | Mask IoU | Mask Dice | Mask MCC | ms / image |",
        "|---|---|---|---|---|---|---|---|---|---|"]
missing = []
for key, label in ROWS:
    r = table.get(key)
    if not r:
        missing.append(label); continue
    seeds = ", ".join(str(s) for s in json.loads(r["seeds"]))
    out.append(f"| {label} | {seeds} | {ms(r, 'accuracy')} | {ms(r, 'macro_f1')} | {ms(r, 'balanced_acc')} | "
               f"{ms(r, 'tampered_recall')} | {ms(r, 'mean_iou')} | {ms(r, 'mean_dice')} | {ms(r, 'mean_mcc')} | "
               f"{float(r['ms_per_image_mean']):.1f} |")
out.append("")
if missing:
    out += ["Not in the table yet: " + "; ".join(missing) + ".", ""]

# ---- zero-shot
cd = J("cross_detail.json")
if cd:
    out += ["### Zero-shot on outside datasets (no adaptation; seed-42 checkpoints)", "",
            "AUC of real versus manipulated with a bootstrap 95% interval, and the detection / false-alarm rates "
            "at the fixed threshold 0.5.", "",
            "| Dataset | Real / manip. | Full: AUC [95% CI] | Full: det. / FA @0.5 | Web: AUC [95% CI] | Web: det. / FA @0.5 |",
            "|---|---|---|---|---|---|"]
    for k, label in SETS:
        cells = []
        n = ""
        for pre in ("cross_", "cross_web_"):
            c = cd["files"].get(pre + k); m = J(f"{pre}{k}_metrics.json")
            if not c or not m:
                cells += ["pending", "pending"]; continue
            o = m["overall"]; n = f"{o['n_real']} / {o['n_manipulated']}"
            cells += [f"{c['auc']:.3f} [{c['auc_ci95'][0]:.3f}, {c['auc_ci95'][1]:.3f}]",
                      f"{o['detection_rate_manipulated_thr0.5']:.3f} / {o['false_alarm_rate_real_thr0.5']:.3f}"]
        out.append(f"| {label} | {n} | " + " | ".join(cells) + " |")
    out.append("")

# ---- few-shot
fs = {}
for f in ("fewshot_metrics.json", "fewshot_gdd_metrics.json"):
    d = J(f)
    if d:
        fs.update(d["datasets"])
if fs:
    f3 = lambda d: "–" if not isinstance(d, dict) else f"{d['mean']:.3f} ± {d['std']:.3f}"
    out += ["### Few-shot adaptation (linear probe on the frozen fine-tuned satellite backbone, seed 42)", "",
            "AUC on the held-out half of each outside dataset with k labelled images per class "
            "(10 random draws; protocol in `FEWSHOT_PROTOCOL.md`).", "",
            "| Dataset | Zero-shot | k = 4 | k = 16 | k = 64 | k = 256 | Whole pool |", "|---|---|---|---|---|---|---|"]
    for k, label in SETS:
        d = fs.get(k)
        if not d:
            out.append(f"| {label} | pending | | | | | |"); continue
        e = d["extractors"]["full_s42"]; z = d["zero_shot"].get("full_s42")
        out.append(f"| {label} | " + (f"{z['auc']:.3f}" if isinstance(z, dict) else "–") + " | "
                   + " | ".join(f3(e[f"k{n}"]["auc"]) if isinstance(e.get(f"k{n}"), dict) else "–" for n in (4, 16, 64, 256))
                   + f" | {e['whole_pool']['auc']:.3f} |")
    out.append("")
out.append(END)

readme = (H / "README.md").read_text()
a, b = readme.index(START), readme.index(END) + len(END)
(H / "README.md").write_text(readme[:a] + "\n".join(out) + readme[b:])
print("README.md results block rewritten;", "missing rows: " + ", ".join(missing) if missing else "all rows present")
