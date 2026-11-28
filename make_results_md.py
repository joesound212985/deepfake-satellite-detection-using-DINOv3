#!/usr/bin/env python3
"""Writes RESULTS.md from result files only (no number is typed by hand). Refuses to overwrite: pass --force
to rebuild after the result files changed.   python make_results_md.py [--force]"""
import csv, json, sys, statistics as st, datetime
from pathlib import Path

H = Path(__file__).resolve().parent
R = Path.home() / "results_dinov3"
OUT = H / "RESULTS.md"
if OUT.exists() and "--force" not in sys.argv:
    sys.exit("RESULTS.md exists; use --force to rebuild")
J = lambda n: json.load(open(H / n)) if (H / n).exists() else None
L = []
h = lambda s, n=2: L.extend(["", "#" * n + " " + s, ""])
p = lambda s: L.extend([s, ""])


def t(head, rows):
    L.append("| " + " | ".join(head) + " |"); L.append("|" + "---|" * len(head))
    L.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows); L.append("")


f3 = lambda v: "–" if v is None else f"{v:.3f}"
f4 = lambda v: "–" if v is None else f"{v:.4f}"
msd = lambda d, k=3: "–" if not isinstance(d, dict) else (f"{d['mean']:.{k}f} ± {d['std']:.{k}f}" if d.get("std") is not None else f"{d['mean']:.{k}f}")
SETS = [("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER (size-matched)"),
        ("gdd_indomain", "GDD in-domain"), ("gdd_crossgenerator", "GDD cross-generator"), ("gdd_crossdomain", "GDD cross-domain"),
        ("gdd_localization", "GDD local edits"), ("fmow", "fmow-fake-small")]
ROWS = [("full", "Full (satellite)"), ("plain", "Plain decoder (satellite)"), ("web", "Web pretraining"), ("frozen", "Frozen backbone"),
        ("resnet", "ResNet-50, backbone LR 2e-6"), ("resnetlr", "ResNet-50, backbone LR 1e-4"), ("radio", "C-RADIOv4-SO400M"),
        ("fullnorm", "Full, backbone normalisation (side experiment)"), ("webnorm", "Web, backbone normalisation (side experiment)")]

runs = {}                                   # run_tag -> test_metrics.json
for f in sorted(R.glob("*/*/*/test_metrics.json")):
    m = json.load(open(f))
    if m.get("run_tag") and m["run_tag"] == f.parent.name: runs[m["run_tag"]] = m      # folder name must equal the tag (skips renamed copies)

L.append("# Results")
p(f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} by `make_results_md.py` from the files named in each section. "
  "Rows with one seed are single runs; the others are mean ± standard deviation over the listed seeds (sample standard deviation, "
  "as written by `aggregate_runs.py`).")

# ------------------------------------------------------------------ environment
h("1. Environment")
rows = []
for tag in sorted(runs):
    m = runs[tag]
    if any(tag.startswith(x) for x in ("full_s", "web_s", "plain_s", "frozen_s", "resnet", "radio_s", "fullnorm", "webnorm")):
        rows.append([tag, m.get("gpu"), m.get("torch_version"), m.get("transformers_version"), m.get("config", {}).get("amp_dtype"),
                     m.get("best_epoch"), f"{m.get('train_sec_per_epoch', float('nan')):.0f}", f"{m.get('ms_per_test_image', float('nan')):.1f}"])
t(["Run", "GPU", "torch", "transformers", "AMP", "Best epoch (validation)", "s / epoch", "ms / test image"], rows)
p("Source: `test_metrics.json` of each run. Timings depend on the machine and on what else ran on it; runs on the laptop GPU are not "
  "comparable in time with runs on the workstation GPU.")
c = runs["full_s42"]["config"]
p("Settings of the full model (`full_s42/test_metrics.json`, `config`): " + ", ".join(f"{k} = {v}" for k, v in c.items()) + ".")
fl = J("flops.json")
if fl:
    t(["Model", "GFLOPs", "GMACs", "Parameters (M)"], [[k, f"{v['gflops']:.1f}", f"{v['gmacs']:.1f}", f"{v['params_M']:.2f}"] for k, v in fl["results"].items()])
    p(f"Source: `flops.json` (input {fl['input']}; {fl['convention']}).")

# ------------------------------------------------------------------ data
h("2. Test set")
m = runs["full_s42"]
p(f"{m['n_test']} test images: " + ", ".join(f"{k} {v['support']}" for k, v in m["per_class"].items()) +
  f"; tamper masks on {m['localization_tampered']['n_images']} tampered images (`full_s42/test_metrics.json`).")

# ------------------------------------------------------------------ main table
h("3. Main table (own test set)")
table = {r["variant"]: r for r in csv.DictReader(open(H / "table.csv"))}
g = lambda r, k, d=3: (f"{float(r[k + '_mean']):.{d}f} ± {float(r[k + '_std']):.{d}f}" if int(r["n_runs"]) > 1 else f"{float(r[k + '_mean']):.{d}f}")
rows = []
for key, lab in ROWS:
    r = table.get(key)
    if not r: rows.append([lab, "missing"] + [""] * 9); continue
    rows.append([lab, ", ".join(map(str, json.loads(r["seeds"]))), g(r, "accuracy"), g(r, "macro_f1"), g(r, "balanced_acc"), g(r, "tampered_recall"),
                 g(r, "mean_iou"), g(r, "mean_dice"), g(r, "mean_mcc"), f"{float(r['params_M_mean']):.1f}", f"{float(r['ms_per_image_mean']):.0f}"])
t(["Row", "Seeds", "Accuracy", "Macro-F1", "Balanced acc.", "Tampered recall", "IoU", "Dice", "MCC", "Params (M)", "ms / image"], rows)
p("Source: `table.csv`. IoU, Dice and MCC are computed per tampered image and averaged over the 888 tampered test images. "
  "`fullnorm` / `webnorm` feed the backbone its own mean / standard-deviation normalisation instead of inputs in [0, 1]; they were run on "
  "the laptop GPU, as were `radio_s42` and `plain_s43`.")
rows = []
for key in ("full", "plain", "web"):
    for s in (42, 43, 44):
        m = runs.get(f"{key}_s{s}")
        if m: rows.append([f"{key}_s{s}", f4(m["accuracy"]), f4(m["macro_f1"]), f4(m["per_class"]["tampered"]["recall"]), f4(m["localization_tampered"]["mean_iou"]), m.get("best_epoch"), m.get("gpu")])
t(["Run", "Accuracy", "Macro-F1", "Tampered recall", "IoU", "Best epoch", "GPU"], rows)
p("Per-seed values, from each run's `test_metrics.json`.")

h("Per-class results and confusion matrices", 3)
rows = []
for key, lab in (("full", "Full (satellite)"), ("web", "Web pretraining")):
    for cls in ("pristine", "tampered", "fake"):
        vals = {k: [runs[f"{key}_s{s}"]["per_class"][cls][k] for s in (42, 43, 44)] for k in ("precision", "recall", "f1", "auc")}
        rows.append([lab, cls] + [f"{st.mean(v):.3f} ± {st.stdev(v):.3f}" for v in vals.values()])
t(["Row", "Class", "Precision", "Recall", "F1", "One-vs-rest AUC"], rows)
p("Mean ± standard deviation over seeds 42, 43, 44, from each run's `test_metrics.json`.")
for tag in ("full_s42", "web_s42"):
    cm = runs[tag]["confusion_matrix_rows_true_cols_pred"]
    t([f"{tag}: true \\ predicted", "pristine", "tampered", "fake"], [[n] + row for n, row in zip(("pristine", "tampered", "fake"), cm)])
b = runs["full_s42"]["area_buckets"]
t(["Tampered-area bucket (full_s42)", "Images", "Classified correctly", "Mean IoU"],
  [[k, v["count"], f3(v["acc_raw"]), f3(v["iou_mean"])] for k, v in b.items() if isinstance(v, dict) and k.startswith("tampered") and v["count"]])
p(f"Small / large is split at a manipulated fraction of {b['thr']} (`full_s42/test_metrics.json`, `area_buckets`).")

# ------------------------------------------------------------------ significance
a = J("analysis_extras.json")
h("4. Significance between rows")
sg = a["significance"]
p(f"{sg['test']}. {sg['note']}. Source: `analysis_extras.json`; per-image files `overhead_*_metrics.per_image.json` (re-evaluation with "
  "`eval_cross_dataset.py`, which can differ from `test_metrics.json` in the fourth decimal).")
rows = []
for pair, d in sg["pairs"].items():
    A, B = pair.split("_vs_")
    for s in ("seed42", "seed43", "seed44"):
        v = d.get(s)
        if v: rows.append([f"{A} vs {B}", s[4:], f4(v[f"acc_{A}"]), f4(v[f"acc_{B}"]), v[f"only_{A}_correct"], v[f"only_{B}_correct"], f"{v['p_value']:.2g}"])
    rows.append([f"{A} vs {B}", "mean diff.", "", "", "", "", msd(d["accuracy_difference"], 4)])
t(["Pair", "Seed", "Acc. A", "Acc. B", "Only A correct", "Only B correct", "p (McNemar) / accuracy difference A − B"], rows)

# ------------------------------------------------------------------ old model
h("5. Earlier checkpoint re-scored")
o = json.load(open(R / "vit/dinov3-vitl16-pretrain-sat493m/old_model_rescored/test_metrics.json"))   # not the BADLOAD folder (same run_tag)
lo = o["localization_tampered"]
p(f"`old_model_rescored/test_metrics.json`: accuracy {f4(o['accuracy'])}, macro-F1 {f4(o['macro_f1'])}, balanced accuracy {f4(o['balanced_accuracy'])}, "
  f"IoU {f4(lo['mean_iou'])}, Dice {f4(lo['mean_dice'])}, MCC {f4(lo['mean_mcc'])} (config: legacy_token_slice = {o['config'].get('legacy_token_slice')}). "
  "This checkpoint predates the validation split, so its selection is not comparable with the rows of the main table. The numbers of the "
  "earlier letter draft are not in any result file and are not repeated here.")

# ------------------------------------------------------------------ two-class and SimLBR
h("6. Two-class scoring and SimLBR")
tc = a["two_class_test"]; rows = []
for k, lab in (("full_s42", "Full (satellite), seed 42"), ("web_s42", "Web pretraining, seed 42")):
    v = tc[k]; rows.append([lab, f"{v['auc']:.3f} [{v['auc_ci95'][0]:.3f}, {v['auc_ci95'][1]:.3f}]", f"{v['accuracy_thr0.5']:.3f} [{v['accuracy_ci95'][0]:.3f}, {v['accuracy_ci95'][1]:.3f}]",
                            f3(v["detection_rate"]), f3(v["false_alarm_rate"])])
for k, lab in (("simlbr_official_sat_lbr", "SimLBR, satellite features, latent blending"), ("simlbr_official_sat_nolbr", "SimLBR, satellite features, no blending"),
               ("simlbr_official_web_lbr", "SimLBR, web features, latent blending"), ("simlbr_official_web_nolbr", "SimLBR, web features, no blending")):
    v = tc[k]; rows.append([lab + " (3 seeds)", msd(v["auc_real_vs_manipulated"]), msd(v["accuracy_thr0.5"]), msd(v["detection_rate_manipulated_thr0.5"]), msd(v["false_alarm_rate_real_thr0.5"])])
t(["Model", "AUC real vs manipulated", "Accuracy @0.5", "Detection rate", "False-alarm rate"], rows)
p(f"Own test set, tampered and fake merged. Intervals: {a['bootstrap']['interval']}, {a['bootstrap']['resamples']} resamples by {a['bootstrap']['unit']}. "
  "SimLBR is trained on our training set with the authors' released code and settings (`simlbr_official_*_metrics.json`). Source: `analysis_extras.json`, `two_class_test`.")
rows = []
for bb in ("sat", "web"):
    for v_, vl in (("lbr", "latent blending"), ("nolbr", "no blending")):
        d = a["simlbr_official_cross"][bb][v_]
        rows.append([f"SimLBR {bb}, {vl}"] + [msd(d[k]["auc_real_vs_manipulated"]) if k in d else "–" for k in ("cross_rsfake", "cross_uw", "cross_dmaer", "cross_fmow")])
t(["Model", "RSFAKE-1M subset", "UW fake satellite", "DM-AER", "fmow-fake-small"], rows)
p("SimLBR AUC on outside datasets, three seeds (`analysis_extras.json`, `simlbr_official_cross`).")

# ------------------------------------------------------------------ robustness
h("7. Robustness (seed 42)")
rows = []
for k, lab in (("full_s42", "Full (satellite)"), ("web_s42", "Web pretraining")):
    for cnd, v in a["robustness"][k].items():
        rows.append([lab, cnd, f4(v["accuracy"]), f4(v["macro_f1"]), f4(v["tampered_recall"]), f4(v["auc_real_vs_manipulated"]), f4(v["false_alarm_rate"]), f4(v["mean_iou_tampered"])])
t(["Model", "Degradation", "Three-class accuracy", "Macro-F1", "Tampered recall", "AUC", "False-alarm rate", "Tamper IoU"], rows)
p("Degradation applied at test time before the 384 x 384 resize. The test images are 512 x 512, so scale 0.75 produces exactly the network "
  "input size and coincides with the clean condition. Source: `analysis_extras.json`, `robustness` (`overhead_*` and `robust_*_metrics.json`). "
  "The supplement draft of 2026-11-26 printed the two-class accuracy (0.9124 for full_s42) under the heading three-class accuracy; the "
  "three-class value is 0.9000, equal to `test_metrics.json`.")

# ------------------------------------------------------------------ zero-shot
h("8. Zero-shot on outside datasets (seed-42 checkpoints)")
cd = J("cross_detail.json"); rows = []
for pre, cl in (("cross_", "Full (satellite)"), ("cross_web_", "Web pretraining"), ("cross_frozen_", "Frozen backbone")):
    for k, sl in SETS:
        m = J(f"{pre}{k}_metrics.json"); c = cd["files"].get(f"{pre}{k}")
        if not m or not c: rows.append([cl, sl, "missing"] + [""] * 5); continue
        o = m["overall"]
        rows.append([cl, sl, o["n_real"], o["n_manipulated"], f"{c['auc']:.3f} [{c['auc_ci95'][0]:.3f}, {c['auc_ci95'][1]:.3f}]",
                     f3(o["detection_rate_manipulated_thr0.5"]), f3(o["false_alarm_rate_real_thr0.5"]), f3(c["tpr_at_5pct_fpr"])])
t(["Checkpoint", "Dataset", "Real", "Manip.", "AUC [95% CI]", "Det. @0.5", "FA @0.5", "Det. @5% FA (oracle threshold)"], rows)
p(f"No adaptation. Bootstrap: {json.dumps(cd['bootstrap'])}. Source: `cross_*_metrics.json`, `cross_detail.json`.")
rows = []
for pre, cl in (("cross_", "Full (satellite)"), ("cross_web_", "Web pretraining")):
    c = cd["files"].get(pre + "allreals_dmaer"); c2 = cd["files"].get(pre + "dmaer")
    if c: rows.append([cl, f"{c2['auc']:.3f}", f"{c['auc']:.3f} [{c['auc_ci95'][0]:.3f}, {c['auc_ci95'][1]:.3f}]", c["n_real"]])
t(["Checkpoint", "DM-AER AUC, size-matched reals", "DM-AER AUC, all reals [95% CI]", "Reals in the all-reals variant"], rows)
rows = []
for k, sl in SETS:
    cf, cw = cd["files"].get(f"cross_{k}", {}).get("per_group", {}), cd["files"].get(f"cross_web_{k}", {}).get("per_group", {})
    for gname, v in cf.items():
        w = cw.get(gname, {})
        rows.append([sl, gname, v["n"], f3(v["auc_vs_all_reals"]), f3(v["detection_rate_thr0.5"]), f3(w.get("auc_vs_all_reals")), f3(w.get("detection_rate_thr0.5"))])
t(["Dataset", "Group", "n", "Full: AUC vs reals", "Full: det. @0.5", "Web: AUC vs reals", "Web: det. @0.5"], rows)
p("Per generator or manipulation type; each group is scored against all real images of its dataset (`cross_detail.json`, `per_group`).")
rows = []
for pre, cl in (("cross_", "Full (satellite)"), ("cross_web_", "Web pretraining"), ("cross_frozen_", "Frozen backbone")):
    m = J(f"{pre}gdd_localization_metrics.json")
    if m: lc = m["overall"]["localization"]; rows.append([cl, lc["n_images_with_mask"], f3(lc["mean_iou"]), f3(lc["mean_dice"]), f3(lc["mean_mcc"])])
t(["Checkpoint", "Images with mask", "IoU", "Dice", "MCC"], rows)
p("Zero-shot tamper localisation on GDD local edits (`cross_*gdd_localization_metrics.json`).")

# ------------------------------------------------------------------ few-shot
h("9. Few-shot adaptation")
fs = {}
for f in ("fewshot_metrics.json", "fewshot_gdd_metrics.json"):
    d = J(f); fs.update(d["datasets"]) if d else None
sd = (J("fewshot_seeds_metrics.json") or {}).get("datasets", {})
EX = [("full_s42", "fine-tuned satellite, seed 42"), ("web_s42", "fine-tuned web, seed 42"), ("sat_orig", "original satellite"), ("web_orig", "original web")]
KS = (4, 16, 64, 256)
rows = []
for k, sl in SETS:
    d = fs[k]
    for ex, el in EX:
        e = d["extractors"][ex]; z = d["zero_shot"].get(ex)
        rows.append([sl, el, f3(z["auc"]) if isinstance(z, dict) else "–"] + [msd(e[f"k{n}"]["auc"]) if isinstance(e.get(f"k{n}"), dict) else "–" for n in KS] + [f3(e["whole_pool"]["auc"])])
t(["Dataset", "Features", "Zero-shot", "k = 4", "k = 16", "k = 64", "k = 256", "Whole pool"], rows)
p("AUC on the held-out half; k labelled images per class; mean ± standard deviation over 10 draws; whole pool is one fit. "
  "Source: `fewshot_metrics.json`, `fewshot_gdd_metrics.json` (protocol `FEWSHOT_PROTOCOL.md`).")
rows = []
for k, sl in SETS:
    for key, lab in (("full", "fine-tuned satellite"), ("web", "fine-tuned web")):
        cells = []
        for n in list(KS) + ["whole_pool"]:
            vals = []
            for s in (42, 43, 44):
                e = (fs[k]["extractors"] if s == 42 else sd.get(k, {}).get("extractors", {})).get(f"{key}_s{s}", {})
                v = e.get(f"k{n}" if n != "whole_pool" else n)
                if isinstance(v, dict): vals.append(v["auc"]["mean"] if isinstance(v["auc"], dict) else v["auc"])
            cells.append(f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}" if len(vals) == 3 else ("–" if not vals else f"{st.mean(vals):.3f} (n={len(vals)})"))
        rows.append([sl, lab] + cells)
t(["Dataset", "Features", "k = 4", "k = 16", "k = 64", "k = 256", "Whole pool"], rows)
p("The same probe on the backbones of seeds 42, 43 and 44: mean ± standard deviation over the three seeds of each seed's mean over 10 draws. "
  "Source: `fewshot_metrics.json`, `fewshot_gdd_metrics.json`, `fewshot_seeds_metrics.json`.")
rows = []
for k, sl in SETS:
    for ex in ("full_s42", "web_s42"):
        cal = fs[k].get("threshold_calibration", {}).get(ex)
        if isinstance(cal, dict): rows.append([sl, dict(EX)[ex]] + [msd(cal[f"k{n}"]["accuracy"]) if f"k{n}" in cal else "–" for n in KS])
t(["Dataset", "Model", "k = 4", "k = 16", "k = 64", "k = 256"], rows)
p("Accuracy when only the decision threshold of the fine-tuned model is re-set on the k shots, without a probe.")

# ------------------------------------------------------------------ label-free
zx = J("zeroshot_extras_metrics.json")
h("10. Label-free alternatives (seed 42)")
if zx:
    rows = []
    for ex, el in EX[:2]:
        for k, sl in SETS:
            v = zx["A_clustering"].get(ex, {}).get(k)
            if isinstance(v, dict): rows.append([el, sl, f3(v["accuracy"]), f3(v["oracle_naming_accuracy"]), f3(v["detection_rate"]), f3(v["false_alarm_rate"])])
    t(["Features", "Dataset", "Accuracy", "Accuracy, best naming (uses labels)", "Det.", "FA"], rows)
    p("A. Two-cluster k-means on each outside dataset; clusters named by the model's own mean score.")
    for key, cap in (("B_frozen_probe", "B. Linear probe fitted on our training set and applied unchanged, threshold 0.5."),
                     ("C_one_class", "C. One-class score (mean cosine distance to the 5 nearest pristine training images), threshold at the 95th percentile of held-out pristine training images.")):
        rows = []
        for ex, el in EX:
            for k, sl in [("own_test", "Own test set")] + SETS:
                v = zx[key].get(ex, {}).get(k)
                if isinstance(v, dict): rows.append([el, sl, f3(v["auc"]), f3(v["accuracy"]), f3(v["detection_rate"]), f3(v["false_alarm_rate"])])
        t(["Features", "Dataset", "AUC", "Accuracy", "Det.", "FA"], rows)
        p(cap)
    p("Source: `zeroshot_extras_metrics.json` (protocol `ZEROSHOT_EXTRAS_PROTOCOL.md`).")

# ------------------------------------------------------------------ additions of 2026-11-27/28 (review requests)
S3 = (42, 43, 44); sd3 = lambda v: f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"
if J("cross_s43_rsfake_metrics.json"):
    h("Other training seeds: zero-shot, robustness, label-free (added 2026-11-27/28)")
    p("Protocol: `ZEROSHOT_EXTRAS_PROTOCOL.md`, addendum 2026-11-27. Unchanged scripts applied to the seed-43 and seed-44 checkpoints "
      "(`run_seed_evals.sh`). Files `cross_s43_*`, `cross_web_s43_*`, `robust_full_s43_*`, `zeroshot_extras_seeds_metrics.json` and the same for seed 44.")
    def zs3(kind, k, key="auc_real_vs_manipulated"):
        pre = "cross_" if kind == "full" else "cross_web_"
        return [J(f"{pre}{'' if s == 42 else f's{s}_'}{k}_metrics.json")["overall"][key] for s in S3]
    h("Zero-shot AUC per seed (whole outside set)", 3)
    t(["Dataset", "Full s42", "Full s43", "Full s44", "Full mean ± sd", "Web s42", "Web s43", "Web s44", "Web mean ± sd"],
      [[lab] + [f3(v) for v in zs3("full", k)] + [sd3(zs3("full", k))] + [f3(v) for v in zs3("web", k)] + [sd3(zs3("web", k))]
       for k, lab in SETS + [("allreals_dmaer", "DM-AER, all real images")]])
    h("Detection and false-alarm rates at threshold 0.5, mean over seeds", 3)
    t(["Dataset", "Full: detection", "Full: false alarm", "Web: detection", "Web: false alarm"],
      [[lab, sd3(zs3("full", k, "detection_rate_manipulated_thr0.5")), sd3(zs3("full", k, "false_alarm_rate_real_thr0.5")),
        sd3(zs3("web", k, "detection_rate_manipulated_thr0.5")), sd3(zs3("web", k, "false_alarm_rate_real_thr0.5"))] for k, lab in SETS])
    AX = J("analysis_extras.json")["robustness"]
    def rb3(m, c, key):
        out = []
        for s in S3:
            if s == 42: out.append(AX[f"{m}_s42"][c][key]); continue
            o = J(f"overhead_{m}_s{s}_metrics.json" if c == "clean" else f"robust_{m}_s{s}_{c}_metrics.json")["overall"]
            out.append(o["three_class"]["accuracy"] if key == "accuracy" else o["localization"]["mean_iou"])
        return out
    h("Robustness on the test set, mean ± sd over seeds 42-44", 3)
    t(["Condition", "Full: accuracy", "Full: tamper IoU", "Web: accuracy", "Web: tamper IoU"],
      [[c, sd3(rb3("full", c, "accuracy")), sd3(rb3("full", c, "mean_iou_tampered")), sd3(rb3("web", c, "accuracy")), sd3(rb3("web", c, "mean_iou_tampered"))]
       for c in ["clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "scale0.75", "scale0.5"]])
    ZX0, ZX1 = J("zeroshot_extras_metrics.json"), J("zeroshot_extras_seeds_metrics.json")
    lf3 = lambda part, k, f, m: [(ZX0 if s == 42 else ZX1)[part][f"{m}_s{s}"][k][f] for s in S3]
    h("Label-free alternatives, mean ± sd over seeds 42-44", 3)
    for m, name in (("full", "full model"), ("web", "web model")):
        p(f"Features of the {name}:")
        t(["Dataset", "Clustering accuracy", "Source probe AUC", "One-class AUC", "One-class false-alarm rate"],
          [[lab, sd3(lf3("A_clustering", k, "accuracy", m)), sd3(lf3("B_frozen_probe", k, "auc", m)), sd3(lf3("C_one_class", k, "auc", m)),
            sd3(lf3("C_one_class", k, "false_alarm_rate", m))] for k, lab in SETS])
LG = J("fewshot_logo_metrics.json")
if LG:
    h("Few-shot with the test generator left out of the labeled images (added 2026-11-27)")
    p("Protocol: `FEWSHOT_PROTOCOL.md`, addendum 2026-11-27 (evening). File `fewshot_logo_metrics.json`. Each generator or manipulation type in "
      "turn is tested alone against the real images of the held-out half. AUC, mean over the left-out groups; fine-tuned extractors: mean over "
      "seeds 42-44. 'seen' = shots of the original protocol, 'left out' = no labeled image of the tested group. The real images of shots and test share their sources.")
    def lgm(d, exs, n, mode):
        v = [LG["datasets"][d]["extractors"][e]["_mean_over_groups_auc"][mode][n] for e in exs]
        return "–" if None in v else f"{st.mean(v):.3f}"
    EX = [("fine-tuned satellite", [f"full_s{s}" for s in S3]), ("fine-tuned web", [f"web_s{s}" for s in S3]), ("original satellite", ["sat_orig"]), ("original web", ["web_orig"])]
    rows = []
    for d, D in LG["datasets"].items():
        lab = dict(SETS)[d]
        for en, exs in EX:
            rows.append([lab, len(D["groups"]), en] + [lgm(d, exs, n, mode) for n in ("k4", "k16", "k64", "k256", "whole_pool") for mode in ("seen", "unseen")])
    t(["Dataset", "Groups", "Features"] + [f"{n}: {m}" for n in ("k=4", "k=16", "k=64", "k=256", "pool") for m in ("seen", "left out")], rows)
TF = J("trufor_overhead_metrics.json")
if TF:
    h("TruFor on the test folder (added 2026-11-28)")
    p("Protocol: `TRUFOR_PROTOCOL.md`. Authors' code and weights, no training, 3,150 test images at 512 x 512, one run. Not a like-for-like "
      "comparison: TruFor was trained on edited photographs; masks are scored at 512 x 512 (our models at 384 x 384).")
    t(["Quantity", "Value"],
      [["AUC, pristine vs manipulated", f3(TF["auc_pristine_vs_manipulated"])], ["AUC, pristine vs tampered", f3(TF["auc_pristine_vs_tampered"])],
       ["AUC, pristine vs fake", f3(TF["auc_pristine_vs_fake"])], ["Accuracy at 0.5", f3(TF["accuracy_thr0.5"])],
       ["Detection rate at 0.5 (tampered / fake)", f3(TF["detection_rate_tampered_thr0.5"]) + " / " + f3(TF["detection_rate_fake_thr0.5"])],
       ["False-alarm rate at 0.5", f3(TF["false_alarm_rate_thr0.5"])],
       ["Tamper IoU / Dice / MCC (888 tampered images)", " / ".join(f3(TF["localization"][k]) for k in ("mean_iou", "mean_dice", "mean_mcc"))]])

# ------------------------------------------------------------------ deviations and open questions
h("11. Deviations, order of analyses, and open questions")
p("Recorded in `RUNLOG.md` and the git history (`git log`). In summary:")
L.extend([
    "- Checkpoints are selected on the validation split (10% of the training folder, seed 42), never on the test set.",
    "- The few-shot, GDD and label-free experiments were added after the zero-shot results were known; each protocol file was committed before its numbers were computed (`FEWSHOT_PROTOCOL.md`, `ZEROSHOT_EXTRAS_PROTOCOL.md`).",
    "- A resampling-augmentation run to improve zero-shot transfer was planned and not run (`ZEROSHOT_AUG_PROTOCOL.md`, marked NOT RUN).",
    "- `full_s42` diverged once at epoch 19 with non-finite losses under the first mixed-precision setting; the log and folder are kept (`full_s42_DIVERGED_ep19_nan`), the fix is in `logs/fix_amp_resume.diff`, and all reported runs use bf16.",
    "- `radio_s42` failed on the workstation (folder kept as `radio_s42_FAILED_workstation`) and was run on the laptop GPU, as were `fullnorm_s42`, `webnorm_s42` and `plain_s43`; `plain_s44` was moved from the laptop to the workstation.",
    "- Inputs are scaled to [0, 1] without the backbones' own normalisation; `fullnorm_s42` and `webnorm_s42` measure the effect (section 3), one seed each.",
    "- p-values are raw (not corrected for multiple comparisons).",
    "- Single runs: frozen, ResNet-50 (both learning rates), C-RADIO, fullnorm, webnorm. Zero-shot, robustness, two-class and label-free numbers use the seed-42 checkpoints only.",
    "",
    "Open questions:",
    "",
    "- The exact MCC definition of May et al. (per image and averaged, or pooled over pixels) has not been confirmed against their code; our MCC is per image, averaged over tampered images.",
    "- The weight of the loss on small tampered regions was not studied; small regions are the main failure (section 3, area buckets).",
    "- SimLBR numbers from its paper are not reproduced here; only SimLBR trained on our data with the authors' code is reported.",
    "",
])
OUT.write_text("\n".join(L)); print("wrote", OUT, len(L), "lines")
