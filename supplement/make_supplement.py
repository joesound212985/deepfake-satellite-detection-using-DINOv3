#!/usr/bin/env python3
"""Builds the supplementary-material draft (SUPPLEMENT.md / .html) from result files only. Re-run any time;
it overwrites only its own outputs. No number is typed by hand: every value is read from a JSON file here."""
import json, html
from pathlib import Path
from datetime import date

OUT_DIR = Path(__file__).resolve().parent      # supplement/
H = OUT_DIR.parent                             # repository root, where the result files are
J = lambda n: json.load(open(H / n)) if (H / n).exists() else None
SETS = [("rsfake", "RSFAKE-1M subset"), ("uw", "UW fake satellite"), ("dmaer", "DM-AER (size-matched)"),
        ("gdd_indomain", "GDD in-domain"), ("gdd_crossgenerator", "GDD cross-generator"), ("gdd_crossdomain", "GDD cross-domain"),
        ("gdd_localization", "GDD local edits"), ("fmow", "fmow-fake-small (120 images)")]
CK = [("cross_", "Full model (SAT), seed 42"), ("cross_web_", "Web backbone, seed 42"), ("cross_frozen_", "Frozen backbone, seed 42")]
EX = [("full_s42", "fine-tuned SAT (full, seed 42)"), ("web_s42", "fine-tuned web (seed 42)"),
      ("sat_orig", "original DINOv3 SAT"), ("web_orig", "original DINOv3 web")]
out = []  # list of ("h"|"p"|"t"|"img", payload)
h = lambda s, l=2: out.append(("h", (l, s))); p = lambda s: out.append(("p", s))
t = lambda head, rows: out.append(("t", (head, rows))); img = lambda f, cap: out.append(("img", (f, cap)))
ms = lambda d: "–" if not isinstance(d, dict) else (f"{d['mean']:.3f} ± {d['std']:.3f}" if d.get("std") is not None else f"{d['mean']:.3f}")

h("Supplementary Material", 1)
p("Fine-Tuned DINOv3 for Detecting and Localizing AI-Generated and Tampered Satellite Images: A Benchmark Study")
p(f"Generated {date.today().isoformat()} from the result files in the repository; no number is typed by hand.")

# ---- S1 zero-shot with intervals
cd = J("cross_detail.json")
h("S1. Zero-shot evaluation on outside datasets")
p("Each checkpoint is applied without any adaptation. Score: P(manipulated) = P(tampered) + P(fake). AUC with bootstrap 95% interval "
  "(2000 resamples within class). 'Det. @0.5' and 'FA @0.5' are the detection and false-alarm rates at the fixed threshold 0.5. "
  "'Det. @5% FA' sets the threshold on the real images of the same dataset; it is an oracle operating point, not a zero-shot number.")
rows = []
for pre, cl in CK:
    for k, sl in SETS:
        m = J(f"{pre}{k}_metrics.json"); c = cd["files"].get(f"{pre}{k}") if cd else None
        if not m or not c: rows.append([cl, sl] + ["pending"] * 6); continue
        o = m["overall"]
        rows.append([cl, sl, o["n_real"], o["n_manipulated"], f"{c['auc']:.3f} [{c['auc_ci95'][0]:.3f}, {c['auc_ci95'][1]:.3f}]",
                     f"{o['detection_rate_manipulated_thr0.5']:.3f}", f"{o['false_alarm_rate_real_thr0.5']:.3f}", f"{c['tpr_at_5pct_fpr']:.3f}"])
t(["Checkpoint", "Dataset", "Real", "Manip.", "AUC [95% CI]", "Det. @0.5", "FA @0.5", "Det. @5% FA"], rows)
p("Table S1. Zero-shot detection on outside datasets.")

# ---- S2 per group
h("S2. Zero-shot results per generator / manipulation type")
p("Each manipulated group is scored against all real images of its dataset (full model, seed 42).")
rows = []
for k, sl in SETS:
    c = cd["files"].get(f"cross_{k}") if cd else None
    for g, v in (c or {}).get("per_group", {}).items():
        rows.append([sl, g, v["n"], f"{v['auc_vs_all_reals']:.3f}", f"{v['detection_rate_thr0.5']:.3f}", f"{v['tpr_at_5pct_fpr']:.3f}"])
t(["Dataset", "Group", "n", "AUC vs reals", "Det. @0.5", "Det. @5% FA"], rows)
p("Table S2. Zero-shot detection per generator or manipulation type.")

# ---- S3 tamper localisation outside
h("S3. Tamper localisation on GDD local edits")
rows = []
for pre, cl in CK:
    m = J(f"{pre}gdd_localization_metrics.json")
    if m:
        L = m["overall"]["localization"]; rows.append([cl, L["n_images_with_mask"], f"{L['mean_iou']:.3f}", f"{L['mean_dice']:.3f}", f"{L['mean_mcc']:.3f}"])
t(["Checkpoint", "Images with mask", "Mean IoU", "Mean Dice", "Mean MCC"], rows)
p("Table S3. Zero-shot tamper localisation on the GDD localisation subset.")

# ---- S4 few-shot
h("S4. Few-shot adaptation")
p("Protocol (fixed and committed before any number was computed; FEWSHOT_PROTOCOL.md in the repository): each outside dataset is split "
  "once 50/50, stratified by group, into an adaptation pool and a held-out half. A logistic-regression probe (C = 1, no tuning) is "
  "fitted on the L2-normalised final-layer CLS feature of a frozen backbone, using k labelled images per class drawn from the pool "
  "(10 random draws; mean ± standard deviation). All numbers are on the held-out half. The held-out images share source and generators "
  "with the shots, so this measures adaptation to a known new source, not generalisation to another unseen one.")
fs = {}
for f in ("fewshot_metrics.json", "fewshot_gdd_metrics.json"):
    d = J(f)
    if d: fs.update(d["datasets"])
for metric, lab, tn in (("auc", "AUC", "S4"), ("accuracy", "accuracy", "S5")):
    rows = []
    for k, sl in SETS:
        d = fs.get(k)
        if not d: rows.append([sl, "pending"] + [""] * 6); continue
        for ex, el in EX:
            e = d["extractors"][ex]; z = d["zero_shot"].get(ex)
            rows.append([sl, el, f"{z[metric]:.3f}" if isinstance(z, dict) else "–"] + [ms(e[f"k{n}"][metric]) if isinstance(e.get(f"k{n}"), dict) else "–" for n in (4, 16, 64, 256)]
                        + [f"{e['whole_pool'][metric]:.3f}"])
    t(["Dataset", "Features", "Zero-shot", "k = 4", "k = 16", "k = 64", "k = 256", "Whole pool"], rows)
    p(f"Table {tn}. Few-shot {lab} on the held-out half (k labelled images per class; '–' = not applicable or pool too small).")
rows = []
for k, sl in SETS:
    d = fs.get(k)
    for ex in ("full_s42", "web_s42"):
        c = (d or {}).get("threshold_calibration", {}).get(ex)
        if isinstance(c, dict):
            rows.append([sl, dict(EX)[ex]] + [ms(c[f"k{n}"]["accuracy"]) if f"k{n}" in c else "–" for n in (4, 16, 64, 256)])
t(["Dataset", "Model", "k = 4", "k = 16", "k = 64", "k = 256"], rows)
p("Table S6. Accuracy when only the decision threshold of the fine-tuned model is re-set on the k shots (no probe).")
sd = J("fewshot_seeds_metrics.json")
p("The same probe on the backbones of training seeds 43 and 44 is summarised in Table S12.")

# ---- S5 SimLBR
h("S5. SimLBR baseline (authors' released code, trained on our training set)")
rows = []
for bb, bl in (("sat", "SAT"), ("web", "web")):
    s = J(f"simlbr_official_{bb}_metrics.json")
    if not s: rows.append([bl, "pending"] + [""] * 3); continue
    for v, vl in (("lbr", "with latent blending"), ("nolbr", "without")):
        m = s["variants"][v]["mean_std_over_seeds"]
        for k, kl in (("overhead_test", "Own test set"), ("cross_rsfake", "RSFAKE-1M subset"), ("cross_uw", "UW fake satellite"), ("cross_dmaer", "DM-AER"), ("cross_fmow", "fmow-fake-small")):
            if k in m: rows.append([bl, vl, kl, ms(m[k]["auc_real_vs_manipulated"]), ms(m[k]["accuracy_thr0.5"])])
t(["Backbone", "Variant", "Evaluated on", "AUC", "Accuracy @0.5"], rows)
p("Table S7. SimLBR trained on our data with the authors' code and settings, three seeds (mean ± standard deviation).")

# ---- S6 robustness
h("S6. Robustness to JPEG compression and down-scaling (own test set)")
rows = []
for r, rl in (("full", "Full model (SAT), seed 42"), ("web", "Web backbone, seed 42")):
    for d, dl in ((None, "none"), ("jpeg90", "JPEG 90"), ("jpeg70", "JPEG 70"), ("jpeg50", "JPEG 50"), ("jpeg30", "JPEG 30"), ("scale0.75", "scale 0.75"), ("scale0.5", "scale 0.5")):
        m = J(f"overhead_{r}_metrics.json" if d is None else f"robust_{r}_{d}_metrics.json")
        if not m: rows.append([rl, dl, "pending", "", ""]); continue
        o = m["overall"]; rows.append([rl, dl, f"{o['three_class']['accuracy']:.4f}", f"{o['auc_real_vs_manipulated']:.4f}", f"{o['localization']['mean_iou']:.4f}"])
t(["Checkpoint", "Degradation", "3-class accuracy", "AUC real vs manip.", "Tamper mean IoU"], rows)
p("Table S8. Test-set performance under degradations applied at test time only.")

# ---- S7 label-free transfer experiments
zx = J("zeroshot_extras_metrics.json")
h("S7. Label-free alternatives to few-shot adaptation")
p("Three ways to use an outside dataset without any of its labels (protocol fixed and committed before any number was computed; "
  "ZEROSHOT_EXTRAS_PROTOCOL.md in the repository). These experiments were chosen after the zero-shot results were known. "
  "All use the L2-normalised final-layer CLS feature of a frozen backbone.")
XS = [("own_test", "Own test set")] + SETS
if not zx:
    p("Pending.")
else:
    rows = []
    for ex, el in EX[:2]:
        for k, sl in SETS:
            v = zx["A_clustering"].get(ex, {}).get(k)
            if isinstance(v, dict):
                rows.append([el, sl, f"{v['accuracy']:.3f}", f"{v['oracle_naming_accuracy']:.3f}", f"{v['detection_rate']:.3f}", f"{v['false_alarm_rate']:.3f}"])
    t(["Features", "Dataset", "Accuracy", "Accuracy, best naming", "Det.", "FA"], rows)
    p("Table S9. Unsupervised two-cluster split (k-means) of each outside dataset. The clusters are named real / manipulated by the "
      "model's own mean score; 'best naming' uses the labels to name them and is an upper bound, not a label-free number.")
    for key, tn, cap in (("B_frozen_probe", "S10", "Linear probe fitted on the features of our own training set (10,000 images) and applied unchanged to each dataset, threshold 0.5."),
                         ("C_one_class", "S11", "One-class score: mean cosine distance to the 5 nearest pristine training images. The threshold is the 95th percentile "
                                                "of held-out pristine training images, so 'FA' above 0.05 on an outside dataset shows that its real images also look unusual.")):
        rows = []
        for ex, el in EX:
            for k, sl in XS:
                v = zx[key].get(ex, {}).get(k)
                if isinstance(v, dict):
                    rows.append([el, sl, f"{v['auc']:.3f}", f"{v['accuracy']:.3f}", f"{v['detection_rate']:.3f}", f"{v['false_alarm_rate']:.3f}"])
        t(["Features", "Dataset", "AUC", "Accuracy", "Det.", "FA"], rows)
        p(f"Table {tn}. {cap}")

# ---- additional tables: few-shot over training seeds, per-seed results, significance, input normalisation
import csv, statistics as st
h("S8. Additional tables")
rows = []
for k, sl in SETS:
    for key, lab in (("full", "fine-tuned SAT"), ("web", "fine-tuned web")):
        cells = []
        for n in ("k4", "k16", "k64", "k256", "whole_pool"):
            vals = []
            for sdd in (42, 43, 44):
                e = (fs[k]["extractors"] if sdd == 42 else (sd or {}).get("datasets", {}).get(k, {}).get("extractors", {})).get(f"{key}_s{sdd}", {})
                v = e.get(n)
                if isinstance(v, dict): vals.append(v["auc"]["mean"] if isinstance(v["auc"], dict) else v["auc"])
            cells.append(f"{st.mean(vals):.3f} ± {st.stdev(vals):.3f}" if len(vals) == 3 else "–")
        rows.append([sl, lab] + cells)
t(["Dataset", "Features", "k = 4", "k = 16", "k = 64", "k = 256", "Whole pool"], rows)
p("Table S12. Few-shot AUC with the backbones of training seeds 42, 43 and 44: mean ± standard deviation over the three seeds of each seed's mean over 10 draws.")
RES = Path.home() / "results_dinov3" / "vit"
rows = []
for key, bb, lab in (("full", "dinov3-vitl16-pretrain-sat493m", "Full (SAT)"), ("plain", "dinov3-vitl16-pretrain-sat493m", "Plain decoder (SAT)"), ("web", "dinov3-vitl16-pretrain-lvd1689m", "Web backbone")):
    for sdd in (42, 43, 44):
        f = RES / bb / f"{key}_s{sdd}" / "test_metrics.json"
        if not f.exists(): rows.append([lab, sdd, "pending", "", "", "", ""]); continue
        m = json.load(open(f)); L = m["localization_tampered"]
        rows.append([lab, sdd, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['per_class']['tampered']['recall']:.4f}", f"{L['mean_iou']:.4f}", m.get("best_epoch")])
for key, sub, lab in (("frozen", "vit/dinov3-vitl16-pretrain-sat493m", "Frozen backbone (SAT)"), ("resnet", "resnet/resnet50", "ResNet-50, LR 2e-6"),
                      ("resnetlr", "resnet/resnet50", "ResNet-50, LR 1e-4"), ("radio", "radio/C-RADIOv4-SO400M", "C-RADIO")):
    for sdd in (42, 43, 44):          # rows added in review: only the seeds that have finished are listed
        f = RES.parent / sub / f"{key}_s{sdd}" / "test_metrics.json"
        if not f.exists(): continue
        m = json.load(open(f)); L = m["localization_tampered"]
        rows.append([lab, sdd, f"{m['accuracy']:.4f}", f"{m['macro_f1']:.4f}", f"{m['per_class']['tampered']['recall']:.4f}", f"{L['mean_iou']:.4f}", m.get("best_epoch")])
t(["Row", "Seed", "Accuracy", "Macro-F1", "Tampered recall", "Tamper IoU", "Best epoch (validation)"], rows)
p("Table S13. Test-set results of each training seed. For the frozen-backbone, ResNet-50 and C-RADIO rows the seeds finished so far are listed.")
an = J("analysis_extras.json"); rows = []
for pair, d in (an or {}).get("significance", {}).get("pairs", {}).items():
    a_, b_ = pair.split("_vs_")
    for sdd in ("seed42", "seed43", "seed44"):
        v = d.get(sdd)
        if v: rows.append([f"{a_} vs {b_}", sdd[4:], f"{v[f'acc_{a_}']:.4f}", f"{v[f'acc_{b_}']:.4f}", v[f"only_{a_}_correct"], v[f"only_{b_}_correct"], f"{v['p_value']:.2g}"])
t(["Pair (A vs B)", "Seed", "Accuracy A", "Accuracy B", "Only A correct", "Only B correct", "p"], rows)
p("Table S14. Exact McNemar tests on three-class correctness of the 3150 test images, per seed; p-values are not corrected for multiple comparisons. "
  "Accuracies come from the re-evaluation script and can differ from Table S13 in the fourth decimal.")
tb = {r["variant"]: r for r in csv.DictReader(open(H / "table.csv"))} if (H / "table.csv").exists() else {}
rows = []
for key, lab in (("full", "Full (SAT), inputs in [0, 1], 3 seeds"), ("fullnorm", "Full (SAT), backbone normalisation, seed 42"),
                 ("web", "Web backbone, inputs in [0, 1], 3 seeds"), ("webnorm", "Web backbone, backbone normalisation, seed 42")):
    r = tb.get(key)
    if r: rows.append([lab] + [(f"{float(r[c + '_mean']):.3f} ± {float(r[c + '_std']):.3f}" if int(r["n_runs"]) > 1 else f"{float(r[c + '_mean']):.3f}") for c in ("accuracy", "macro_f1", "tampered_recall", "mean_iou")])
t(["Row", "Accuracy", "Macro-F1", "Tampered recall", "Tamper IoU"], rows)
p("Table S15. Side experiment on input normalisation: the backbone's own mean and standard-deviation normalisation against inputs scaled to [0, 1] (the setting of all other runs).")

# ---- per-seed zero-shot, held-out generator, TruFor (added 2026-11-28)
rows = []
for pre, lab in (("cross_", "Full (SAT)"), ("cross_web_", "Web backbone")):
    for k, sl in SETS:
        v = [J(f"{pre}{'' if sdd == 42 else f's{sdd}_'}{k}_metrics.json")["overall"]["auc_real_vs_manipulated"] for sdd in (42, 43, 44)]
        rows.append([lab, sl] + [f"{x:.3f}" for x in v] + [f"{st.mean(v):.3f} ± {st.stdev(v):.3f}"])
t(["Model", "Dataset", "Seed 42", "Seed 43", "Seed 44", "Mean ± std"], rows)
p("Table S16. Zero-shot AUC (real against manipulated, no adaptation) of each training seed on the outside datasets.")
lo = J("fewshot_logo_metrics.json")
if lo:
    LS = [(k, sl) for k, sl in SETS if k in lo["datasets"]]
    def lgm(k, key, n, mode):
        v = [lo["datasets"][k]["extractors"][f"{key}_s{sdd}"]["_mean_over_groups_auc"][mode][n] for sdd in (42, 43, 44)]
        return "–" if None in v else f"{st.mean(v):.3f}"
    rows = [[sl, fl] + [lgm(k, key, n, mode) for n in ("k16", "k64", "whole_pool") for mode in ("seen", "unseen")]
            for k, sl in LS for key, fl in (("full", "fine-tuned SAT"), ("web", "fine-tuned web"))]
    t(["Dataset", "Features", "k = 16: seen", "k = 16: left out", "k = 64: seen", "k = 64: left out", "Pool: seen", "Pool: left out"], rows)
    p("Table S17. Few-shot AUC with the test generator or manipulation type left out of the labelled images, against the same test with it "
      "included ('seen'). Mean over the groups of each dataset and over training seeds 42, 43 and 44. Protocol: addendum in FEWSHOT_PROTOCOL.md.")
    rows = []
    for k, sl in LS:
        ex = lo["datasets"][k]["extractors"]
        for g in [g for g in ex["full_s42"] if not g.startswith("_")]:
            cell = lambda mode: st.mean(ex[f"full_s{sdd}"][g][mode]["k64"]["auc"]["mean"] for sdd in (42, 43, 44))
            try: rows.append([sl, g.split("/", 1)[-1], f"{cell('seen'):.3f}", f"{cell('unseen'):.3f}"])
            except (KeyError, TypeError): rows.append([sl, g.split("/", 1)[-1], "–", "–"])
    t(["Dataset", "Generator or manipulation type", "k = 64: seen", "k = 64: left out"], rows)
    p("Table S18. The same comparison for each generator or manipulation type (fine-tuned SAT backbone, k = 64, mean over training seeds 42, 43 and 44).")
tf = J("trufor_overhead_metrics.json")
if tf:
    L = tf["localization"]
    t(["AUC pristine vs. manipulated", "AUC pristine vs. tampered", "AUC pristine vs. fake", "Accuracy @0.5", "Det. tampered", "Det. fake", "FA", "Tamper IoU", "Dice", "MCC"],
      [[f"{tf[c]:.3f}" for c in ("auc_pristine_vs_manipulated", "auc_pristine_vs_tampered", "auc_pristine_vs_fake", "accuracy_thr0.5",
                                 "detection_rate_tampered_thr0.5", "detection_rate_fake_thr0.5", "false_alarm_rate_thr0.5")]
       + [f"{L[c]:.3f}" for c in ("mean_iou", "mean_dice", "mean_mcc")]])
    p(f"Table S19. TruFor with its authors' code and weights, without training, on the {tf['n_images']} test images at their native 512 × 512 resolution "
      f"(tamper map thresholded at {tf['localization']['mask_threshold']}; localisation over the {L['n_images_with_mask']} tampered images). Protocol: TRUFOR_PROTOCOL.md.")

# ---- real-anchored matched filter (added 2026-11-28; REAL_ANCHOR_PROTOCOL.md)
ra = J("real_anchor_metrics.json")
if ra:
    RAX = ra["extractors"]
    def rv(ex, k, key, name, f="auc"):
        d = RAX[ex]["sets"][k].get(key)
        if not isinstance(d, dict): return None
        v = d[name][f]; return v["mean"] if isinstance(v, dict) else v
    def r3(m, k, key, name, f="auc"):          # original backbones: one extractor; fine-tuned: mean over training seeds
        if m.endswith("_orig"): return rv(m, k, key, name, f)
        v = [rv(f"{m}_s{sdd}", k, key, name, f) for sdd in (42, 43, 44)]; return None if None in v else st.mean(v)
    def na(m, k, f="auc"):
        return RAX[m]["sets"][k]["no_anchor"][f] if m.endswith("_orig") else st.mean(RAX[f"{m}_s{sdd}"]["sets"][k]["no_anchor"][f] for sdd in (42, 43, 44))
    c = lambda v: "–" if v is None else f"{v:.3f}"
    FEATS = (("sat_orig", "original SAT"), ("full", "fine-tuned SAT"), ("web_orig", "original web"), ("web", "fine-tuned web"))
    h("S9. Matched filter anchored on authentic images of the new source")
    p("Protocol (fixed and committed before any number was computed; REAL_ANCHOR_PROTOCOL.md): the manipulation signature is the mean feature of "
      "the manipulated training images minus that of the pristine training images. An image of an outside dataset is scored by the matched "
      "filter with that signature after centring and whitening with the mean and the Ledoit-Wolf covariance of k authentic images drawn from "
      "the adaptation pool (10 draws; the whole pool once). No manipulated image of the outside dataset is used. All numbers are on the "
      "held-out half. 'No anchor' uses the mean and covariance of the pristine training images instead, so no image of the new source. "
      "The experiment was designed after the other results were known. The rule fixed in advance was that the method counts as a success "
      "only if, at k = 64, its AUC exceeds the zero-shot AUC of the fine-tuned model on at least six of the eight sets for the original "
      "SAT backbone or for the fine-tuned SAT backbone; it does so on fewer (Table S21). On our own test folder the signature with "
      "training statistics reaches an AUC of " + ", ".join(f"{na(m, 'own_test'):.3f} ({lab})" for m, lab in FEATS) + ".")
    rows = [[sl, lab, c(na(m, k))] + [c(r3(m, k, kk, "anchor_matched_filter")) for kk in ("k4", "k16", "k64", "k256", "whole_pool")]
            for k, sl in SETS for m, lab in FEATS]
    t(["Dataset", "Features", "No anchor", "k = 4", "k = 16", "k = 64", "k = 256", "Whole pool"], rows)
    p("Table S20. Matched filter anchored on k authentic images of the new source: AUC on the held-out half. Fine-tuned features: mean over "
      "training seeds 42, 43 and 44 of each seed's mean over 10 draws. '–': the pool has fewer than k authentic images.")
    rows = []
    for k, sl in SETS:
        zsw = st.mean(J(f"cross_{'' if sdd == 42 else f's{sdd}_'}{k}_metrics.json")["overall"]["auc_real_vs_manipulated"] for sdd in (42, 43, 44))
        for m, lab in FEATS[:2]:
            rows.append([sl, lab, f"{zsw:.3f}", c(na(m, k))] + [c(r3(m, k, "k64", n)) for n in ("anchor_mean", "anchor_matched_filter", "anchor_rx")]
                        + [c(r3(m, k, "k64", "anchor_matched_filter", "detection_rate")), c(r3(m, k, "k64", "anchor_matched_filter", "false_alarm_rate"))])
    t(["Dataset", "Features", "Zero-shot AUC", "No anchor AUC", "Anchored mean AUC", "Matched filter AUC", "RX AUC", "Det.", "FA"], rows)
    p("Table S21. The comparison fixed in advance, at k = 64. Zero-shot: the fine-tuned full model without adaptation, whole outside set, mean over "
      "three seeds (Table S16). Anchored mean: target mean with the training covariance (same ranking as 'no anchor'). Matched filter: target mean "
      "and covariance. RX: squared Mahalanobis distance to the k authentic images, without the signature. Det. and FA: detection and false-alarm "
      "rate of the matched filter at the threshold 1.645, which would give 5% false alarms if the score were standard normal on authentic images.")

# ---- figures
h("S10. Figures")
if (H / "tsne_features.png").exists():
    img("tsne_features.png", "Fig. S1. t-SNE of frozen CLS features (up to 500 real and 500 manipulated images per dataset; perplexity 30, one run, "
        "settings fixed in advance). Top: original DINOv3 SAT. Bottom: fine-tuned full model. Illustration only.")
if (H / "qualitative.png").exists():
    img("qualitative.png", "Fig. S2. Qualitative examples chosen by a fixed rule (see qualitative.json): image, ground-truth mask, full-model mask, web-backbone mask.")

# ---- write (IEEE-like look: Times, centred small-caps section heads with roman numerals, table captions above
#      the table as "TABLE S1" + small-caps title, horizontal rules only, "Fig. S1." captions below figures)
ROM = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
CSS = """@page{size:letter;margin:0.75in 0.7in}
body{font-family:'Times New Roman','Nimbus Roman',Times,serif;font-size:10pt;line-height:1.15;text-align:justify}
h1{font-size:22pt;font-weight:normal;text-align:center;margin:0 0 4pt}
.sub{text-align:center;font-size:11pt;margin:0 0 2pt}.note{text-align:center;font-size:8pt;font-style:italic;margin:0 0 14pt}
h2{font-size:10pt;font-weight:normal;font-variant:small-caps;text-align:center;margin:14pt 0 5pt;break-after:avoid}
p{margin:0 0 5pt;text-indent:1em}
.cap{text-align:center;font-size:8pt;margin:10pt 0 3pt;text-indent:0;break-after:avoid}.cap span{font-variant:small-caps;display:block}
table{border-collapse:collapse;font-size:8pt;margin:0 auto 8pt;border-top:1.2px solid #000;border-bottom:1.2px solid #000}
th{font-weight:normal;border-bottom:0.7px solid #000;padding:2px 7px;text-align:center}
td{padding:1px 6px;text-align:center;white-space:nowrap}td:first-child,td:nth-child(2){text-align:left}
tr.sep td{border-top:0.5px solid #000}tr{break-inside:avoid}
.fig{text-align:center;margin:10pt 0 2pt;break-inside:avoid;page-break-inside:avoid}.fig img{max-width:100%;max-height:8.3in;width:auto;height:auto}
.figcap{font-size:8pt;text-align:justify;text-indent:0;margin:0 0 8pt}"""
import re
items = []
for kind, x in out:      # move "Table Sn. ..." paragraphs above their table
    if kind == "p" and x.startswith("Table S") and items and items[-1][0] == "t":
        tb = items.pop(); items.append(("cap", x)); items.append(tb)
    else:
        items.append((kind, x))
md, ht, nsec = [], [f"<html><head><meta charset='utf-8'><title>Supplementary Material</title><style>{CSS}</style></head><body>"], 0
for kind, x in items:
    if kind == "h" and x[0] == 1:
        md.append("# " + x[1]); ht.append(f"<h1>{html.escape(x[1])}</h1>")
    elif kind == "h":
        title = re.sub(r"^S\d+\.\s*", "", x[1]); nsec += 1
        md.append(f"## {ROM[nsec-1]}. {title}"); ht.append(f"<h2>{ROM[nsec-1]}. {html.escape(title)}</h2>")
    elif kind == "p":
        cls = " class='sub'" if x.startswith("Detection and Localization of AI-Generated") else (" class='note'" if x.startswith("Generated ") else "")
        md.append(x); ht.append(f"<p{cls}>{html.escape(x)}</p>")
    elif kind == "cap":
        num, title = x.split(". ", 1)
        md.append(f"**{num.upper()}** — {title}"); ht.append(f"<p class='cap'>{html.escape(num.upper())}<span>{html.escape(title.rstrip('.'))}</span></p>")
    elif kind == "img":
        md.append(f"![{x[1]}](../{x[0]})\n\n{x[1]}"); ht.append(f"<div class='fig'><img src='../{x[0]}'><p class='figcap'>{html.escape(x[1])}</p></div>")
    else:
        hd, rows = x
        md.append("| " + " | ".join(hd) + " |\n|" + "---|" * len(hd) + "\n" + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows))
        body, prev = [], None
        for r in rows:      # a thin rule whenever the first column changes; repeated first-column labels are left blank
            first = str(r[0]); sep = prev is not None and first != prev
            cells = [("" if (first == prev and len(hd) > 4) else first)] + [str(c) for c in r[1:]]; prev = first
            body.append(f"<tr{' class=sep' if sep else ''}>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        ht.append("<table><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in hd) + "</tr>" + "".join(body) + "</table>")
    md.append("")
(OUT_DIR / "SUPPLEMENT.md").write_text("\n".join(md)); (OUT_DIR / "SUPPLEMENT.html").write_text("\n".join(ht) + "</body></html>")
print("wrote supplement/SUPPLEMENT.md, supplement/SUPPLEMENT.html;", sum(k == "t" for k, _ in out), "tables")

# ---- Word version (needs python-docx; kept outside the experiment environment:
#      PYTHONPATH=~/.local/share/grsl-docx-tools python supplement/make_supplement.py)
try:
    import docx
except ImportError:
    docx = None
    print("python-docx not importable: SUPPLEMENT.docx not written")
if docx:
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    from PIL import Image
    D = docx.Document(); sec = D.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.left_margin = sec.right_margin = Inches(0.7); sec.top_margin = sec.bottom_margin = Inches(0.75)
    st = D.styles["Normal"]; st.font.name = "Times New Roman"; st.font.size = Pt(10)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    st.paragraph_format.space_after = Pt(4); st.paragraph_format.space_before = Pt(0)

    def para(text, size=10, align=AL.JUSTIFY, caps=False, italic=False, before=0, after=4, keep=False, indent=False):
        q = D.add_paragraph(); r = q.add_run(text); r.font.size = Pt(size); r.font.small_caps = caps; r.italic = italic
        q.alignment = align; f = q.paragraph_format; f.space_before = Pt(before); f.space_after = Pt(after); f.keep_with_next = keep
        if indent: f.first_line_indent = Pt(10)
        return q

    def border(cell, **edges):
        tcPr = cell._tc.get_or_add_tcPr(); b = tcPr.find(qn("w:tcBorders"))
        if b is None: b = OxmlElement("w:tcBorders"); tcPr.append(b)
        for e, sz in edges.items():
            el = OxmlElement(f"w:{e}"); el.set(qn("w:val"), "single"); el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "000000"); b.append(el)

    nsec = 0
    for kind, x in items:
        if kind == "h" and x[0] == 1: para(x[1], 22, AL.CENTER, after=2)
        elif kind == "h":
            nsec += 1; para(f"{ROM[nsec-1]}. " + re.sub(r"^S\d+\.\s*", "", x[1]), 10, AL.CENTER, caps=True, before=12, after=5, keep=True)
        elif kind == "p":
            if x.startswith("Detection and Localization of AI-Generated"): para(x, 11, AL.CENTER, after=2)
            elif x.startswith("Generated "): para(x, 8, AL.CENTER, italic=True, after=12)
            else: para(x, indent=True)
        elif kind == "cap":
            num, title = x.split(". ", 1)
            para(num.upper(), 8, AL.CENTER, before=8, after=0, keep=True); para(title.rstrip("."), 8, AL.CENTER, caps=True, after=2, keep=True)
        elif kind == "img":
            w, hgt = Image.open(H / x[0]).size; W = min(7.0, 8.0 * w / hgt)
            D.add_picture(str(H / x[0]), width=Inches(W)); D.paragraphs[-1].alignment = AL.CENTER; D.paragraphs[-1].paragraph_format.keep_with_next = True
            para(x[1], 8, after=8)
        else:
            hd, rows = x; T = D.add_table(rows=1 + len(rows), cols=len(hd)); T.alignment = WD_TABLE_ALIGNMENT.CENTER; T.autofit = True
            prev = None
            for i, r in enumerate([hd] + rows):
                first = str(r[0]); grp = i > 1 and first != prev
                vals = [("" if (i > 1 and first == prev and len(hd) > 4) else first)] + [str(c) for c in r[1:]]
                if i > 0: prev = first
                for j, v in enumerate(vals):
                    c = T.cell(i, j); q = c.paragraphs[0]; q.paragraph_format.space_after = Pt(0)
                    q.alignment = AL.LEFT if (i > 0 and j < 2) else AL.CENTER
                    rr = q.add_run(v); rr.font.size = Pt(8)
                    if i == 0: border(c, top=10, bottom=6)
                    if grp: border(c, top=4)
                    if i == len(rows): border(c, bottom=10)
            D.add_paragraph().paragraph_format.space_after = Pt(2)
    D.save(OUT_DIR / "SUPPLEMENT.docx"); print("wrote supplement/SUPPLEMENT.docx")
