#!/usr/bin/env python3
"""Builds the supplementary-material draft (SUPPLEMENT.md / .html) from result files only. Re-run any time;
it overwrites only its own outputs. No number is typed by hand: every value is read from a JSON file here."""
import json, html
from pathlib import Path
from datetime import date

H = Path(__file__).resolve().parent
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
p("Detecting AI-Generated Satellite Imagery Deepfakes Using DINOv3")
p(f"DRAFT generated {date.today().isoformat()} by make_supplement.py from the result files in the repository. "
  "Entries marked 'pending' belong to runs that have not finished.")

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
p("Few-shot results with the backbones of seeds 43 and 44: " + ("see fewshot_seeds_metrics.json (table to be added)." if sd else "pending (queued after the training runs)."))

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
        o = m["overall"]; rows.append([rl, dl, f"{o['accuracy_argmax']:.4f}", f"{o['auc_real_vs_manipulated']:.4f}", f"{o['localization']['mean_iou']:.4f}"])
t(["Checkpoint", "Degradation", "3-class accuracy", "AUC real vs manip.", "Tamper mean IoU"], rows)
p("Table S8. Test-set performance under degradations applied at test time only.")

# ---- figures
h("S7. Figures")
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
.fig{text-align:center;margin:10pt 0 2pt;break-inside:avoid}.fig img{width:100%}
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
        cls = " class='sub'" if x.startswith("Detecting AI-Generated") else (" class='note'" if x.startswith("DRAFT") else "")
        md.append(x); ht.append(f"<p{cls}>{html.escape(x)}</p>")
    elif kind == "cap":
        num, title = x.split(". ", 1)
        md.append(f"**{num.upper()}** — {title}"); ht.append(f"<p class='cap'>{html.escape(num.upper())}<span>{html.escape(title.rstrip('.'))}</span></p>")
    elif kind == "img":
        md.append(f"![{x[1]}]({x[0]})\n\n{x[1]}"); ht.append(f"<div class='fig'><img src='{x[0]}'></div><p class='figcap'>{html.escape(x[1])}</p>")
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
(H / "SUPPLEMENT.md").write_text("\n".join(md)); (H / "SUPPLEMENT.html").write_text("\n".join(ht) + "</body></html>")
print("wrote SUPPLEMENT.md, SUPPLEMENT.html;", sum(k == "t" for k, _ in out), "tables")
