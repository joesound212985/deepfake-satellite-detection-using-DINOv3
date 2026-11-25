#!/usr/bin/env python3
"""
Qualitative figure for the letter: image | ground-truth mask | full_s42 mask | web_s42 mask.

Examples are chosen by a fixed rule from the saved per-image results of full_s42 (no hand picking):
  tampered images ranked by full_s42 IoU -> the images at the 90th, 50th and 10th percentile;
  the small-region failure: among tampered images with manipulated_fraction < 0.0625 that full_s42
    classified as pristine, the one with the median manipulated_fraction;
  the first fake and the first pristine image by filename.
Writes qualitative.png and qualitative.json (ids, rule, predictions).
"""
import csv, json, sys
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = Path(__file__).resolve().parent
sys.path.insert(0, str(G))
from dinov3_forensics_train_v2 import DINOv3ViTSegFormer3C, tta_predict_3class_and_seg, remap_state_dict_keys  # noqa: E402
from eval_cross_dataset import read_rgb, read_mask, CLASS_NAMES, THR_MASK  # noqa: E402

ROOT = Path.home() / "data/overhead/test_jpg/test"
RES = Path.home() / "results_dinov3/vit"
MODELS = [("full_s42 (satellite-pretrained)", "facebook/dinov3-vitl16-pretrain-sat493m", RES / "dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt"),
          ("web_s42 (web-pretrained)", "facebook/dinov3-vitl16-pretrain-lvd1689m", RES / "dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt")]


def pick():
    recs = json.load(open(G / "overhead_full_metrics.per_image.json"))
    meta = {Path(r["image_filename"]).stem: r for r in csv.DictReader(open(ROOT / "metadata.csv", newline="", encoding="utf-8"))}
    tam = sorted([r for r in recs if r.get("cls3") == 1 and "iou" in r], key=lambda r: (r["iou"], r["id"]))
    chosen = [(f"tampered, IoU {q}th percentile", tam[min(len(tam) - 1, int(round(q / 100 * (len(tam) - 1))))]["id"]) for q in (90, 50, 10)]
    small = sorted([r for r in recs if r.get("cls3") == 1 and r["pred_class"] == 0
                    and float(meta[r["id"]]["manipulated_fraction"]) < 0.0625],
                   key=lambda r: (float(meta[r["id"]]["manipulated_fraction"]), r["id"]))
    if small:
        chosen.append(("small-region miss", small[len(small) // 2]["id"]))
    for c, name in ((2, "fake"), (0, "pristine")):
        ids = sorted(r["id"] for r in recs if r.get("cls3") == c)
        chosen.append((f"{name} (first by filename)", ids[0]))
    return chosen, meta


def main():
    chosen, meta = pick()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imgs = [read_rgb(ROOT / meta[i]["image_filename"]) for _, i in chosen]
    x = torch.stack([torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1) for im in imgs]).to(device)
    preds = []
    for label, repo, ck in MODELS:
        model = DINOv3ViTSegFormer3C(dinov3_ckpt=repo, embed_dim=256, num_classes=3, use_edge_gate=True, legacy_token_slice=False).to(device)
        state = torch.load(ck, map_location=device)
        sd = remap_state_dict_keys(state["model"] if isinstance(state, dict) and "model" in state else state, model)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"checkpoint key mismatch for {ck}")
        model.eval()
        with torch.no_grad():
            ps, segs = [], []
            for k in range(0, len(imgs), 2):
                p, s = tta_predict_3class_and_seg(model, x[k:k + 2], use_tta=True, use_amp=True, channels_last=False, heavy_tta=False)
                ps.append(np.asarray(p)); segs.append(np.asarray(s))
        preds.append((label, np.concatenate(ps), np.concatenate(segs)))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(chosen)
    fig, ax = plt.subplots(n, 4, figsize=(8.0, 2.05 * n))
    info = []
    for r, (why, iid) in enumerate(chosen):
        row = meta[iid]
        gt = read_mask(ROOT / row["mask_filename"])
        gt = np.zeros((384, 384), np.uint8) if gt is None or "pristine" in row["manipulation_type"].lower() else gt
        ax[r, 0].imshow(imgs[r]); ax[r, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[r, 0].set_ylabel(why, fontsize=6.5)
        rec = {"id": iid, "why": why, "manipulation_type": row["manipulation_type"], "manipulated_fraction": float(row["manipulated_fraction"])}
        for c, (label, p, seg) in enumerate(preds):
            ax[r, 2 + c].imshow((seg[r, 0] >= THR_MASK).astype(np.uint8), cmap="gray", vmin=0, vmax=1)
            ax[r, 2 + c].set_xlabel(f"pred: {CLASS_NAMES[int(np.argmax(p[r]))]}", fontsize=6.5)
            rec[label] = {"pred_class": CLASS_NAMES[int(np.argmax(p[r]))], "probs": [float(v) for v in p[r]]}
        info.append(rec)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    for c, t in enumerate(["Image", "Ground truth"] + [m[0] for m in preds]):
        ax[0, c].set_title(t, fontsize=7.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(G / "qualitative.png", dpi=300)
    (G / "qualitative.json").write_text(json.dumps({"selection_rule": __doc__, "examples": info}, indent=2))
    print("[save] qualitative.png, qualitative.json")


if __name__ == "__main__":
    main()
