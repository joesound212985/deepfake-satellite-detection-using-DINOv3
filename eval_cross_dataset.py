#!/usr/bin/env python3
"""
Zero-shot cross-dataset evaluation of a trained DINOv3 forensics checkpoint.

Imports the model class and the TTA function from dinov3_forensics_train_v2.py
(no copy-paste). Inputs are preprocessed exactly as the training script does:
BGR->RGB, cv2.INTER_AREA resize to 384x384, float32 / 255, no mean/std.
Same 4-view TTA (none, hflip, vflip, hvflip) and 0.5 mask threshold.

Two-class scoring: P(manipulated) = P(tampered) + P(fake).

Datasets:
  fmow    geodf/fmow-fake-small   (train/ + val/, real/ and fake/{simple-splice,object-splice,inpaint})
  rsfake  TZHSW/RSFAKE            (official test split, fixed-seed per-generator subset)
  uw      UW Fake Satellite Imagery (figshare 12197655; whole-image labels)
  overhead  the main test split (metadata.csv): two-class + three-class scores, tampered-only IoU;
          with --degrade jpeg:Q | scale:S it gives the robustness numbers
  dmaer   DM-AER-DeepFake-V1 official sample (validation/FAKE StyleGAN2, validation/REAL Million-AID;
          whole-image labels, same folder-name rule as uw)

Usage:
  python eval_cross_dataset.py --dataset fmow   --root ~/data/fmow-fake-small --ckpt <best_checkpoint.pt>
  python eval_cross_dataset.py --dataset rsfake --root ~/data/rsfake --per-gen 500 --ckpt <best_checkpoint.pt>
"""
import os, sys, json, math, argparse, random, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dinov3_forensics_train_v2 import DINOv3ViTSegFormer3C, tta_predict_3class_and_seg, remap_state_dict_keys  # noqa: E402

IMG_SIZE = 384
THR_MASK = 0.5


# ----------------------------------------------------------------------------
# image / mask readers (handle TIFF, 16-bit, >3 bands)
# ----------------------------------------------------------------------------
def degrade_bgr(img: np.ndarray, spec: str) -> np.ndarray:
    """Robustness perturbation applied to the decoded uint8 image at native resolution, before the
    usual 384 resize.  spec: 'jpeg:Q' (re-encode with OpenCV at quality Q) or 'scale:S' (resize both
    sides by factor S with INTER_AREA; the standard 384 resize then follows)."""
    kind, val = spec.split(":")
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(val)])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if kind == "scale":
        h, w = img.shape[:2]
        return cv2.resize(img, (max(8, round(w * float(val))), max(8, round(h * float(val)))), interpolation=cv2.INTER_AREA)
    raise ValueError(f"unknown --degrade spec: {spec}")


def read_rgb(path: Path, crop: int = 0, degrade: str = "") -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # to BGR for the common path
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = img.astype(np.float32)
        mx = float(img.max())
        if mx > 255.0:            # 16-bit style data
            img = img / 65535.0 * 255.0 if mx > 4095 else img / mx * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    if degrade:
        img = degrade_bgr(img, degrade)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop:                      # centre crop to crop x crop at native resolution (DM-AER size matching)
        h, w = img.shape[:2]
        t, l = (h - crop) // 2, (w - crop) // 2
        img = np.ascontiguousarray(img[t:t + crop, l:l + crop])
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img


def read_mask(path):
    if path is None or not Path(path).exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        from PIL import Image
        m = np.array(Image.open(path))
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m.astype(np.float32) > (127 if m.max() > 1 else 0.5)).astype(np.uint8) * 255
    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


class ListDataset(Dataset):
    """items: list of dicts {img, mask (path or None), label (0 real / 1 manipulated), group, id}"""
    def __init__(self, items, degrade: str = ""):
        self.items = items
        self.degrade = degrade

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = read_rgb(Path(it["img"]), it.get("crop", 0), self.degrade)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        m = read_mask(it.get("mask"))
        has_mask = m is not None and it["label"] == 1
        if m is None:
            m = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
        return {"pixel_values": x, "mask": torch.from_numpy(m), "has_mask": has_mask,
                "label": it["label"], "group": it["group"], "id": it["id"], "idx": i, "cls3": it.get("cls3", -1)}


def collate(b):
    return {
        "pixel_values": torch.stack([d["pixel_values"] for d in b]),
        "mask": torch.stack([d["mask"] for d in b]),
        "has_mask": [d["has_mask"] for d in b],
        "label": [d["label"] for d in b],
        "group": [d["group"] for d in b],
        "id": [d["id"] for d in b],
        "cls3": [d["cls3"] for d in b],
    }


# ----------------------------------------------------------------------------
# dataset builders
# ----------------------------------------------------------------------------
def build_fmow(root: Path):
    items = []
    for split in ("train", "val"):
        rd = root / split / "real"
        for p in sorted(rd.glob("*_img.tif")) if rd.exists() else []:
            items.append({"img": p, "mask": None, "label": 0, "group": f"real", "id": f"{split}/{p.name}"})
        fd = root / split / "fake"
        if fd.exists():
            for typ in sorted(d for d in fd.iterdir() if d.is_dir()):
                for p in sorted(typ.glob("*_img.tif")):
                    seg = p.with_name(p.name.replace("_img.tif", "_seg.tif"))
                    items.append({"img": p, "mask": seg if seg.exists() else None, "label": 1,
                                  "group": typ.name, "id": f"{split}/{typ.name}/{p.name}"})
    return items


def build_rsfake_from_csv(root: Path, csv_path: Path):
    """Subset written by prepare_rsfake_subset.py (image_path,label,gen,exists)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    items = []
    for r in df.itertuples():
        p = root / r.image_path
        if not p.exists():
            continue
        mask = None
        if r.label == 1:
            for cand in (p.with_name(p.stem + "_mask" + p.suffix), p.with_name(p.stem + "_mask.png")):
                if cand.exists():
                    mask = cand
                    break
        items.append({"img": p, "mask": mask, "label": int(r.label), "group": r.gen, "id": r.image_path})
    print(f"[data] subset csv {csv_path}: {len(df)} rows, {len(items)} files present")
    return items


def build_rsfake(root: Path, per_gen: int, seed: int, real_dir: Path = None):
    import pandas as pd
    df = pd.read_csv(root / "SPLIT" / "RSFAKE_test_new.csv")
    df["gen"] = df.image_path.str.split("/").str[1]
    rng = random.Random(seed)
    items = []
    gens = sorted(df[df.label == 1].gen.unique())
    for g in gens:
        rows = df[(df.label == 1) & (df.gen == g)].image_path.tolist()
        rows.sort()
        rng.shuffle(rows)
        chosen = rows[:per_gen]
        for rel in chosen:
            p = root / rel
            if not p.exists():
                # zips may extract with or without the FAKE/ prefix; try alternatives
                alt = root / "FAKE" / Path(rel).relative_to("FAKE")
                p = alt if alt.exists() else p
            # masks for rsinpaint if provided next to the image
            mask = None
            for cand in (p.with_name(p.stem + "_mask" + p.suffix), p.with_name(p.stem + "_mask.png"),
                         p.parent.parent / (p.parent.name + "_mask") / p.name):
                if cand.exists():
                    mask = cand
                    break
            items.append({"img": p, "mask": mask, "label": 1, "group": g, "id": rel})
    n_fake = len(items)
    if real_dir is not None and real_dir.exists():
        reals = df[df.label == 0].image_path.tolist()
        reals.sort()
        rng.shuffle(reals)
        got = 0
        for rel in reals:
            p = root / rel
            if not p.exists():
                p = real_dir / Path(rel).name
            if p.exists():
                items.append({"img": p, "mask": None, "label": 0, "group": "real_fmow", "id": rel})
                got += 1
                if got >= n_fake:
                    break
    missing = [str(it["img"]) for it in items if not Path(it["img"]).exists()]
    if missing:
        print(f"[warn] {len(missing)} files missing, e.g. {missing[:3]}")
        items = [it for it in items if Path(it["img"]).exists()]
    return items


def build_uw(root: Path):
    """UW Fake Satellite Imagery: any image under a folder whose name contains 'fake' is label 1,
    'real'/'true'/'original' is label 0. Grouped by top-level folder (city)."""
    items = []
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        parts = [s.lower() for s in p.relative_to(root).parts]
        lab = None
        if any("fake" in s for s in parts):
            lab = 1
        elif any(s in ("real", "true", "original", "authentic") or "real" in s for s in parts):
            lab = 0
        if lab is None:
            continue
        group = "/".join(p.relative_to(root).parts[:2]) if len(parts) > 2 else parts[0]
        items.append({"img": p, "mask": None, "label": lab, "group": group, "id": str(p.relative_to(root))})
    return items


CLASS_NAMES = ["pristine", "tampered", "fake"]


def label_3class(manip: str) -> int:
    """Same mapping as ForensicsDataset3C in dinov3_forensics_train_v2.py."""
    manip = manip.lower().strip()
    if "pristine" in manip:
        return 0
    if any(k in manip for k in ("inpaint", "partial", "building", "roads", "greenspace", "water")):
        return 1
    if any(k in manip for k in ("synthetic", "generated", "gen", "diff")):
        return 2
    raise ValueError(f"unknown manipulation type: {manip}")


def build_overhead(root: Path):
    """The main (May et al.) test split, read from metadata.csv. Two-class label = class > 0; the
    three-class label is kept as cls3; masks are used for the tampered class only (as in training)."""
    import csv
    items = []
    with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = label_3class(row.get("manipulation_type", ""))
            mask = root / row["mask_filename"] if c == 1 and row.get("mask_filename") else None
            items.append({"img": root / row["image_filename"], "mask": mask, "label": int(c > 0), "cls3": c,
                          "group": CLASS_NAMES[c], "id": Path(row["image_filename"]).stem})
    return items


DMAER_SIZE = 512


def build_dmaer(root: Path, matched: bool):
    """DM-AER-DeepFake-V1 official sample. Fakes are all 512x512; reals (Million-AID) range from
    137 to 4432 px. matched=True removes the resolution confound: keep only reals whose shorter
    side is >= 512 and centre-crop every image to 512x512 (a no-op for the fakes), so both classes
    reach the model at the same native size and go through the same 512->384 resize."""
    items = build_uw(root)
    if not matched:
        return items
    from PIL import Image
    keep = []
    for it in items:
        with Image.open(it["img"]) as im:
            w, h = im.size
        if min(w, h) >= DMAER_SIZE:
            keep.append(dict(it, crop=DMAER_SIZE))
    return keep


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def pix_stats(pred: np.ndarray, gt: np.ndarray):
    tp = float(np.sum((pred == 1) & (gt == 1)))
    fp = float(np.sum((pred == 1) & (gt == 0)))
    fn = float(np.sum((pred == 0) & (gt == 1)))
    tn = float(gt.size) - tp - fp - fn
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return iou, dice, mcc


def summarise(recs):
    """recs: list of dicts with label, p_manip, pred_class, group, iou/dice/mcc (optional)."""
    y = np.array([r["label"] for r in recs])
    p = np.array([r["p_manip"] for r in recs])
    out = {"n_images": int(len(recs)), "n_real": int((y == 0).sum()), "n_manipulated": int((y == 1).sum())}
    if out["n_real"] > 0 and out["n_manipulated"] > 0:
        out["auc_real_vs_manipulated"] = float(roc_auc_score(y, p))
        out["accuracy_thr0.5"] = float(((p > 0.5).astype(int) == y).mean())
        out["accuracy_argmax"] = float((np.array([r["pred_class"] > 0 for r in recs]).astype(int) == y).mean())
    if out["n_manipulated"] > 0:
        out["detection_rate_manipulated_thr0.5"] = float((p[y == 1] > 0.5).mean())
    if out["n_real"] > 0:
        out["false_alarm_rate_real_thr0.5"] = float((p[y == 0] > 0.5).mean())
    loc = [r for r in recs if "iou" in r]
    if loc:
        out["localization"] = {
            "n_images_with_mask": len(loc),
            "mean_iou": float(np.mean([r["iou"] for r in loc])),
            "mean_dice": float(np.mean([r["dice"] for r in loc])),
            "mean_mcc": float(np.mean([r["mcc"] for r in loc])),
        }
    if recs and all("cls3" in r for r in recs):
        from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
        y3 = np.array([r["cls3"] for r in recs]); p3 = np.array([r["pred_class"] for r in recs])
        if len(set(y3.tolist())) == 3:
            out["three_class"] = {
                "accuracy": float((y3 == p3).mean()),
                "macro_f1": float(f1_score(y3, p3, average="macro", labels=[0, 1, 2])),
                "balanced_accuracy": float(balanced_accuracy_score(y3, p3)),
                "tampered_recall": float((p3[y3 == 1] == 1).mean()),
                "confusion_matrix_rows_true_cols_pred": confusion_matrix(y3, p3, labels=[0, 1, 2]).tolist(),
            }
            pm = (p3 > 0).astype(int); ym = (y3 > 0).astype(int)
            out["two_class_from_argmax"] = {
                "accuracy": float((pm == ym).mean()),
                "f1_manipulated": float(f1_score(ym, pm)),
                "precision_manipulated": float((ym[pm == 1] == 1).mean()) if pm.sum() else float("nan"),
                "recall_manipulated": float((pm[ym == 1] == 1).mean()),
            }
    cls = np.array([r["pred_class"] for r in recs])
    out["predicted_class_counts"] = {"pristine": int((cls == 0).sum()), "tampered": int((cls == 1).sum()),
                                     "fake": int((cls == 2).sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fmow", "rsfake", "uw", "dmaer", "overhead"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dinov3", default=os.environ.get("DINOV3_CKPT", "facebook/dinov3-vitl16-pretrain-sat493m"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-gen", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--subset-csv", default=None, help="rsfake: subset csv from prepare_rsfake_subset.py")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--legacy-token-slice", action="store_true")
    ap.add_argument("--dmaer-matched", action="store_true",
                    help="dmaer: reals with shorter side >= 512 only, every image centre-cropped to 512x512")
    ap.add_argument("--no-edge-gate", action="store_true",
                    help="checkpoint was trained with ABL_EDGE_GATE=0 (the plain row)")
    ap.add_argument("--degrade", default="",
                    help="robustness perturbation before the 384 resize: jpeg:Q or scale:S (see degrade_bgr)")
    ap.add_argument("--limit", type=int, default=0, help="debug: evaluate only a seeded random subset of this size")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / f"cross_{args.dataset}_metrics.json"

    if args.dataset == "fmow":
        items = build_fmow(root)
    elif args.dataset == "rsfake" and args.subset_csv:
        items = build_rsfake_from_csv(root, Path(args.subset_csv).expanduser())
    elif args.dataset == "rsfake":
        items = build_rsfake(root, args.per_gen, args.seed, Path(args.real_dir).expanduser() if args.real_dir else None)
    elif args.dataset == "dmaer":
        items = build_dmaer(root, args.dmaer_matched)
    elif args.dataset == "overhead":
        items = build_overhead(root)
    else:
        items = build_uw(root)
    print(f"[data] {args.dataset}: {len(items)} images; groups:",
          dict(sorted(((g, sum(1 for i in items if i['group'] == g)) for g in {i['group'] for i in items}))))
    if not items:
        sys.exit("no images found")
    if args.limit and args.limit < len(items):
        random.Random(args.seed).shuffle(items)
        items = sorted(items[:args.limit], key=lambda d: d["id"])
        print(f"[data] --limit {args.limit}: debug subset only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3ViTSegFormer3C(dinov3_ckpt=args.dinov3, embed_dim=256, num_classes=3,
                                 use_edge_gate=not args.no_edge_gate, legacy_token_slice=args.legacy_token_slice).to(device)
    state = torch.load(args.ckpt, map_location=device)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    sd = remap_state_dict_keys(sd, model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint key mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    print(f"[load] {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)} epoch={state.get('epoch') if isinstance(state, dict) else None}")
    if missing:
        print("[load] missing keys:", missing[:10])
    model.eval()

    dl = DataLoader(ListDataset(items, args.degrade), batch_size=args.batch, shuffle=False, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True)
    recs = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for b in tqdm(dl, desc=f"[eval] {args.dataset}"):
            x = b["pixel_values"].to(device, non_blocking=True)
            p_cls, seg = tta_predict_3class_and_seg(model, x, use_tta=True, use_amp=True,
                                                    channels_last=False, heavy_tta=False)
            for k in range(x.shape[0]):
                r = {"id": b["id"][k], "group": b["group"][k], "label": int(b["label"][k]),
                     "p_pristine": float(p_cls[k, 0]), "p_tampered": float(p_cls[k, 1]),
                     "p_fake": float(p_cls[k, 2]), "p_manip": float(p_cls[k, 1] + p_cls[k, 2]),
                     "pred_class": int(np.argmax(p_cls[k]))}
                if b["cls3"][k] >= 0:
                    r["cls3"] = int(b["cls3"][k])
                if b["has_mask"][k]:
                    gt = b["mask"][k].numpy().astype(np.uint8)
                    pred = (seg[k, 0] >= THR_MASK).astype(np.uint8)
                    if gt.sum() > 0:
                        r["iou"], r["dice"], r["mcc"] = pix_stats(pred, gt)
                recs.append(r)
    elapsed = time.perf_counter() - t0

    result = {
        "dataset": args.dataset, "root": str(root), "checkpoint": args.ckpt, "dinov3": args.dinov3,
        "seed": args.seed, "per_gen": args.per_gen if args.dataset == "rsfake" else None,
        "dmaer_matched": (args.dmaer_matched if args.dataset == "dmaer" else None),
        "dmaer_note": ("reals with shorter side >= 512 only; every image centre-cropped to 512x512 before the 384 resize"
                       if args.dataset == "dmaer" and args.dmaer_matched else None),
        "degrade": args.degrade or None, "edge_gate": not args.no_edge_gate,
        "preprocessing": "BGR->RGB, cv2.INTER_AREA 384x384, /255, no mean-std; TTA none/hflip/vflip/hvflip; mask thr 0.5",
        "scoring": "P(manipulated)=P(tampered)+P(fake); threshold 0.5",
        "overall": summarise(recs),
        "per_group": {g: summarise([r for r in recs if r["group"] == g]) for g in sorted({r["group"] for r in recs})},
        "wall_sec": elapsed,
        "torch_version": torch.__version__,
    }
    out.write_text(json.dumps(result, indent=2))
    (out.with_suffix(".per_image.json")).write_text(json.dumps(recs))
    print(json.dumps(result["overall"], indent=2))
    for g, s in result["per_group"].items():
        print(g, {k: v for k, v in s.items() if k != "predicted_class_counts"})
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
