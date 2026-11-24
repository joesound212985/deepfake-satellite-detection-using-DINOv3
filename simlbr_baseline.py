#!/usr/bin/env python3
"""
[baseline] SimLBR-style detector (Dhakal et al., "SimLBR: Learning to Detect Fake Images by
Learning to Detect Real Images", arXiv:2602.20412, CVPR 2026) trained on the May et al.
training split and scored on the same test set and cross-dataset sets as our model.

This is a RE-IMPLEMENTATION from the paper text (the authors' code was not available when
this was written). What the paper states, and is followed here:
  - frozen DINOv3 feature extractor, embeddings pre-computed once
  - Latent Blending Regularization: emb = a*emb(real) + (1-a)*emb(fake), a ~ U(0.5, 0.8),
    the blend is labelled fake; un-blended real embeddings are labelled real (their Sec. 12:
    "whenever a fake image is encountered, we sample a real image to form a pair")
  - two-layer MLP with ReLU, binary cross-entropy
  - Adam, lr 1e-4, weight decay 1e-2, batch size 500, at most 10 epochs
What the paper does NOT state, and was chosen here (all recorded in the output json):
  - backbone variant: facebook/dinov3-vitl16-pretrain-sat493m (the same backbone as our model)
  - embedding: pooler_output (CLS token after the final layer norm)
  - input: RGB, cv2.INTER_AREA resize to 384x384 (same as our model), then the backbone's own
    mean/std normalisation
  - MLP hidden width 1024, no dropout, no embedding normalisation
  - epoch chosen by balanced accuracy on OUR validation split (same VAL_SEED / VAL_SPLIT as the
    main script; un-blended real vs manipulated); never a test split
  - decision threshold 0.5, no test-time augmentation
Binary task: real = pristine; manipulated = tampered + fake (same two-class scoring as
eval_cross_dataset.py). Detection only -- this baseline produces no masks.

--no-lbr trains the identical head on un-blended fakes (control: same features, no LBR).

Usage:
  python simlbr_baseline.py                       # seeds 42 43 44, LBR  -> simlbr_metrics.json
  python simlbr_baseline.py --no-lbr              # control              -> simlbr_nolbr_metrics.json
"""
import os, sys, csv, json, random, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from transformers import AutoModel, AutoImageProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cross_dataset import read_rgb, build_fmow, build_rsfake_from_csv, build_uw  # noqa: E402

DATA = Path(os.environ.get("DATA", str(Path.home() / "data")))
TRAIN_ROOT = Path(os.environ.get("TRAIN_ROOT", str(DATA / "overhead/train_jpg/train")))
TEST_ROOT = Path(os.environ.get("TEST_ROOT", str(DATA / "overhead/test_jpg/test")))
VAL_SPLIT = float(os.environ.get("VAL_SPLIT", "0.1"))
VAL_SEED = int(os.environ.get("VAL_SEED", "42"))
BACKBONE = "facebook/dinov3-vitl16-pretrain-sat493m"
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


def overhead_items(root: Path):
    items = []
    with open(root / "metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = label_3class(row.get("manipulation_type", ""))
            items.append({"img": root / row["image_filename"], "label": int(c > 0),
                          "group": CLASS_NAMES[c], "id": Path(row["image_filename"]).stem})
    return items


def val_ids():
    """Identical split to the main script (ids shuffled with random.Random(VAL_SEED))."""
    ids = []
    with open(TRAIN_ROOT / "metadata.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(Path(row["image_filename"]).stem)
    random.Random(VAL_SEED).shuffle(ids)
    return set(ids[:max(1, int(len(ids) * VAL_SPLIT))])


class ImgList(Dataset):
    def __init__(self, items, mean, std):
        self.items, self.mean, self.std = items, np.array(mean, np.float32), np.array(std, np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = read_rgb(Path(self.items[i]["img"])).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()


@torch.no_grad()
def embed(name: str, items, cache_dir: Path, batch=32, workers=8):
    """Frozen-backbone embeddings, cached on disk (keyed by dataset name and item count)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{name}_n{len(items)}.npy"
    if f.exists():
        return np.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(BACKBONE)
    model = AutoModel.from_pretrained(BACKBONE).to(device).eval()
    dl = DataLoader(ImgList(items, proc.image_mean, proc.image_std), batch_size=batch, shuffle=False,
                    num_workers=workers, pin_memory=True)
    out = []
    for x in tqdm(dl, desc=f"[embed] {name}"):
        e = model(pixel_values=x.to(device, non_blocking=True)).pooler_output
        out.append(e.float().cpu().numpy())
    out = np.concatenate(out)
    if not np.isfinite(out).all():
        raise RuntimeError(f"non-finite embeddings for {name}")
    np.save(f, out)
    del model
    torch.cuda.empty_cache()
    return out


def summarise(y, p, groups):
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    out = {"n_images": int(len(y)), "n_real": int((y == 0).sum()), "n_manipulated": int((y == 1).sum())}
    if out["n_real"] and out["n_manipulated"]:
        out["auc_real_vs_manipulated"] = float(roc_auc_score(y, p))
        out["accuracy_thr0.5"] = float(((p > 0.5).astype(int) == y).mean())
        out["balanced_accuracy_thr0.5"] = float(balanced_accuracy_score(y, (p > 0.5).astype(int)))
    if out["n_manipulated"]:
        out["detection_rate_manipulated_thr0.5"] = float((p[y == 1] > 0.5).mean())
    if out["n_real"]:
        out["false_alarm_rate_real_thr0.5"] = float((p[y == 0] > 0.5).mean())
    per = {}
    real_p = p[y == 0]
    for g in sorted(set(groups.tolist())):
        m = groups == g
        d = {"n": int(m.sum()), "label": int(y[m][0]), "flagged_rate_thr0.5": float((p[m] > 0.5).mean())}
        if y[m][0] == 1 and len(real_p):   # this group's fakes against ALL reals of the dataset
            d["auc_vs_all_reals"] = float(roc_auc_score(np.r_[np.zeros(len(real_p)), np.ones(int(m.sum()))],
                                                        np.r_[real_p, p[m]]))
        per[g] = d
    out["per_group"] = per
    return out


def train_head(Xr, Xf, Xval, yval, seed, lbr=True, hidden=1024, epochs=10, bs=500, lr=1e-4, wd=1e-2,
               a_lo=0.5, a_hi=0.8):
    """Returns (best_state_dict, info). Xr / Xf: train embeddings of real / manipulated images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    head = nn.Sequential(nn.Linear(Xr.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Xr_t, Xf_t = torch.from_numpy(Xr), torch.from_numpy(Xf)
    Xv = torch.from_numpy(Xval).to(device)
    n_r, n_f = len(Xr_t), len(Xf_t)
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        head.train()
        # one pass over every training image: reals as they are; every fake paired with a random real
        if lbr:
            a = torch.empty(n_f, 1).uniform_(a_lo, a_hi, generator=g)
            partner = torch.randint(0, n_r, (n_f,), generator=g)
            Xfake = a * Xr_t[partner] + (1.0 - a) * Xf_t
        else:
            Xfake = Xf_t
        X = torch.cat([Xr_t, Xfake])
        y = torch.cat([torch.zeros(n_r), torch.ones(n_f)])
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            xb, yb = X[idx].to(device), y[idx].to(device)
            loss = lossf(head(xb).squeeze(1), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss in SimLBR head training")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head(Xv).squeeze(1)).cpu().numpy()
        bacc = float(balanced_accuracy_score(yval, (pv > 0.5).astype(int)))
        hist.append({"epoch": ep + 1, "train_loss": tot / len(X), "val_balanced_acc": bacc,
                     "val_auc": float(roc_auc_score(yval, pv))})
        if bacc > best:
            best, best_state = bacc, {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, {"best_val_balanced_acc": best, "best_epoch": int(np.argmax([h["val_balanced_acc"] for h in hist]) + 1),
                  "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--no-lbr", action="store_true")
    ap.add_argument("--per-gen", type=int, default=500)
    ap.add_argument("--cache", default=str(DATA / "features" / "dinov3_sat493m_pooler_384"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=10, help="paper: at most 10; larger = disclosed deviation")
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    out = Path(args.out) if args.out else here / ("simlbr_nolbr_metrics.json" if args.no_lbr else "simlbr_metrics.json")
    cache = Path(args.cache)

    # ---- source data (May et al.): train / val split identical to the main script
    tr_all = overhead_items(TRAIN_ROOT)
    vid = val_ids()
    E = embed("overhead_train", tr_all, cache)
    is_val = np.array([it["id"] in vid for it in tr_all])
    y_all = np.array([it["label"] for it in tr_all])
    Xr, Xf = E[(~is_val) & (y_all == 0)], E[(~is_val) & (y_all == 1)]
    Xval, yval = E[is_val], y_all[is_val]
    print(f"[data] train real={len(Xr)} manipulated={len(Xf)} | val n={len(yval)} (real={(yval == 0).sum()})")

    # ---- evaluation sets (None of them is used for training or model selection)
    evalsets = {}
    te = overhead_items(TEST_ROOT)
    evalsets["overhead_test"] = (te, embed("overhead_test", te, cache))
    fm = DATA / "fmow-fake-small"
    if (fm / "train").exists() or (fm / "val").exists():
        it = build_fmow(fm)
        evalsets["cross_fmow"] = (it, embed("fmow", it, cache))
    rs = DATA / "rsfake"
    sub = rs / f"subset_seed42_n{args.per_gen}.csv"
    if sub.exists():
        it = build_rsfake_from_csv(rs, sub)
        evalsets["cross_rsfake"] = (it, embed(f"rsfake_seed42_pergen{args.per_gen}", it, cache))
    uw = DATA / "uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
    if uw.exists():
        it = build_uw(uw)
        evalsets["cross_uw"] = (it, embed("uw", it, cache))
    skipped = [n for n in ("cross_fmow", "cross_rsfake", "cross_uw") if n not in evalsets]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = []
    for seed in args.seeds:
        t0 = time.perf_counter()
        head, info = train_head(Xr, Xf, Xval, yval, seed, lbr=not args.no_lbr, epochs=args.epochs)
        info["train_sec"] = time.perf_counter() - t0
        res = {"seed": seed, "selection": info, "n_head_parameters": sum(p.numel() for p in head.parameters())}
        for name, (items, X) in evalsets.items():
            with torch.no_grad():
                p = torch.sigmoid(head(torch.from_numpy(X).to(device)).squeeze(1)).cpu().numpy()
            res[name] = summarise([i["label"] for i in items], p, [i["group"] for i in items])
        runs.append(res)
        print(f"[seed {seed}] best epoch {info['best_epoch']} val bal-acc {info['best_val_balanced_acc']:.4f} | " +
              " | ".join(f"{n}: acc {res[n].get('accuracy_thr0.5', float('nan')):.4f} auc {res[n].get('auc_real_vs_manipulated', float('nan')):.4f}"
                         for n in evalsets))

    # ---- mean / std over seeds (sample std, ddof=1)
    agg = {}
    for name in evalsets:
        agg[name] = {}
        for k in ("accuracy_thr0.5", "balanced_accuracy_thr0.5", "auc_real_vs_manipulated",
                  "detection_rate_manipulated_thr0.5", "false_alarm_rate_real_thr0.5"):
            v = [r[name][k] for r in runs if k in r[name]]
            if v:
                agg[name][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                "n_seeds": len(v)}
    result = {
        "method": "SimLBR-style re-implementation" + (" WITHOUT latent blending (control)" if args.no_lbr else ""),
        "reference": "Dhakal et al., arXiv:2602.20412 (CVPR 2026); re-implemented from the paper text, not the authors' code",
        "config": {"backbone": BACKBONE, "embedding": "pooler_output (CLS after final norm), backbone frozen",
                   "input": "RGB, cv2.INTER_AREA 384x384, /255, backbone mean/std", "head": "Linear(1024,1024)-ReLU-Linear(1024,1)",
                   "lbr": not args.no_lbr, "alpha": "U(0.5, 0.8), blend = a*real + (1-a)*fake, labelled fake",
                   "optimizer": "Adam lr 1e-4 wd 1e-2", "batch_size": 500, "max_epochs": args.epochs,
                   "epoch_selection": "best balanced accuracy on the source validation split (un-blended)",
                   "threshold": 0.5, "tta": False, "val_split": VAL_SPLIT, "val_seed": VAL_SEED,
                   "train_real": int(len(Xr)), "train_manipulated": int(len(Xf)), "val_n": int(len(yval)),
                   "labels": "real = pristine; manipulated = tampered + fake"},
        "seeds": args.seeds, "skipped_eval_sets": skipped,
        "mean_std_over_seeds": agg, "runs": runs,
        "torch_version": torch.__version__,
    }
    out.write_text(json.dumps(result, indent=2))
    print(f"[save] {out}")


if __name__ == "__main__":
    main()
