#!/usr/bin/env python3
"""
SimLBR baseline trained and evaluated with the AUTHORS' RELEASED CODE (github.com/mvrl/SimLBR,
commit ac90f86, unmodified checkout), on our data.

What is theirs (imported from the checkout, not copied): the dataset classes and the 256-px
transform (simlbr.datasets / simlbr.utils), the model with Latent Blending Regularization and its
training / validation / test steps (simlbr.model.SimLBR), the training entry point with its
checkpoint selection on val_acc (simlbr.train.main), the argument defaults (simlbr.config) and the
per-subset evaluation (simlbr.evaluate.evaluate_subset).

What this driver adds (all at run time; no file of the checkout is edited):
  1. Data layout. Our images are exposed in the AIGCDetectionBenchMark folder layout their dataset
     classes read (train/ProGAN/<category>/{0_real,1_fake}/*.png and test/<subset>/{0_real,1_fake}/*)
     through symlinks. Files that PIL would not read like our pipeline does (TIFF) and the
     size-matched DM-AER crops are written once as lossless PNG. real = pristine,
     fake = tampered + fake. Training uses the same train/validation split as the main model.
  2. Backbone weights. Their loader needs a Meta download key (DINO_V3_KEY). We load the same
     DINOv3 ViT-L/16 architecture from Hugging Face instead and return the same CLS token
     (x_norm_clstoken == pooler_output). `--backbone sat|web` picks the weights.
  3. Subset names. Our evaluation-set names are appended to AIGCDataset.all_models so that their
     dataset class accepts them.
Hyperparameters are the release script's (scripts/train_aigc_lbr.sh): batch 200, 5 epochs, lr 1e-4,
weight decay 1e-2, 2 hidden layers, ReLU, dropout 0.3, LBR alpha ~ U(0.5, 0.8), 16-mixed precision.
Seeds 42/43/44 (their script fixes 56). AUC is computed here from the per-image scores their test
loop produces; accuracy / AP / F1 are the values their code logs.

  python simlbr_official.py --backbone sat          # layout + 2 variants x 3 seeds + aggregate
  python simlbr_official.py --backbone web
Outputs: simlbr_official_<backbone>_metrics.json, per-run files under simlbr_official_runs/.
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import numpy as np

G = Path(__file__).resolve().parent
OFFICIAL = Path(os.environ.get("SIMLBR_REPO", str(Path.home() / "third_party/SimLBR")))
COMMIT = "ac90f86"
DATA = Path(os.environ.get("DATA", str(Path.home() / "data")))
REPOS = {"sat": "facebook/dinov3-vitl16-pretrain-sat493m", "web": "facebook/dinov3-vitl16-pretrain-lvd1689m"}
SUBSETS = ["overhead_val", "overhead_test", "cross_fmow", "cross_rsfake", "cross_uw", "cross_dmaer"]
NATIVE_OK = {".jpg", ".jpeg", ".png"}


def check_checkout():
    if not (OFFICIAL / "simlbr" / "train.py").exists():
        subprocess.run(["git", "clone", "-q", "https://github.com/mvrl/SimLBR", str(OFFICIAL)], check=True)
        subprocess.run(["git", "-C", str(OFFICIAL), "checkout", "-q", COMMIT], check=True)
    head = subprocess.check_output(["git", "-C", str(OFFICIAL), "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(OFFICIAL), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if not head.startswith(COMMIT) or dirty:
        raise SystemExit(f"official checkout must be an unmodified {COMMIT}; found HEAD={head} dirty={bool(dirty)}")
    return head


# ------------------------------------------------------------------ data layout (ours)
def build_layout(layout: Path, dry: int = 0):
    """Expose our images in the folder layout the official dataset classes read. Idempotent."""
    if (layout / ".done").exists():
        return json.load(open(layout / "manifest.json"))
    import cv2
    sys.path.insert(0, str(G))
    from simlbr_baseline import overhead_items, val_ids, read_rgb_native, TRAIN_ROOT, TEST_ROOT
    from eval_cross_dataset import build_fmow, build_rsfake_from_csv, build_uw, build_dmaer

    vids = val_ids()
    tr_all = overhead_items(TRAIN_ROOT)
    sets = {"train": [it for it in tr_all if it["id"] not in vids],
            "overhead_val": [it for it in tr_all if it["id"] in vids],
            "overhead_test": overhead_items(TEST_ROOT)}
    fm = DATA / "fmow-fake-small"
    if fm.exists():
        sets["cross_fmow"] = build_fmow(fm)
    sub = DATA / "rsfake/subset_seed42_n500.csv"
    if sub.exists():
        sets["cross_rsfake"] = build_rsfake_from_csv(DATA / "rsfake", sub)
    uw = DATA / "uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
    if uw.exists():
        sets["cross_uw"] = build_uw(uw)
    dm = DATA / "dm_aer/DM-AER-DeepFake-V1-sample"
    if dm.exists():
        sets["cross_dmaer"] = build_dmaer(dm, True)      # size-matched, as in eval_cross_dataset.py

    manifest = {}
    for name, items in sets.items():
        if dry:      # debug: a few items per class
            items = [i for i in items if i["label"] == 0][:dry] + [i for i in items if i["label"] == 1][:dry]
        base = layout / ("train/ProGAN/overhead" if name == "train" else f"test/{name}")
        man = {}
        for lab, folder in ((0, "0_real"), (1, "1_fake")):
            d = base / folder
            d.mkdir(parents=True, exist_ok=True)
            for k, it in enumerate([i for i in items if i["label"] == lab]):
                src = Path(it["img"]).resolve()
                crop = it.get("crop", 0)
                materialise = bool(crop) or src.suffix.lower() not in NATIVE_OK
                # the official training class globs *.png, so every link is named .png (PIL reads by content)
                dst = d / f"{k:06d}.png"
                if not dst.exists():
                    if materialise:
                        img = read_rgb_native(src)
                        if crop:
                            h, w = img.shape[:2]
                            t, l = (h - crop) // 2, (w - crop) // 2
                            img = img[t:t + crop, l:l + crop]
                        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    else:
                        os.symlink(src, dst)
                man[f"{folder}/{dst.name}"] = {"id": str(it["id"]), "group": it["group"], "materialised_png": materialise}
        manifest[name] = man
    (layout / "manifest.json").write_text(json.dumps(manifest))
    (layout / ".done").write_text("ok")
    return manifest


# ------------------------------------------------------------------ run-time patches (see docstring)
def patch_official(repo: str, subsets):
    sys.path.insert(0, str(OFFICIAL))
    import torch.nn as nn
    from transformers import AutoModel
    import simlbr.backbones as B
    from simlbr.datasets import AIGCDataset

    class HFDinoV3(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = AutoModel.from_pretrained(repo)

        def forward_features(self, x):
            return {"x_norm_clstoken": self.m(pixel_values=x).pooler_output}

    B.get_dinov3 = lambda: (HFDinoV3(), 1024)
    AIGCDataset.all_models = list(AIGCDataset.all_models) + [s for s in subsets if s not in AIGCDataset.all_models]


# ------------------------------------------------------------------ one training + evaluation run
def run_one(a):
    import torch
    import lightning.pytorch as pl
    from sklearn.metrics import roc_auc_score

    layout = Path(a.layout)
    manifest = json.load(open(layout / "manifest.json"))
    subsets = [s for s in SUBSETS if s in manifest]
    patch_official(REPOS[a.backbone], subsets)
    from simlbr.config import get_args
    from simlbr import train as T
    from simlbr.utils import set_seed
    from simlbr.model import SimLBR
    from simlbr.datasets import AIGCDataset
    from simlbr.evaluate import evaluate_subset

    run = f"{'lbr' if a.lbr else 'nolbr'}_s{a.seed}"
    rundir = Path(a.runs) / a.backbone / run
    out = rundir / "metrics.json"
    if out.exists():
        print(f"[skip] {out}")
        return
    rundir.mkdir(parents=True, exist_ok=True)
    logdir = rundir / "lightning"
    argv = ["--dataset_name", "aigc", "--data_dir", str(layout), "--train_model", "ProGAN",
            "--val_model", "overhead_val", "--ds_fraction", "1.0",
            "--batch_size", str(a.batch_size), "--num_workers", str(a.workers),
            "--max_epochs", str(a.epochs), "--devices", "1", "--precision", a.precision,
            "--accelerator", a.accelerator, "--val_check_interval", "1.0", "--backbone", "dinov3",
            "--lr", "1e-4", "--wt_decay", "1e-2", "--activation", "relu", "--hidden_layers", "2", "--dropout", "0.3",
            "--log_dir", str(logdir), "--run_name", run, "--project_name", "SimLBR_overhead", "--wandb_mode", "disabled"]
    if a.lbr:
        argv += ["--lbr", "--lbrdist", "0.5", "0.8"]
    args = get_args(argv)
    set_seed(a.seed)
    torch.set_float32_matmul_precision("medium")
    T.main(args)                                            # official training loop

    ckdir = logdir / "SimLBR_overhead" / run / "checkpoints"
    best = sorted(ckdir.glob("*val_acc*.ckpt"))
    if len(best) != 1:
        raise RuntimeError(f"expected one best-val_acc checkpoint in {ckdir}, found {len(best)}")
    model = SimLBR.load_from_checkpoint(str(best[0]))      # official loader
    trainer = pl.Trainer(accelerator=a.accelerator, devices=1, precision=a.precision, logger=False, enable_checkpointing=False)

    class EvalArgs:
        data_dir = str(layout); batch_size = a.batch_size; num_workers = a.workers

    res = {"run": run, "seed": a.seed, "lbr": bool(a.lbr), "backbone_repo": REPOS[a.backbone],
           "selected_checkpoint": best[0].name, "subsets": {}}
    for s in subsets:
        row = evaluate_subset(EvalArgs, trainer, model, AIGCDataset, s)      # official evaluation
        p = model.all_preds.float().cpu().numpy(); y = model.all_labels.int().cpu().numpy()
        files = AIGCDataset(root_dir=str(layout), model=s, mode="test", fraction=1.0).all_files
        if len(files) != len(p):
            raise RuntimeError(f"{s}: {len(files)} files but {len(p)} predictions")
        keys = ["/".join(Path(f).parts[-2:]) for f in files]
        if any(int(k.startswith("1_fake")) != int(l) for k, l in zip(keys, y)):
            raise RuntimeError(f"{s}: file order does not match label order")
        groups = [manifest[s][k]["group"] for k in keys]
        d = {"n_images": int(len(y)), "n_real": int((y == 0).sum()), "n_manipulated": int((y == 1).sum()),
             "official": {k: float(v) for k, v in row.items() if k != "subset"},
             "accuracy_thr0.5": float(((p > 0.5).astype(int) == y).mean()),
             "detection_rate_manipulated_thr0.5": float((p[y == 1] > 0.5).mean()),
             "false_alarm_rate_real_thr0.5": float((p[y == 0] > 0.5).mean()),
             "auc_real_vs_manipulated": float(roc_auc_score(y, p)),
             "per_group_flagged_rate_thr0.5": {g: float(np.mean([pp > 0.5 for pp, gg in zip(p, groups) if gg == g])) for g in sorted(set(groups))}}
        res["subsets"][s] = d
        (rundir / f"{s}.per_image.json").write_text(json.dumps(
            [{"id": manifest[s][k]["id"], "group": g, "label": int(l), "p_manip": float(pp)} for k, g, l, pp in zip(keys, groups, y, p)]))
        print(s, {k: round(v, 4) for k, v in d.items() if isinstance(v, float)}, flush=True)
    model.export_pretrained(str(rundir / "head"))          # official head export (2.6 MB)
    shutil.rmtree(ckdir, ignore_errors=True)               # full Lightning checkpoints include the 1.2 GB frozen backbone
    out.write_text(json.dumps(res, indent=2))
    print(f"[save] {out}")


# ------------------------------------------------------------------ aggregate over seeds
def aggregate(a, head):
    import torch
    res = {"method": "SimLBR, authors' released code (github.com/mvrl/SimLBR) trained on our data",
           "official_commit": head, "backbone_repo": REPOS[a.backbone],
           "config": {"batch_size": a.batch_size, "max_epochs": a.epochs, "lr": 1e-4, "wt_decay": 1e-2, "hidden_layers": 2,
                      "activation": "relu", "dropout": 0.3, "lbrdist": [0.5, 0.8], "precision": a.precision,
                      "resize": 256, "normalisation": "official make_dinov3_transform (0.430,0.411,0.296)/(0.213,0.156,0.143)",
                      "model_selection": "official ModelCheckpoint, best val_acc on our validation split (overhead_val)"},
           "runtime_patches": ["backbone weights loaded from Hugging Face instead of DINO_V3_KEY (same CLS token)",
                               "our subset names appended to AIGCDataset.all_models",
                               "data exposed in AIGC folder layout via symlinks; TIFF and DM-AER crops written as PNG"],
           "labels": "real = pristine; manipulated = tampered + fake", "seeds": a.seeds,
           "torch_version": torch.__version__, "variants": {}}
    for variant in ("lbr", "nolbr"):
        runs = []
        for s in a.seeds:
            f = Path(a.runs) / a.backbone / f"{variant}_s{s}" / "metrics.json"
            if f.exists():
                runs.append(json.load(open(f)))
        if not runs:
            continue
        ms = {}
        for sub in runs[0]["subsets"]:
            ms[sub] = {}
            keys = ["accuracy_thr0.5", "auc_real_vs_manipulated", "detection_rate_manipulated_thr0.5", "false_alarm_rate_real_thr0.5"]
            for k in keys:
                v = [r["subsets"][sub][k] for r in runs]
                ms[sub][k] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n_seeds": len(v)}
            for k in ("ap", "f1"):
                v = [r["subsets"][sub]["official"][k] for r in runs]
                ms[sub][f"official_{k}"] = {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n_seeds": len(v)}
        res["variants"][variant] = {"mean_std_over_seeds": ms, "runs": runs}
    # debug runs (--dry) never write next to the real results
    out = (Path(a.runs) if a.dry else G) / f"simlbr_official_{a.backbone}_metrics.json"
    if a.dry:
        res["DRY_RUN_NOT_A_RESULT"] = a.dry
    out.write_text(json.dumps(res, indent=2))
    for variant, v in res["variants"].items():
        for sub, m in v["mean_std_over_seeds"].items():
            print(f"{variant:6s} {sub:14s} acc={m['accuracy_thr0.5']['mean']:.4f} auc={m['auc_real_vs_manipulated']['mean']:.4f} n={m['accuracy_thr0.5']['n_seeds']}")
    print(f"[save] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=list(REPOS), default="sat")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--layout", default=str(DATA / "simlbr_official_layout"))
    ap.add_argument("--runs", default=str(G / "simlbr_official_runs"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--precision", default="16-mixed")
    ap.add_argument("--accelerator", default="gpu")
    ap.add_argument("--dry", type=int, default=0, help="debug: this many images per class per set")
    ap.add_argument("--stage", default="all", choices=["all", "run"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lbr", action="store_true")
    a = ap.parse_args()

    head = check_checkout()
    if a.stage == "run":
        return run_one(a)
    build_layout(Path(a.layout), a.dry)
    for seed in a.seeds:
        for lbr in (True, False):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--stage", "run", "--backbone", a.backbone, "--seed", str(seed),
                   "--layout", a.layout, "--runs", a.runs, "--epochs", str(a.epochs), "--batch-size", str(a.batch_size),
                   "--workers", str(a.workers), "--precision", a.precision, "--accelerator", a.accelerator] + (["--lbr"] if lbr else [])
            rc = subprocess.run(cmd, env=dict(os.environ, WANDB_MODE="disabled")).returncode
            if rc != 0:
                raise SystemExit(f"run failed (seed {seed}, lbr {lbr}), exit {rc}")
    aggregate(a, head)


if __name__ == "__main__":
    main()
