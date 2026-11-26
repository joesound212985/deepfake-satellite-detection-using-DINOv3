#!/usr/bin/env python3
"""Few-shot adaptation experiment. Protocol: FEWSHOT_PROTOCOL.md (fixed before any number was computed).

  python fewshot_adapt.py extract     # GPU: CLS features of 4 frozen extractors on the 4 external sets (cached)
  python fewshot_adapt.py analyse     # CPU: splits, linear probes, threshold calibration -> fewshot_metrics.json

Nothing is trained except a logistic-regression probe on frozen features. No existing file is overwritten.
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_cross_dataset import (ListDataset, collate, build_fmow, build_rsfake_from_csv, build_uw, build_dmaer, build_gdd)  # noqa: E402
from dinov3_forensics_train_v2 import DINOv3ViTSegFormer3C, remap_state_dict_keys  # noqa: E402

HOME = Path.home()
DATA = HOME / "data"
RES = HOME / "results_dinov3" / "vit"
SAT = "facebook/dinov3-vitl16-pretrain-sat493m"
WEB = "facebook/dinov3-vitl16-pretrain-lvd1689m"
EXTRACTORS = {  # name: (hf repo, checkpoint or None)
    "full_s42": (SAT, RES / "dinov3-vitl16-pretrain-sat493m" / "full_s42" / "best_checkpoint.pt"),
    "web_s42":  (WEB, RES / "dinov3-vitl16-pretrain-lvd1689m" / "web_s42" / "best_checkpoint.pt"),
    "sat_orig": (SAT, None),
    "web_orig": (WEB, None),
}
ZEROSHOT = {"full_s42": "cross_{d}_metrics.per_image.json", "web_s42": "cross_web_{d}_metrics.per_image.json"}
DATASETS = ["rsfake", "uw", "dmaer", "fmow"]
GDD_SETS = ["gdd_indomain", "gdd_crossgenerator", "gdd_crossdomain", "gdd_localization"]   # addendum in FEWSHOT_PROTOCOL.md
KS = [4, 16, 64, 256]
DRAWS = 10
FEAT = DATA / "features" / "fewshot_cls384"


def build(name):
    if name == "fmow":
        return build_fmow(DATA / "fmow-fake-small")
    if name == "rsfake":
        return build_rsfake_from_csv(DATA / "rsfake", DATA / "rsfake" / "subset_seed42_n500.csv")
    if name == "uw":
        return build_uw(DATA / "uw_fake_sat" / "anti-deepfake" / "anti-deepfake-data and code" / "data")
    if name == "dmaer":
        return build_dmaer(DATA / "dm_aer" / "DM-AER-DeepFake-V1-sample", True)
    if name.startswith("gdd_"):
        return build_gdd(DATA / "gdd", name[4:])
    raise ValueError(name)


# ---------------------------------------------------------------------------------------------- extract
@torch.no_grad()
def extract(limit=0, out_dir=FEAT, batch=16, workers=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sets = {d: build(d) for d in DATASETS}
    for d, items in sets.items():
        ids = [it["id"] for it in items]
        assert len(set(ids)) == len(ids), f"duplicate ids in {d}"
        if limit:
            sets[d] = items[:: max(1, len(items) // limit)][:limit]
    for ex, (repo, ck) in EXTRACTORS.items():
        todo = [d for d in DATASETS if not (out_dir / ex / f"{d}.npy").exists()]
        if not todo:
            continue
        model = DINOv3ViTSegFormer3C(dinov3_ckpt=repo, embed_dim=256, num_classes=3, use_edge_gate=True).to(device)
        if ck is not None:
            state = torch.load(ck, map_location=device)
            sd = state["model"] if isinstance(state, dict) and "model" in state else state
            missing, unexpected = model.load_state_dict(remap_state_dict_keys(sd, model), strict=False)
            if missing or unexpected:
                raise RuntimeError(f"{ex}: checkpoint key mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
        model.eval()
        (out_dir / ex).mkdir(parents=True, exist_ok=True)
        for d in todo:
            items = sets[d]
            dl = DataLoader(ListDataset(items), batch_size=batch, shuffle=False, num_workers=workers,
                            collate_fn=collate, pin_memory=True)
            fs = []
            for b in tqdm(dl, desc=f"[feat] {ex} / {d}"):
                x = b["pixel_values"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    cls = model.encode(x).last_hidden_state[:, 0]
                fs.append(cls.float().cpu().numpy())
            F = np.concatenate(fs)
            if not np.isfinite(F).all():
                raise RuntimeError(f"non-finite features: {ex}/{d}")
            np.save(out_dir / ex / f"{d}.npy", F)
            (out_dir / ex / f"{d}.ids.json").write_text(json.dumps(
                [{"id": it["id"], "label": int(it["label"]), "group": it["group"]} for it in items]))
            print(f"[save] {ex}/{d}: {F.shape}")
        del model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------- analyse
def make_split(meta):
    """50/50 stratified by group, RandomState(0). Returns sorted index arrays (pool, held)."""
    rs = np.random.RandomState(0)
    pool, held = [], []
    groups = sorted({m["group"] for m in meta})
    for g in groups:
        idx = np.array([i for i, m in enumerate(meta) if m["group"] == g])
        idx = idx[rs.permutation(len(idx))]
        h = len(idx) // 2
        pool += idx[:len(idx) - h].tolist(); held += idx[len(idx) - h:].tolist()
    return np.array(sorted(pool)), np.array(sorted(held))


def scores(y, s, thr):
    pred = (s >= thr).astype(int)
    tpr = float(pred[y == 1].mean()); tnr = float(1 - pred[y == 0].mean())
    return {"auc": float(roc_auc_score(y, s)), "accuracy": float((pred == y).mean()),
            "balanced_accuracy": (tpr + tnr) / 2, "detection_rate": tpr, "false_alarm_rate": 1 - tnr}


def agg(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else None,
                "n_draws": len(rows)} for k in rows[0]}


def best_threshold(y, s):
    """Threshold maximising balanced accuracy on the shots; ties -> midpoint of the widest best interval."""
    u = np.unique(s)
    cands = np.concatenate([[u[0] - 1e-6], (u[:-1] + u[1:]) / 2, [u[-1] + 1e-6]])
    ba = np.array([((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) / 2 for t in cands])
    best = np.flatnonzero(ba >= ba.max() - 1e-12)
    # widest run of consecutive best candidates
    runs, start = [], best[0]
    for a, b in zip(best[:-1], best[1:]):
        if b != a + 1:
            runs.append((start, a)); start = b
    runs.append((start, best[-1]))
    lo, hi = max(runs, key=lambda r: cands[r[1]] - cands[r[0]])
    return float((cands[lo] + cands[hi]) / 2)


def draw(rs, pool, y, k):
    r = rs.choice(pool[y[pool] == 0], k, replace=False); f = rs.choice(pool[y[pool] == 1], k, replace=False)
    return np.concatenate([r, f])


def analyse(feat_dir=FEAT, out=HERE / "fewshot_metrics.json", splits_out=HERE / "fewshot_splits.json"):
    if out.exists():
        sys.exit(f"{out} exists; not overwriting")
    result = {"protocol": "FEWSHOT_PROTOCOL.md", "ks": KS, "draws": DRAWS,
              "probe": "L2-normalised CLS (last_hidden_state[:,0], 384x384, /255) -> LogisticRegression(C=1.0, max_iter=2000)",
              "datasets": {}}
    splits = {}
    for d in DATASETS:
        meta = json.loads((feat_dir / "full_s42" / f"{d}.ids.json").read_text())
        y = np.array([m["label"] for m in meta]); groups = np.array([m["group"] for m in meta])
        pool, held = make_split(meta)
        splits[d] = {"pool": [meta[i]["id"] for i in pool], "held_out": [meta[i]["id"] for i in held]}
        kmax = int(min((y[pool] == 0).sum(), (y[pool] == 1).sum()))
        R = {"n_pool": int(len(pool)), "n_held_out": int(len(held)), "held_out_real": int((y[held] == 0).sum()),
             "held_out_manipulated": int((y[held] == 1).sum()), "max_k": kmax, "extractors": {}, "zero_shot": {}, "threshold_calibration": {}}
        # ---- linear probes
        for ex in EXTRACTORS:
            m2 = json.loads((feat_dir / ex / f"{d}.ids.json").read_text())
            assert [m["id"] for m in m2] == [m["id"] for m in meta], f"id order differs: {ex}/{d}"
            F = np.load(feat_dir / ex / f"{d}.npy").astype(np.float64)
            F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-12

            def fit_eval(tr):
                clf = LogisticRegression(C=1.0, max_iter=2000).fit(F[tr], y[tr])
                s = clf.predict_proba(F[held])[:, 1]
                r = scores(y[held], s, 0.5)
                pg = {g: float((s[groups[held] == g] >= 0.5).mean()) for g in sorted(set(groups[held]))}
                return r, pg
            E = {}
            for k in KS:
                if k > kmax:
                    E[f"k{k}"] = "skipped: pool has fewer than k images in one class"; continue
                rows, pgs = [], []
                for seed in range(DRAWS):
                    r, pg = fit_eval(draw(np.random.RandomState(seed), pool, y, k)); rows.append(r); pgs.append(pg)
                E[f"k{k}"] = agg(rows)
                E[f"k{k}"]["flagged_manipulated_rate_per_group_mean"] = {g: float(np.mean([p[g] for p in pgs])) for g in pgs[0]}
            r, pg = fit_eval(pool)
            E["whole_pool"] = dict(r, n_train=int(len(pool)), flagged_manipulated_rate_per_group=pg)
            R["extractors"][ex] = E
        # ---- zero-shot + threshold calibration from the saved per-image scores
        for ex, pat in ZEROSHOT.items():
            f = HERE / pat.format(d=d)
            if not f.exists():
                R["zero_shot"][ex] = f"missing {f.name}"; continue
            recs = {r["id"]: r for r in json.loads(f.read_text())}
            assert all(m["id"] in recs for m in meta), f"ids missing in {f.name}"
            assert all(recs[m["id"]]["label"] == m["label"] for m in meta)
            s = np.array([recs[m["id"]]["p_manip"] for m in meta])
            R["zero_shot"][ex] = scores(y[held], s[held], 0.5)
            C = {}
            for k in KS:
                if k > kmax:
                    continue
                rows = []
                for seed in range(DRAWS):
                    tr = draw(np.random.RandomState(seed), pool, y, k)
                    rows.append(scores(y[held], s[held], best_threshold(y[tr], s[tr])))
                C[f"k{k}"] = agg(rows)
            R["threshold_calibration"][ex] = C
        result["datasets"][d] = R
    if not splits_out.exists():
        splits_out.write_text(json.dumps(splits))
    out.write_text(json.dumps(result, indent=2))
    print(f"[save] {out}")
    for d, R in result["datasets"].items():
        print(f"\n{d}: held-out n={R['n_held_out']}")
        for ex in ZEROSHOT:
            z = R["zero_shot"][ex]
            if isinstance(z, dict):
                print(f"  zero-shot {ex:<9} AUC={z['auc']:.3f} acc={z['accuracy']:.3f}")
        for ex, E in R["extractors"].items():
            cells = [f"{k}: AUC={v['auc']['mean']:.3f} acc={v['accuracy']['mean']:.3f}" for k, v in E.items() if k.startswith("k") and isinstance(v, dict)]
            print(f"  probe {ex:<9} " + " | ".join(cells) + f" | pool: AUC={E['whole_pool']['auc']:.3f} acc={E['whole_pool']['accuracy']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "analyse"])
    ap.add_argument("--dry", type=int, default=0, help="debug: N images per set, written to a scratch dir, NOT a result")
    ap.add_argument("--scratch", default="")
    ap.add_argument("--sets", choices=["external", "gdd"], default="external")
    a = ap.parse_args()
    tag = ""
    if a.sets == "gdd":
        DATASETS[:] = GDD_SETS; tag = "_gdd"
    if a.dry:
        if not a.scratch:
            sys.exit("--dry needs --scratch DIR")
        sd = Path(a.scratch)
        if a.cmd == "extract":
            extract(limit=a.dry, out_dir=sd / "feat")
        else:
            analyse(feat_dir=sd / "feat", out=sd / "DRY_RUN_NOT_A_RESULT_fewshot.json", splits_out=sd / "splits.json")
    elif a.cmd == "extract":
        extract()
    else:
        analyse(out=HERE / f"fewshot{tag}_metrics.json", splits_out=HERE / f"fewshot{tag}_splits.json")
