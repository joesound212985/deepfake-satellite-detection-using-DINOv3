#!/usr/bin/env python3
"""
Build a fixed-seed evaluation subset of RSFAKE-1M (official test split only).

1. Sample N fake images per generator (seed) from SPLIT/RSFAKE_test_new.csv and
   extract exactly those members from FAKE/generated_*.zip.
2. Sample 10*N real patches (same seed) from the test split, download their parent
   fMoW-rgb test images from the public SpaceNet S3 bucket over HTTPS, and crop
   them with the same formula as the repo's REAL/fmow_process/crop.py
   (PIL open -> convert RGB -> crop(x, y, x+ps, y+ps) -> save JPEG, PIL defaults).
3. Write <root>/subset_seed<seed>_n<N>.csv  (image_path,label,gen)

Usage: python prepare_rsfake_subset.py --root ~/data/rsfake --per-gen 500 --seed 42
"""
import argparse, random, zipfile, io, sys, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

S3 = "https://spacenet-dataset.s3.amazonaws.com/Hosted-Datasets/fmow/fmow-rgb/"


def extract_fakes(root: Path, rels):
    by_gen = {}
    for rel in rels:
        gen = Path(rel).parts[1]
        by_gen.setdefault(gen, []).append(rel)
    for gen, lst in sorted(by_gen.items()):
        zp = root / "FAKE" / f"{gen}.zip"
        need = [rel for rel in lst if not (root / rel).exists()]
        print(f"[fake] {gen}: {len(lst)} chosen, {len(need)} to extract")
        if not need:
            continue
        with zipfile.ZipFile(zp) as z:
            names = set(z.namelist())
            for rel in need:
                member = str(Path(rel).relative_to("FAKE"))
                if member not in names:
                    print("  [warn] not in zip:", member)
                    continue
                dst = root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(z.read(member))


def parse_real(rel: str):
    # REAL/fmow/real/test/512/patch_<basename>_<x>_<y>_<ps>.jpg
    m = re.match(r"patch_(.+)_(\d+)_(\d+)_(\d+)\.jpg$", Path(rel).name)
    return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))


def fetch(url: str, dst: Path, tries=4):
    if dst.exists() and dst.stat().st_size > 0:
        return True
    for t in range(tries):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(r.content)
                return True
            if r.status_code == 404:
                return False
        except Exception:
            pass
        time.sleep(2 * (t + 1))
    return False


def build_reals(root: Path, rels, workers=8):
    meta = pd.read_csv(root / "REAL/fmow_process/fmow_test_patches.csv")
    meta["key"] = meta.basename + "_" + meta.x.astype(str) + "_" + meta.y.astype(str) + "_" + meta.patch_size.astype(str)
    meta = meta.set_index("key")
    jobs = []
    for rel in rels:
        b, x, y, ps = parse_real(rel)
        key = f"{b}_{x}_{y}_{ps}"
        if key not in meta.index:
            print("[warn] no metadata row for", rel)
            continue
        row = meta.loc[key]
        jobs.append((rel, row.image_path, x, y, ps))
    parents = sorted({j[1] for j in jobs})
    print(f"[real] {len(jobs)} patches from {len(parents)} parent fMoW test images")
    raw = root / "REAL/fmow_raw"

    def dl(p):
        return p, fetch(S3 + p.split("fmow/", 1)[1], raw / p)
    ok = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(dl, p) for p in parents]
        for i, f in enumerate(as_completed(futs)):
            p, good = f.result()
            ok += int(good)
            if (i + 1) % 200 == 0:
                print(f"[real] downloaded {i+1}/{len(parents)} (ok={ok})", flush=True)
    print(f"[real] downloads ok={ok}/{len(parents)}")

    written = 0
    cache = {}
    for rel, p, x, y, ps in jobs:
        src = raw / p
        dst = root / rel
        if dst.exists():
            written += 1
            continue
        if not src.exists():
            continue
        try:
            img = cache.get(p)
            if img is None:
                img = Image.open(src).convert("RGB")
                cache = {p: img}
            patch = img.crop((x, y, x + ps, y + ps))
            dst.parent.mkdir(parents=True, exist_ok=True)
            patch.save(dst)
            written += 1
        except Exception as e:
            print(f"[warn] crop failed {rel}: {e}")
    print(f"[real] patches written/present: {written}/{len(jobs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="~/data/rsfake")
    ap.add_argument("--per-gen", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-real", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    df = pd.read_csv(root / "SPLIT/RSFAKE_test_new.csv")
    df["gen"] = df.image_path.str.split("/").str[1]
    rng = random.Random(a.seed)

    fakes = []
    for g in sorted(df[df.label == 1].gen.unique()):
        rows = sorted(df[(df.label == 1) & (df.gen == g)].image_path.tolist())
        rng.shuffle(rows)
        fakes += [(r, 1, g) for r in rows[:a.per_gen]]
    reals_all = sorted(df[df.label == 0].image_path.tolist())
    rng.shuffle(reals_all)
    reals = [(r, 0, "real_fmow") for r in reals_all[:len(fakes)]]

    extract_fakes(root, [f[0] for f in fakes])
    if not a.no_real:
        build_reals(root, [r[0] for r in reals])

    sub = pd.DataFrame(fakes + reals, columns=["image_path", "label", "gen"])
    sub["exists"] = [(root / p).exists() for p in sub.image_path]
    out = root / f"subset_seed{a.seed}_n{a.per_gen}.csv"
    sub.to_csv(out, index=False)
    print(sub.groupby(["gen", "exists"]).size())
    print("[save]", out)


if __name__ == "__main__":
    main()
