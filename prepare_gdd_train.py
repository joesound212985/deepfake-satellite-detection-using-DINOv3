#!/usr/bin/env python3
"""Writes GDD's training set and in-domain test set in the layout dinov3_forensics_train_v2.py reads
(GDD_TRAIN_PROTOCOL.md). Nothing is copied or converted: <out>/{train,test}/gdd is a link to the GDD folder and
metadata.csv lists paths below it.

  train = train_detection.csv + train_localization.csv (union)
  test  = test_indomain_detection.csv + test_indomain_localization.csv (union)

Class strings are the ones the training script maps: pristine (real), generated (fully synthesized, no mask),
partial (any manipulation type, with GDD's mask).  --check reads every image and mask the way the training script
does (cv2) and stops if one cannot be read.
"""
import argparse, collections, csv, os, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gdd", default=str(Path.home() / "data/gdd"))
ap.add_argument("--out", default=str(Path.home() / "data/gdd_layout"))
ap.add_argument("--check", action="store_true")
args = ap.parse_args()
G, OUT = Path(args.gdd), Path(args.out)

SPLITS = {"train": ["train_detection.csv", "train_localization.csv"],
          "test": ["test_indomain_detection.csv", "test_indomain_localization.csv"]}
bad = []
for split, files in SPLITS.items():
    rows, seen, stems = [], set(), collections.Counter()
    for f in files:
        with open(G / f, newline="") as fh:
            for r in csv.DictReader(fh):
                p = r["image_filepath"].strip()
                fake = r["label"].strip().lower() == "fake"
                man = (r.get("manipulation_type") or "").strip()
                cls = "pristine" if not fake else ("partial" if man else "generated")
                mask = (r.get("mask_filepath") or "").strip() if cls == "partial" else ""
                if p in seen:
                    prev = next(x for x in rows if x[0] == p)
                    assert prev[2] == cls, f"{p}: listed with two classes"
                    continue
                if cls == "partial" and not mask:
                    sys.exit(f"{p}: manipulated image without a mask path")
                seen.add(p); stems[Path(p).stem] += 1
                rows.append((p, mask, cls, man or (r.get("generator") or "").strip() or (r.get("source_dataset") or "").strip()))
    dup = [s for s, n in stems.items() if n > 1]
    if dup:   # the training script identifies images by file stem for its validation hold-out
        sys.exit(f"{split}: {len(dup)} file stems are not unique, e.g. {dup[0]}")
    missing = [p for p, m, _, _ in rows if not (G / p).exists() or (m and not (G / m).exists())]
    if missing:
        sys.exit(f"{split}: {len(missing)} of {len(rows)} files missing, e.g. {missing[0]}")
    d = OUT / split
    d.mkdir(parents=True, exist_ok=True)
    link = d / "gdd"
    if not link.exists():
        os.symlink(G, link)
    with open(d / "metadata.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_filename", "mask_filename", "manipulation_type", "gdd_group"])
        for p, m, cls, grp in rows:
            w.writerow([f"gdd/{p}", f"gdd/{m}" if m else "", cls, grp])
    c = collections.Counter(cls for _, _, cls, _ in rows)
    print(f"{split}: {len(rows)} images  {dict(sorted(c.items()))}  -> {d / 'metadata.csv'}")
    if args.check:
        import cv2
        empty = 0
        for p, m, cls, _ in rows:
            img = cv2.imread(str(G / p))
            if img is None or img.ndim != 3 or img.shape[2] != 3:
                bad.append(p); continue
            if m:
                mk = cv2.imread(str(G / m), cv2.IMREAD_GRAYSCALE)
                if mk is None:
                    bad.append(m); continue
                if mk.shape[:2] != img.shape[:2]:
                    bad.append(m + " (size differs from image)")
                if cv2.resize(mk, (384, 384), interpolation=cv2.INTER_NEAREST).sum() <= 15:
                    empty += 1
        print(f"{split}: read check done, unreadable or mismatched: {len(bad)}, manipulated images whose mask is empty at 384 px: {empty}")
if bad:
    sys.exit("could not read: " + ", ".join(bad[:10]))
