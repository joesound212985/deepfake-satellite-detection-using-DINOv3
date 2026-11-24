#!/bin/bash
# ============================================================================
# Fetch / verify EVERY dataset and model used by the GRSL experiments.
# Idempotent: anything already present is skipped.  Called automatically by
# run_everything.sh (set SKIP_SETUP=1 to bypass).   Standalone:  ./setup_data.sh
#
# Layout (all under $DATA, default ~/data):
#   overhead/{train_jpg/train,test_jpg/test}   May et al. 2023 (Google-Form gated -> copied, see below)
#   rsfake/                                     RSFAKE-1M fakes + split csvs + rebuilt fMoW reals (subset)
#   fmow-fake-small/                            geodf/fmow-fake-small (HF, approval required)
#   uw_fake_sat/                                UW Fake Satellite Imagery (figshare)
# Models are cached by Hugging Face / timm in ~/.cache.
#
# The main dataset cannot be downloaded by script (authors' Google Form). Provide it as ONE of:
#   OVERHEAD_TAR=/path/overhead.tar      (made by ./make_bundle.sh on the original machine)
#   OVERHEAD_SRC=user@host:~/data/overhead   (rsync source)
#   or a Desktop copy at /mnt/c/Users/*/Desktop/{train,test}/metadata.csv
# ============================================================================
set -u
cd "$(dirname "$0")"
DATA=${DATA:-$HOME/data}
PER_GEN=${PER_GEN:-500}
mkdir -p "$DATA" logs
ok()   { echo "[setup]  ok   $*"; }
todo() { echo "[setup]  ...  $*"; }
warn() { echo "[setup]  WARN $*"; }

# ---------------------------------------------------------------- 1. main dataset
OV=$DATA/overhead
if [ -f "$OV/train_jpg/train/metadata.csv" ] && [ -f "$OV/test_jpg/test/metadata.csv" ]; then
  ok "overhead dataset: $(ls "$OV/train_jpg/train" | wc -l) train files, $(ls "$OV/test_jpg/test" | wc -l) test files"
else
  todo "overhead dataset missing -> trying OVERHEAD_TAR / OVERHEAD_SRC / Windows Desktop"
  if [ -n "${OVERHEAD_TAR:-}" ] && [ -f "$OVERHEAD_TAR" ]; then
    mkdir -p "$OV" && tar -xf "$OVERHEAD_TAR" -C "$OV" && ok "extracted $OVERHEAD_TAR"
  elif [ -n "${OVERHEAD_SRC:-}" ]; then
    mkdir -p "$OV" && rsync -a "$OVERHEAD_SRC/" "$OV/" && ok "rsynced from $OVERHEAD_SRC"
  else
    for d in /mnt/c/Users/*/Desktop; do
      if [ -f "$d/train/metadata.csv" ] && [ -f "$d/test/metadata.csv" ]; then
        mkdir -p "$OV/train_jpg" "$OV/test_jpg"
        cp -r "$d/train" "$OV/train_jpg/" && cp -r "$d/test" "$OV/test_jpg/" && ok "copied from $d"
        break
      fi
    done
  fi
  [ -f "$OV/train_jpg/train/metadata.csv" ] || { warn "overhead dataset NOT available. Fill the authors' form at"
    warn "https://stresearch.github.io/synthetic-overhead-dataset or bring overhead.tar from make_bundle.sh"; }
fi

# ---------------------------------------------------------------- 2. HF login + models
if hf auth whoami >/dev/null 2>&1; then ok "Hugging Face login: $(hf auth whoami 2>/dev/null | grep -o 'user=.*' | head -1)"
else warn "not logged in to Hugging Face: run  hf auth login   (gated DINOv3 repos need it)"; fi
for repo in facebook/dinov3-vitl16-pretrain-sat493m facebook/dinov3-vitl16-pretrain-lvd1689m nvidia/C-RADIOv4-SO400M; do
  if hf download "$repo" --quiet >/dev/null 2>logs/setup_$(basename $repo).log; then ok "model cached: $repo"
  else warn "could not download $repo (accept its licence on huggingface.co, then rerun): $(tail -1 logs/setup_$(basename $repo).log)"; fi
done
python - <<'EOF' 2>/dev/null && echo "[setup]  ok   model cached: timm resnet50 (ImageNet)" || echo "[setup]  WARN timm resnet50 weights not downloadable"
import timm; timm.create_model("resnet50", pretrained=True, features_only=True, out_indices=(3,4))
EOF

# ---------------------------------------------------------------- 3. RSFAKE-1M
RS=$DATA/rsfake
if [ -f "$RS/subset_seed42_n$PER_GEN.csv" ] && python - "$RS/subset_seed42_n$PER_GEN.csv" "$RS" <<'EOF'
import sys, pandas as pd, pathlib
df = pd.read_csv(sys.argv[1]); root = pathlib.Path(sys.argv[2])
missing = sum(not (root / p).exists() for p in df.image_path)
print(f"[setup]  ok   RSFAKE subset: {len(df)} rows, {missing} missing files" if missing == 0 else f"[setup]  ...  RSFAKE subset has {missing} missing files -> rebuild")
sys.exit(0 if missing == 0 else 1)
EOF
then :; elif [ -n "${RSFAKE_TAR:-}" ] && [ -f "$RSFAKE_TAR" ]; then
  mkdir -p "$RS" && tar -xf "$RSFAKE_TAR" -C "$RS" && ok "RSFAKE subset extracted from $RSFAKE_TAR"
else
  todo "RSFAKE-1M: downloading split csvs + all fake zips (27.6 GB) + fMoW crop metadata"
  hf download TZHSW/RSFAKE --repo-type dataset --include "SPLIT/*" --include "FAKE/*" --include "REAL/*" --local-dir "$RS" > logs/setup_rsfake_download.log 2>&1 \
    && ok "RSFAKE files downloaded" || warn "RSFAKE download failed (accept conditions at huggingface.co/datasets/TZHSW/RSFAKE): $(tail -1 logs/setup_rsfake_download.log)"
  if [ -f "$RS/SPLIT/RSFAKE_test_new.csv" ]; then
    todo "RSFAKE-1M: extracting $PER_GEN fakes/generator and rebuilding $((PER_GEN*10)) real fMoW patches (public S3)"
    python prepare_rsfake_subset.py --root "$RS" --per-gen $PER_GEN --seed 42 > logs/prepare_rsfake.log 2>&1 \
      && ok "RSFAKE subset built: $RS/subset_seed42_n$PER_GEN.csv" || warn "RSFAKE subset build failed, see logs/prepare_rsfake.log"
  fi
fi

# ---------------------------------------------------------------- 4. fmow-fake-small (approval-gated)
FM=$DATA/fmow-fake-small
if [ -d "$FM/train" ] || [ -d "$FM/val" ]; then ok "fmow-fake-small: $(find "$FM" -name '*_img.tif' | wc -l) images"
else
  hf download geodf/fmow-fake-small --repo-type dataset --local-dir "$FM" > logs/setup_fmow.log 2>&1
  for z in "$FM"/train.zip "$FM"/val.zip; do [ -f "$z" ] && unzip -q -o "$z" -d "$FM"; done
  if [ -d "$FM/train" ] || [ -d "$FM/val" ]; then ok "fmow-fake-small downloaded"
  else warn "fmow-fake-small not accessible yet (request access at huggingface.co/datasets/geodf/fmow-fake-small): $(grep -m1 -i 'denied\|error' logs/setup_fmow.log)"; fi
fi

# ---------------------------------------------------------------- 5. UW Fake Satellite Imagery (figshare 12197655)
UW=$DATA/uw_fake_sat
UWD="$UW/anti-deepfake/anti-deepfake-data and code/data"
if [ -d "$UWD/authentic" ] && [ -d "$UWD/fake" ]; then ok "UW fake satellite: $(find "$UWD/authentic" "$UWD/fake" -type f | wc -l) images"
else
  todo "UW fake satellite: downloading 1.9 GB from figshare"
  mkdir -p "$UW" && cd "$UW" \
    && curl -sL -o deep-fake-in-geography.zip https://ndownloader.figshare.com/files/22424628 \
    && curl -sL -o anti-deepfake-data-and-code.zip https://ndownloader.figshare.com/files/28889103 \
    && unzip -q -o deep-fake-in-geography.zip -d deep-fake-in-geography \
    && unzip -q -o anti-deepfake-data-and-code.zip -d anti-deepfake \
    && ok "UW fake satellite downloaded" || warn "UW download failed"
  cd "$OLDPWD"
fi

echo "[setup] done.  DATA=$DATA"
