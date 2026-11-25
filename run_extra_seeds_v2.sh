#!/bin/bash
# Replaces run_extra_seeds.sh (2026-11-25). Waits for the seed-42 queue, then, while the GPU is free:
#   1. scores full_s42 on DM-AER (official sample; primary = size-matched: 1000 StyleGAN2 fakes + the 494
#      Million-AID reals >= 512 px, all centre-cropped to 512; plus an all-reals run as a sensitivity check),
#   2. scores web_s42 on fmow / rsfake / uw / dmaer (same subsets as full_s42),
#   3. re-runs the SimLBR baseline (+ no-LBR control) with DM-AER added, to new output files,
#   4. extra analyses: test set two-class scoring, JPEG / down-scaling robustness, qualitative figure,
#      analysis_extras.json,
#   5. SimLBR with the authors' code; resnetlr_s42 (ResNet-50 with backbone LR 1e-4),
#   6. trains seeds 43 and 44: web first, then full (plain seeds 43/44 run on the laptop, run_laptop.sh).
# Existing outputs are skipped, never overwritten. Kill this script to cancel.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while pgrep -f "run_everything.sh" >/dev/null; do sleep 120; done
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
D=${DATA:-$HOME/data}
R=$HOME/results_dinov3/vit
xeval() {  # xeval <outprefix> <dinov3 repo> <ckpt> <dataset> <args...>
  local pre=$1 repo=$2 ck=$3 name=$4; shift 4
  local out=${pre}${name}_metrics.json
  [ -f "$out" ] && return
  [ -f "$ck" ] || { log "" "${pre}${name}" "SKIPPED: checkpoint missing"; return; }
  log "\`python eval_cross_dataset.py --dataset $name --dinov3 $repo --ckpt $ck --out $out $*\`" "${pre}${name}" "started"
  python eval_cross_dataset.py --dataset $name --dinov3 $repo --ckpt "$ck" --out "$out" "$@" > logs/${pre}${name}.log 2>&1
  local rc=$?
  [ -f "$out" ] && log "" "${pre}${name}" "exit=$rc, $out written" || log "" "${pre}${name}" "exit=$rc, FAILED (logs/${pre}${name}.log)"
}
SATREPO=facebook/dinov3-vitl16-pretrain-sat493m
WEBREPO=facebook/dinov3-vitl16-pretrain-lvd1689m
FULL=$R/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt
WEB=$R/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt
DM="$D/dm_aer/DM-AER-DeepFake-V1-sample"
UW="$D/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
xeval cross_     $SATREPO "$FULL" dmaer  --root "$DM" --dmaer-matched          # primary: size-matched reals
xeval cross_allreals_ $SATREPO "$FULL" dmaer --root "$DM"                        # sensitivity: all reals, native size
xeval cross_web_ $WEBREPO "$WEB"  fmow   --root "$D/fmow-fake-small"
xeval cross_web_ $WEBREPO "$WEB"  rsfake --root "$D/rsfake" --subset-csv "$D/rsfake/subset_seed42_n500.csv" --per-gen 500
xeval cross_web_ $WEBREPO "$WEB"  uw     --root "$UW"
xeval cross_web_ $WEBREPO "$WEB"  dmaer  --root "$DM" --dmaer-matched
xeval cross_web_allreals_ $WEBREPO "$WEB" dmaer --root "$DM"
if [ ! -f simlbr_dmaer_metrics.json ]; then
  log "\`python simlbr_baseline.py --out simlbr_dmaer_metrics.json\`" "simlbr_dmaer" "started"
  python simlbr_baseline.py --out simlbr_dmaer_metrics.json > logs/simlbr_dmaer.log 2>&1; log "" "simlbr_dmaer" "exit=$?"
fi
if [ ! -f simlbr_nolbr_dmaer_metrics.json ]; then
  log "\`python simlbr_baseline.py --no-lbr --out simlbr_nolbr_dmaer_metrics.json\`" "simlbr_nolbr_dmaer" "started"
  python simlbr_baseline.py --no-lbr --out simlbr_nolbr_dmaer_metrics.json > logs/simlbr_nolbr_dmaer.log 2>&1; log "" "simlbr_nolbr_dmaer" "exit=$?"
fi
# ---- extra analyses on the two seed-42 checkpoints (added 2026-11-25): test set re-scored with the
# cross-dataset code (two-class numbers next to SimLBR), robustness to JPEG re-compression and
# down-scaling, qualitative figure, then analyze_extras.py (per-generator RSFAKE, bootstrap CIs).
oeval() {  # oeval <outstem> <dinov3 repo> <ckpt> <extra args...>
  local stem=$1 repo=$2 ck=$3; shift 3
  local out=${stem}_metrics.json
  [ -f "$out" ] && return
  [ -f "$ck" ] || { log "" "$stem" "SKIPPED: checkpoint missing"; return; }
  log "\`python eval_cross_dataset.py --dataset overhead --dinov3 $repo --ckpt $ck --out $out $*\`" "$stem" "started"
  python eval_cross_dataset.py --dataset overhead --root "$D/overhead/test_jpg/test" --dinov3 $repo --ckpt "$ck" --out "$out" "$@" > logs/$stem.log 2>&1
  local rc=$?
  [ -f "$out" ] && log "" "$stem" "exit=$rc, $out written" || log "" "$stem" "exit=$rc, FAILED (logs/$stem.log)"
}
for n in full web; do
  if [ $n = full ]; then repo=$SATREPO; ck=$FULL; else repo=$WEBREPO; ck=$WEB; fi
  oeval overhead_$n $repo "$ck"
  for dg in jpeg:90 jpeg:70 jpeg:50 jpeg:30 scale:0.75 scale:0.5; do
    oeval robust_${n}_${dg/:/} $repo "$ck" --degrade $dg
  done
done
if [ ! -f qualitative.png ] && [ -f overhead_full_metrics.per_image.json ]; then
  python make_qualitative.py > logs/make_qualitative.log 2>&1; log "\`python make_qualitative.py\`" "qualitative" "exit=$?"
fi
# ---- SimLBR with the authors' released code (github.com/mvrl/SimLBR @ ac90f86), both backbones, 3 seeds, LBR + no-LBR
for bb in sat web; do
  if [ ! -f simlbr_official_${bb}_metrics.json ]; then
    log "\`python simlbr_official.py --backbone $bb\`" "simlbr_official_$bb" "started"
    python simlbr_official.py --backbone $bb > logs/simlbr_official_$bb.log 2>&1; log "" "simlbr_official_$bb" "exit=$?"
  fi
done
python analyze_extras.py > logs/analyze_extras.log 2>&1; log "\`python analyze_extras.py\`" "analysis_extras" "exit=$? (analysis_extras.json)"

# ---- ResNet-50 with a conventional backbone learning rate (added 2026-11-25, Joe approved).
# The resnet row uses the DINOv3 rates (backbone 2e-6), far below normal for a CNN, and was stuck near
# 66% validation accuracy at epoch 18. This row changes ONLY LR_BACKBONE to 1e-4, fixed in advance
# (not tuned on results); the decoder rate, schedule and everything else are identical. Reported as a
# separate row (resnetlr) next to the original resnet row.
RL=$HOME/results_dinov3/resnet/resnet50/resnetlr_s42
if [ ! -f "$RL/test_metrics.json" ]; then
  export TRAIN_ROOT=${TRAIN_ROOT:-$D/overhead/train_jpg/train} TEST_ROOT=${TEST_ROOT:-$D/overhead/test_jpg/test} PYTHONUNBUFFERED=1
  log "\`SEED=42 BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s42 python dinov3_forensics_train_v2.py\`" "resnetlr_s42" "started"
  SEED=42 BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s42 python dinov3_forensics_train_v2.py > logs/resnetlr_s42.log 2>&1
  rc=$?
  [ -f "$RL/test_metrics.json" ] && log "" "resnetlr_s42" "exit=$rc, test_metrics.json written" || log "" "resnetlr_s42" "exit=$rc, NO test_metrics.json (see logs/resnetlr_s42.log)"
fi

# Order chosen 2026-11-25 so the most important rows finish first (same runs, same settings, only
# the order differs): web (best seed-42 row), then full, then plain. Each call skips finished rows.
# 2026-11-25 (Joe's request): plain seeds 43/44 moved to the laptop (run_laptop.sh), so this machine
# trains only web and full here. EXTRA_ROWS="web full plain" restores the old behaviour.
for row in ${EXTRA_ROWS:-web full}; do
  SKIP_SETUP=1 SEEDS="43 44" ROWS="$row" ./run_everything.sh >> logs/run_extra_seeds.out 2>&1
done
