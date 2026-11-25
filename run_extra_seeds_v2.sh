#!/bin/bash
# Replaces run_extra_seeds.sh (2026-11-25). Waits for the seed-42 queue, then, while the GPU is free:
#   1. scores full_s42 on DM-AER (official sample: 1000 StyleGAN2 fakes + 1000 Million-AID reals),
#   2. scores web_s42 on fmow / rsfake / uw / dmaer (same subsets as full_s42),
#   3. re-runs the SimLBR baseline (+ no-LBR control) with DM-AER added, to new output files,
#   4. trains seeds 43 and 44 for full + plain.
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
xeval cross_     $SATREPO "$FULL" dmaer  --root "$DM"
xeval cross_web_ $WEBREPO "$WEB"  fmow   --root "$D/fmow-fake-small"
xeval cross_web_ $WEBREPO "$WEB"  rsfake --root "$D/rsfake" --subset-csv "$D/rsfake/subset_seed42_n500.csv" --per-gen 500
xeval cross_web_ $WEBREPO "$WEB"  uw     --root "$UW"
xeval cross_web_ $WEBREPO "$WEB"  dmaer  --root "$DM"
if [ ! -f simlbr_dmaer_metrics.json ]; then
  log "\`python simlbr_baseline.py --out simlbr_dmaer_metrics.json\`" "simlbr_dmaer" "started"
  python simlbr_baseline.py --out simlbr_dmaer_metrics.json > logs/simlbr_dmaer.log 2>&1; log "" "simlbr_dmaer" "exit=$?"
fi
if [ ! -f simlbr_nolbr_dmaer_metrics.json ]; then
  log "\`python simlbr_baseline.py --no-lbr --out simlbr_nolbr_dmaer_metrics.json\`" "simlbr_nolbr_dmaer" "started"
  python simlbr_baseline.py --no-lbr --out simlbr_nolbr_dmaer_metrics.json > logs/simlbr_nolbr_dmaer.log 2>&1; log "" "simlbr_nolbr_dmaer" "exit=$?"
fi
SKIP_SETUP=1 SEEDS="43 44" ROWS="full plain" ./run_everything.sh > logs/run_extra_seeds.out 2>&1
