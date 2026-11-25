#!/bin/bash
# Added 2026-11-25 at Joe's request: web-pretrained DINOv3 was the best seed-42 row.
# Waits for the seed-42 queue and the full/plain extra seeds, then
#   1. trains web seeds 43 and 44 if still missing (since the 2026-11-25 reorder they normally run
#      earlier, in run_extra_seeds_v2.sh, and are skipped here),
#   2. scores the web_s42 checkpoint zero-shot on the three external sets
#      (same subsets as the full_s42 cross-dataset evaluation),
#   3. re-scores the remaining full / plain / web checkpoints on the test set (per-image predictions)
#      and runs the McNemar significance tests (analyze_extras.py),
#   4. rebuilds table.csv.
# Kill this script to cancel. Never overwrites: existing outputs are skipped.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while pgrep -f "run_everything.sh|run_extra_seeds" >/dev/null; do sleep 120; done
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }

SKIP_SETUP=1 SEEDS="43 44" ROWS="web" ./run_everything.sh > logs/run_web_seeds.out 2>&1

WEBREPO=facebook/dinov3-vitl16-pretrain-lvd1689m
CK=$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt
D=${DATA:-$HOME/data}
xeval() {
  local name=$1; shift
  local out=cross_web_${name}_metrics.json
  [ -f "$out" ] && return
  log "\`python eval_cross_dataset.py --dataset $name --dinov3 $WEBREPO --ckpt web_s42 $*\`" "cross_web_$name" "started"
  python eval_cross_dataset.py --dataset $name --dinov3 $WEBREPO --ckpt "$CK" --out "$out" "$@" > logs/cross_web_$name.log 2>&1
  local rc=$?
  [ -f "$out" ] && log "" "cross_web_$name" "exit=$rc, $out written" || log "" "cross_web_$name" "exit=$rc, FAILED (logs/cross_web_$name.log)"
}
if [ -f "$CK" ]; then
  xeval fmow --root "$D/fmow-fake-small"
  xeval rsfake --root "$D/rsfake" --subset-csv "$D/rsfake/subset_seed42_n500.csv" --per-gen 500
  xeval uw --root "$D/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
fi
# ---- significance tests need per-image test predictions for every full / plain / web seed.
# full_s42 and web_s42 are done in run_extra_seeds_v2.sh; the other seven checkpoints are re-scored here.
SATREPO=facebook/dinov3-vitl16-pretrain-sat493m
R=$HOME/results_dinov3/vit
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
oeval overhead_plain_s42 $SATREPO $R/dinov3-vitl16-pretrain-sat493m/plain_s42/best_checkpoint.pt --no-edge-gate
for S in 43 44; do
  oeval overhead_full_s$S  $SATREPO $R/dinov3-vitl16-pretrain-sat493m/full_s$S/best_checkpoint.pt
  oeval overhead_plain_s$S $SATREPO $R/dinov3-vitl16-pretrain-sat493m/plain_s$S/best_checkpoint.pt --no-edge-gate
  oeval overhead_web_s$S   $WEBREPO $R/dinov3-vitl16-pretrain-lvd1689m/web_s$S/best_checkpoint.pt
done
python analyze_extras.py > logs/analyze_extras.log 2>&1; log "\`python analyze_extras.py\`" "analysis_extras" "exit=$? (significance tests added)"
python aggregate_runs.py --csv table.csv > logs/aggregate.log 2>&1 && log "\`python aggregate_runs.py --csv table.csv\`" "aggregate" "table.csv rebuilt with web seeds"
