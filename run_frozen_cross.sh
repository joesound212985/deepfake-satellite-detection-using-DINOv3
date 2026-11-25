#!/bin/bash
# Added 2026-11-25: zero-shot scoring of the existing frozen_s42 checkpoint on the four external sets
# (same subsets and code as full_s42). Evaluation only, no training. Existing outputs are skipped.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while [ ! -f overhead_full_metrics.json ] && pgrep -f "out overhead_full_metrics.json" >/dev/null; do sleep 20; done
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
D=${DATA:-$HOME/data}
REPO=facebook/dinov3-vitl16-pretrain-sat493m
CK=$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/frozen_s42/best_checkpoint.pt
xeval() {
  local name=$1; shift
  local out=cross_frozen_${name}_metrics.json
  [ -f "$out" ] && return
  log "\`python eval_cross_dataset.py --dataset $name --dinov3 $REPO --ckpt frozen_s42/best_checkpoint.pt --out $out $*\`" "cross_frozen_$name" "started"
  python eval_cross_dataset.py --dataset $name --dinov3 $REPO --ckpt "$CK" --out "$out" "$@" > logs/cross_frozen_$name.log 2>&1
  local rc=$?
  [ -f "$out" ] && log "" "cross_frozen_$name" "exit=$rc, $out written" || log "" "cross_frozen_$name" "exit=$rc, FAILED (logs/cross_frozen_$name.log)"
}
xeval fmow   --root "$D/fmow-fake-small"
xeval rsfake --root "$D/rsfake" --subset-csv "$D/rsfake/subset_seed42_n500.csv" --per-gen 500
xeval uw     --root "$D/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
xeval dmaer  --root "$D/dm_aer/DM-AER-DeepFake-V1-sample" --dmaer-matched
