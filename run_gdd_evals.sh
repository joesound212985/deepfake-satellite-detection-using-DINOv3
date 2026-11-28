#!/bin/bash
# Added 2026-11-28 (Joe: "yes second family, train on GDD"). Evaluation of the GDD-trained models gdd_s42-44
# (trained on the third machine, run_calypso_gdd.sh; checkpoints copied back first) in both directions:
# GDD's own tests and the benchmark test split plus the other external sets. Protocol: GDD_TRAIN_PROTOCOL.md.
# Evaluation only; existing outputs are skipped. Waits for other GPU jobs: one at a time.
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
while pgrep -f "dinov3_forensics_train_v[2].py|train_nor[m].py|eval_cross_datase[t].py|trufor_eva[l].py" >/dev/null; do sleep 60; done
R=$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m; D=$HOME/data
SAT=facebook/dinov3-vitl16-pretrain-sat493m
ev() {   # ev NAME CKPT args...
  local name=$1 ck=$2; shift 2
  [ -f "${name}_metrics.json" ] && return 0
  log "\`python eval_cross_dataset.py --dinov3 $SAT --ckpt $ck --out ${name}_metrics.json $*\`" "$name" "started"
  python eval_cross_dataset.py --dinov3 "$SAT" --ckpt "$ck" --out "${name}_metrics.json" "$@" > "logs/$name.log" 2>&1
  local rc=$?; [ -f "${name}_metrics.json" ] && log "" "$name" "exit=$rc, ${name}_metrics.json written" || log "" "$name" "exit=$rc, FAILED (logs/$name.log)"
}
for S in ${SEEDS:-42 43 44}; do
  ck=$R/gdd_s$S/best_checkpoint.pt; p=cross_gddtrain_s${S}_
  [ -f "$ck" ] && [ -f "$R/gdd_s$S/test_metrics.json" ] || { echo "gdd_s$S not here yet"; continue; }
  for split in indomain crossgenerator crossdomain localization; do
    ev ${p}gdd_$split $ck --dataset gdd --gdd-split $split --root $D/gdd
  done
  ev ${p}overhead $ck --dataset overhead --root $D/overhead/test_jpg/test
  ev ${p}fmow $ck --dataset fmow --root $D/fmow-fake-small
  ev ${p}rsfake $ck --dataset rsfake --root $D/rsfake --subset-csv $D/rsfake/subset_seed42_n500.csv --per-gen 500
  ev ${p}uw $ck --dataset uw --root "$D/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
  ev ${p}dmaer $ck --dataset dmaer --root $D/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched
  ev ${p}allreals_dmaer $ck --dataset dmaer --root $D/dm_aer/DM-AER-DeepFake-V1-sample
done
