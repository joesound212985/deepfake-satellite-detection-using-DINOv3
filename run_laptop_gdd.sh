#!/bin/bash
# Second machine (laptop), added 2026-11-28 (Joe turned it on for the GDD seeds). Trains the unchanged full model
# on GDD's training set (GDD_TRAIN_PROTOCOL.md): existing switches only, TRAIN_ROOT / TEST_ROOT pointed at the
# layout written by prepare_gdd_train.py. Default seeds 43 and 44 (seed 42 runs on the third machine).
# Finished runs are skipped. Different GPU from the workstation: use accuracy numbers, not timing.
source ~/Desktop/.venv/bin/activate
cd "$(dirname "$0")"; mkdir -p logs
export PYTHONUNBUFFERED=1
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
L=$HOME/data/gdd_layout; R=$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m
[ -f "$L/train/metadata.csv" ] && [ -f "$L/test/metadata.csv" ] || { log "" "laptop gdd" "NOT started: $L is missing"; exit 1; }
log "\`SEEDS=\"${SEEDS:-43 44}\" ./run_laptop_gdd.sh\` on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1))" "laptop gdd" "started"
for S in ${SEEDS:-43 44}; do
  O=$R/gdd_s$S
  [ -f "$O/test_metrics.json" ] && continue
  log "\`SEED=$S TRAIN_ROOT=$L/train TEST_ROOT=$L/test EVAL_SPLITS=val,test RUN_TAG=gdd_s$S python dinov3_forensics_train_v2.py\`" "gdd_s$S" "started"
  SEED=$S TRAIN_ROOT=$L/train TEST_ROOT=$L/test EVAL_SPLITS=val,test RUN_TAG=gdd_s$S python dinov3_forensics_train_v2.py > logs/gdd_s$S.log 2>&1
  rc=$?
  [ -f "$O/test_metrics.json" ] && log "" "gdd_s$S" "exit=$rc, test_metrics.json written" || log "" "gdd_s$S" "exit=$rc, NO test_metrics.json (see logs/gdd_s$S.log)"
done
log "" "laptop gdd" "finished"
