#!/bin/bash
# Third machine, added 2026-11-28 (Joe: "yes second family, train on GDD"). Trains the unchanged full model on GDD's
# training set (GDD_TRAIN_PROTOCOL.md): existing switches only, TRAIN_ROOT / TEST_ROOT pointed at the layout written
# by prepare_gdd_train.py.  GPU=1 SEEDS="42 43" after the frozen lane, GPU=0 SEEDS="44" after the radio lane.
# Waits until the earlier lane of that GPU (tmux session $AFTER) has ended: one job of ours per GPU.
# Everything stays inside ~/joe_grsl (HOME is pointed there). Finished runs are skipped.
# Different GPU from the workstation: use accuracy numbers, not timing.
GPU=${GPU:-1}; AFTER=${AFTER:-joe_grsl_review}
REAL=$HOME
source "$REAL/miniconda3/etc/profile.d/conda.sh"; conda activate joe_grsl
export HOME=$REAL/joe_grsl/home PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU
cd "$HOME/grsl"; mkdir -p logs
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
while tmux has-session -t "$AFTER" 2>/dev/null; do sleep 60; done
L=$HOME/data/gdd_layout; R=$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m
[ -f "$L/train/metadata.csv" ] && [ -f "$L/test/metadata.csv" ] || { log "" "calypso gdd (GPU $GPU)" "NOT started: $L is missing"; exit 1; }
log "\`GPU=$GPU SEEDS=\"$SEEDS\" ./run_calypso_gdd.sh\` on $(hostname)" "calypso gdd (GPU $GPU)" "started"
for S in ${SEEDS:-42 43}; do
  O=$R/gdd_s$S
  [ -f "$O/test_metrics.json" ] && continue
  log "\`SEED=$S TRAIN_ROOT=$L/train TEST_ROOT=$L/test EVAL_SPLITS=val,test RUN_TAG=gdd_s$S python dinov3_forensics_train_v2.py\`" "gdd_s$S" "started"
  SEED=$S TRAIN_ROOT=$L/train TEST_ROOT=$L/test EVAL_SPLITS=val,test RUN_TAG=gdd_s$S python dinov3_forensics_train_v2.py > logs/gdd_s$S.log 2>&1
  rc=$?
  [ -f "$O/test_metrics.json" ] && log "" "gdd_s$S" "exit=$rc, test_metrics.json written" || log "" "gdd_s$S" "exit=$rc, NO test_metrics.json (see logs/gdd_s$S.log)"
done
log "" "calypso gdd (GPU $GPU)" "finished"
