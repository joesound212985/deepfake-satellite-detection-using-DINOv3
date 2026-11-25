#!/bin/bash
# Side experiment, runs LAST (after every main row): input-normalisation check, seed 42.
# Same script and settings as full_s42 / web_s42, but the encoder input is normalised with the
# backbone's own image_mean / image_std (train_norm.py). Rows are named fullnorm / webnorm so they
# can never be averaged into the main rows. Kill this script to cancel.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while pgrep -f "run_everything.sh|run_extra_seeds|run_web_seeds" >/dev/null; do sleep 120; done
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
export TRAIN_ROOT=${TRAIN_ROOT:-$HOME/data/overhead/train_jpg/train}
export TEST_ROOT=${TEST_ROOT:-$HOME/data/overhead/test_jpg/test}
export PYTHONUNBUFFERED=1
R=$HOME/results_dinov3/vit
nrun() {  # nrun <results subdir> <tag> <env...>
  local out=$R/$1/$2 tag=$2; shift 2
  [ -f "$out/test_metrics.json" ] && return
  log "\`$* EVAL_SPLITS=val,test RUN_TAG=$tag python train_norm.py\`" "$tag" "started"
  env "$@" EVAL_SPLITS=val,test RUN_TAG=$tag python train_norm.py > logs/$tag.log 2>&1
  local rc=$?
  [ -f "$out/test_metrics.json" ] && log "" "$tag" "exit=$rc, test_metrics.json written" || log "" "$tag" "exit=$rc, NO test_metrics.json (see logs/$tag.log)"
}
nrun dinov3-vitl16-pretrain-sat493m  fullnorm_s42 SEED=42
nrun dinov3-vitl16-pretrain-lvd1689m webnorm_s42  SEED=42 DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m
python aggregate_runs.py --csv table.csv > logs/aggregate.log 2>&1 && log "\`python aggregate_runs.py --csv table.csv\`" "aggregate" "table.csv rebuilt with fullnorm/webnorm rows"
python analyze_extras.py > logs/analyze_extras.log 2>&1
echo "[$(date -Is)] run_norm_check.sh finished" >> RUNLOG.md
