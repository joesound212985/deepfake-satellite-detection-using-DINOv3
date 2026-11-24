#!/bin/bash
# Step 6 queue. Skips runs whose test_metrics.json exists. Appends to RUNLOG.md.
# Usage: ./run_all.sh [seeds...]   (default: 42)
cd ~/grsl
SEEDS=${@:-42}
ROWS=${ROWS:-"full plain noedge nobound nods frozen web"}   # subset of rows, e.g. ROWS="full plain"
RES=$HOME/results_dinov3/vit
SAT=dinov3-vitl16-pretrain-sat493m
WEB=dinov3-vitl16-pretrain-lvd1689m

run() {  # run <ckpt_dirname> <run_tag> <env...>
  local ckdir=$1 tag=$2; shift 2
  local row=${tag%_s*}
  case " $ROWS " in *" $row "*) ;; *) return;; esac
  local out=$RES/$ckdir/$tag
  if [ -f "$out/test_metrics.json" ]; then echo "[skip] $tag"; return; fi
  echo "| $(date -Is) | \`$* EVAL_SPLITS=val,test RUN_TAG=$tag python dinov3_forensics_train_v2.py\` | $tag | started |" >> RUNLOG.md
  env PYTHONUNBUFFERED=1 "$@" EVAL_SPLITS=val,test RUN_TAG=$tag python dinov3_forensics_train_v2.py > logs/$tag.log 2>&1
  local rc=$?
  local outcome="exit=$rc"
  [ -f "$out/test_metrics.json" ] && outcome="$outcome, test_metrics.json written" || outcome="$outcome, NO test_metrics.json"
  echo "| $(date -Is) | | $tag | $outcome |" >> RUNLOG.md
}

for S in $SEEDS; do
  run $SAT full_s$S     SEED=$S
  run $SAT plain_s$S    SEED=$S ABL_EDGE_GATE=0 ABL_BOUNDARY=0 ABL_DEEP_SUP=0
  run $SAT noedge_s$S   SEED=$S ABL_EDGE_GATE=0
  run $SAT nobound_s$S  SEED=$S ABL_BOUNDARY=0
  run $SAT nods_s$S     SEED=$S ABL_DEEP_SUP=0
  run $SAT frozen_s$S   SEED=$S FREEZE_BACKBONE=1
  run $WEB web_s$S      SEED=$S DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m
done
echo "[$(date -Is)] run_all.sh finished seeds=$SEEDS" >> RUNLOG.md
