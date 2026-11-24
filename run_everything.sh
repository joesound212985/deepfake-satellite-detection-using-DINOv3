#!/bin/bash
# ============================================================================
# ONE SCRIPT FOR ALL GRSL EXPERIMENTS.  Idempotent: finished steps are skipped.
#
#   cd ~/grsl && nohup ./run_everything.sh > logs/run_everything.out 2>&1 &
#
# Optional:  SEEDS="42 43 44" ./run_everything.sh      (default "42")
#            ROWS="full plain"  ./run_everything.sh     (default: all 7 rows)
#            PER_GEN=500 (RSFAKE fakes per generator, default 500)
#
# Order:
#   1. training rows for each seed  (full, plain, noedge, nobound, nods, frozen, web, resnet, radio)
#   2. cross-dataset zero-shot eval of the full-model seed-42 checkpoint
#      (fmow-fake-small if downloadable, RSFAKE-1M subset, UW fake satellite)
#   3. aggregate_runs.py -> table.csv
# Everything is appended to RUNLOG.md; per-run stdout goes to logs/<tag>.log.
# ============================================================================
set -u
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONUNBUFFERED=1

# ---- 0. make sure every dataset / model is present (idempotent; SKIP_SETUP=1 to bypass)
if [ "${SKIP_SETUP:-0}" != "1" ]; then
  ./setup_data.sh 2>&1 | tee logs/setup_data.log
  [ -f "${DATA:-$HOME/data}/overhead/train_jpg/train/metadata.csv" ] || { echo "[run] main dataset missing; see setup_data.sh header"; exit 1; }
fi
export TRAIN_ROOT=${TRAIN_ROOT:-${DATA:-$HOME/data}/overhead/train_jpg/train}
export TEST_ROOT=${TEST_ROOT:-${DATA:-$HOME/data}/overhead/test_jpg/test}

SEEDS=${SEEDS:-"42"}
ROWS=${ROWS:-"full plain noedge nobound nods frozen web resnet radio"}
PER_GEN=${PER_GEN:-500}
RES=$HOME/results_dinov3/vit
SAT=dinov3-vitl16-pretrain-sat493m
WEB=dinov3-vitl16-pretrain-lvd1689m
FULL_CKPT=$RES/$SAT/full_s42/best_checkpoint.pt

log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; echo "[$(date -Is)] $2: $3"; }

# ---------------------------------------------------------------- training ---
run() {  # run <ckpt_dirname> <run_tag> <env...>
  local ckdir=$1 tag=$2; shift 2
  local row=${tag%_s*}
  case " $ROWS " in *" $row "*) ;; *) return;; esac
  local out=$RES/$ckdir/$tag
  if [ -f "$out/test_metrics.json" ]; then echo "[skip] $tag done"; return; fi
  log "\`$* EVAL_SPLITS=val,test RUN_TAG=$tag python dinov3_forensics_train_v2.py\`" "$tag" "started"
  env "$@" EVAL_SPLITS=val,test RUN_TAG=$tag python dinov3_forensics_train_v2.py > logs/$tag.log 2>&1
  local rc=$?
  if [ -f "$out/test_metrics.json" ]; then log "" "$tag" "exit=$rc, test_metrics.json written"
  else log "" "$tag" "exit=$rc, NO test_metrics.json (see logs/$tag.log)"; fi
}

for S in $SEEDS; do
  run $SAT full_s$S     SEED=$S
  run $SAT plain_s$S    SEED=$S ABL_EDGE_GATE=0 ABL_BOUNDARY=0 ABL_DEEP_SUP=0
  run $SAT noedge_s$S   SEED=$S ABL_EDGE_GATE=0
  run $SAT nobound_s$S  SEED=$S ABL_BOUNDARY=0
  run $SAT nods_s$S     SEED=$S ABL_DEEP_SUP=0
  run $SAT frozen_s$S   SEED=$S FREEZE_BACKBONE=1
  run $WEB web_s$S      SEED=$S DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m
  run ../resnet/resnet50 resnet_s$S SEED=$S BACKBONE=resnet50
  run ../radio/C-RADIOv4-SO400M radio_s$S SEED=$S BACKBONE=radio
done

# ------------------------------------------------------- cross-dataset eval ---
if [ ! -f "$FULL_CKPT" ]; then
  log "" "cross-dataset" "SKIPPED: $FULL_CKPT not found (full_s42 has not finished)"
else
  xeval() {  # xeval <name> <args...>
    local name=$1; shift
    local out=cross_${name}_metrics.json
    if [ -f "$out" ]; then echo "[skip] $out exists"; return; fi
    log "\`python eval_cross_dataset.py --dataset $name $*\`" "cross_$name" "started"
    python eval_cross_dataset.py --dataset $name --ckpt "$FULL_CKPT" --out "$out" "$@" > logs/cross_$name.log 2>&1
    local rc=$?
    [ -f "$out" ] && log "" "cross_$name" "exit=$rc, $out written" || log "" "cross_$name" "exit=$rc, FAILED (logs/cross_$name.log)"
  }

  # 7a fmow-fake-small (gated: needs approval on HF; skip cleanly if denied)
  FM=${DATA:-$HOME/data}/fmow-fake-small
  if [ ! -d "$FM/train" ]; then
    hf download geodf/fmow-fake-small --repo-type dataset --local-dir "$FM" > logs/fmow_download.log 2>&1
    for z in "$FM"/train.zip "$FM"/val.zip; do [ -f "$z" ] && unzip -q -o "$z" -d "$FM"; done
  fi
  if [ -d "$FM/train" ] || [ -d "$FM/val" ]; then xeval fmow --root "$FM"
  else log "" "cross_fmow" "SKIPPED: fmow-fake-small not accessible ($(tail -1 logs/fmow_download.log 2>/dev/null))"; fi

  # 7b RSFAKE-1M subset (official test split, seed 42, PER_GEN per generator + equal reals)
  RS=${DATA:-$HOME/data}/rsfake
  SUB=$RS/subset_seed42_n$PER_GEN.csv
  if [ ! -f "$SUB" ]; then
    log "\`python prepare_rsfake_subset.py --root $RS --per-gen $PER_GEN --seed 42\`" "rsfake_subset" "started"
    python prepare_rsfake_subset.py --root "$RS" --per-gen $PER_GEN --seed 42 > logs/prepare_rsfake.log 2>&1
  fi
  [ -f "$SUB" ] && xeval rsfake --root "$RS" --subset-csv "$SUB" --per-gen $PER_GEN \
                || log "" "cross_rsfake" "SKIPPED: subset csv missing"

  # 7c UW Fake Satellite Imagery (optional)
  UW="${DATA:-$HOME/data}/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
  [ -d "$UW" ] && xeval uw --root "$UW" || log "" "cross_uw" "SKIPPED: $UW not found"
fi

# -------------------------------------------------------------- aggregate ---
python aggregate_runs.py --csv table.csv > logs/aggregate.log 2>&1 && log "\`python aggregate_runs.py --csv table.csv\`" "aggregate" "table.csv written"
echo "[$(date -Is)] run_everything.sh finished (SEEDS=$SEEDS ROWS=$ROWS)" >> RUNLOG.md
