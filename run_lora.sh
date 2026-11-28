#!/bin/bash
# Added 2026-11-28 (Joe: "yes lora"; review request). LoRA variant of the full model, seeds 42-44, on the workstation
# after the queue that is running. Protocol: LORA_PROTOCOL.md (rank 16, alpha 16, query/value projections of all 24
# blocks, adapter learning rate 1e-5 = LR_DECODER, encoder weights frozen; everything else as the full row).
# After each seed: zero-shot evaluation on the external sets, same rule as for the other models.
# Idempotent: finished runs and existing output files are skipped, nothing is overwritten. One GPU job at a time.
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
busy() { pgrep -f "^/bin/bash ./run_(review_seeds|seed_evals|trufor|gdd_evals).sh" >/dev/null || pgrep -f "dinov3_forensics_train_v[2].py|train_nor[m].py|eval_cross_datase[t].py|trufor_eva[l].py" >/dev/null; }
while busy; do sleep 60; done
log "\`./run_lora.sh\`" "lora" "started (seeds ${SEEDS:-42 43 44})"
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
  O=$R/lora_s$S
  if [ ! -f "$O/test_metrics.json" ]; then
    log "\`SEED=$S LORA_RANK=16 LORA_ALPHA=16 EVAL_SPLITS=val,test RUN_TAG=lora_s$S python dinov3_forensics_train_v2.py\`" "lora_s$S" "started"
    SEED=$S LORA_RANK=16 LORA_ALPHA=16 EVAL_SPLITS=val,test RUN_TAG=lora_s$S python dinov3_forensics_train_v2.py > logs/lora_s$S.log 2>&1
    rc=$?
    [ -f "$O/test_metrics.json" ] && log "" "lora_s$S" "exit=$rc, test_metrics.json written" || { log "" "lora_s$S" "exit=$rc, NO test_metrics.json (see logs/lora_s$S.log)"; continue; }
  fi
  ck=$O/best_checkpoint.pt; p=cross_lora_s${S}_
  ev ${p}fmow $ck --dataset fmow --root $D/fmow-fake-small
  ev ${p}rsfake $ck --dataset rsfake --root $D/rsfake --subset-csv $D/rsfake/subset_seed42_n500.csv --per-gen 500
  ev ${p}uw $ck --dataset uw --root "$D/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
  ev ${p}dmaer $ck --dataset dmaer --root $D/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched
  ev ${p}allreals_dmaer $ck --dataset dmaer --root $D/dm_aer/DM-AER-DeepFake-V1-sample
  for split in indomain crossgenerator crossdomain localization; do
    ev ${p}gdd_$split $ck --dataset gdd --gdd-split $split --root $D/gdd
  done
done
python aggregate_runs.py --csv table.csv > logs/aggregate.log 2>&1 && log "\`python aggregate_runs.py --csv table.csv\`" "aggregate" "table.csv rebuilt with the lora row"
log "" "lora" "finished"
