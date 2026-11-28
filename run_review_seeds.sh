#!/bin/bash
# Added 2026-11-28 (Joe: "start the experiments for grsl"; review request for more seeds of the single-run rows).
# Workstation share: seeds 43 and 44 of the two ResNet-50 rows (resnet, resnetlr), with the existing switches only (no change to the model,
# the losses or the settings). Waits for the seed evaluation queue and TruFor: one GPU job at a time.
# Idempotent: finished runs are skipped. The frozen and C-RADIO seeds run on Calypso (run_calypso.sh).
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
while pgrep -f "^/bin/bash ./run_(seed_evals|trufor).sh" >/dev/null || pgrep -f "dinov3_forensics_train_v[2].py|train_nor[m].py|eval_cross_datase[t].py|trufor_eva[l].py" >/dev/null; do sleep 60; done
log "\`./run_review_seeds.sh\`" "review seeds" "started (resnet, resnetlr; seeds 43 44)"
SKIP_SETUP=1 SEEDS="43 44" ROWS="resnet" ./run_everything.sh >> logs/run_everything.out 2>&1
for S in 43 44; do
  RL=$HOME/results_dinov3/resnet/resnet50/resnetlr_s$S
  [ -f "$RL/test_metrics.json" ] && continue
  log "\`SEED=$S BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s$S python dinov3_forensics_train_v2.py\`" "resnetlr_s$S" "started"
  SEED=$S BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s$S python dinov3_forensics_train_v2.py > logs/resnetlr_s$S.log 2>&1
  rc=$?
  [ -f "$RL/test_metrics.json" ] && log "" "resnetlr_s$S" "exit=$rc, test_metrics.json written" || log "" "resnetlr_s$S" "exit=$rc, NO test_metrics.json (see logs/resnetlr_s$S.log)"
done
python aggregate_runs.py --csv table.csv > logs/aggregate.log 2>&1 && log "\`python aggregate_runs.py --csv table.csv\`" "aggregate" "table.csv rebuilt with frozen / resnet seeds"
log "" "review seeds" "finished"
