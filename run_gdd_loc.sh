#!/bin/bash
# Follow-up to run_gdd.sh (2026-11-25): GDD's detection CSVs hold no locally manipulated images; those are in
# test_indomain_localization.csv (500 real + 500 manipulated, six manipulation types, pixel masks).
# Scores the same three checkpoints on it (detection + tamper-mask IoU / Dice). Evaluation only.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while pgrep -f "^/bin/bash ./run_gdd.sh" >/dev/null; do sleep 30; done
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
G=$HOME/data/gdd; R=$HOME/results_dinov3/vit
SAT=facebook/dinov3-vitl16-pretrain-sat493m; WEB=facebook/dinov3-vitl16-pretrain-lvd1689m
for m in "cross_ $SAT $R/dinov3-vitl16-pretrain-sat493m/full_s42" "cross_web_ $WEB $R/dinov3-vitl16-pretrain-lvd1689m/web_s42" "cross_frozen_ $SAT $R/dinov3-vitl16-pretrain-sat493m/frozen_s42"; do
  set -- $m; out=${1}gdd_localization_metrics.json
  [ -f "$out" ] && continue
  log "\`python eval_cross_dataset.py --dataset gdd --gdd-split localization --dinov3 $2 --ckpt $3/best_checkpoint.pt --out $out\`" "${1}gdd_localization" "started"
  python eval_cross_dataset.py --dataset gdd --gdd-split localization --root "$G" --dinov3 $2 --ckpt "$3/best_checkpoint.pt" --out "$out" > logs/${1}gdd_localization.log 2>&1
  rc=$?; [ -f "$out" ] && log "" "${1}gdd_localization" "exit=$rc, $out written" || log "" "${1}gdd_localization" "exit=$rc, FAILED (logs/${1}gdd_localization.log)"
done
