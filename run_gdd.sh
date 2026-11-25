#!/bin/bash
# Added 2026-11-25 (Joe approved): GDD (arXiv 2611.31487, CC BY 4.0) as a fifth external test set.
# Downloads the three test splits + split CSVs (retries until access is granted on Hugging Face), then
# scores full_s42, web_s42 and frozen_s42 zero-shot on each split. Evaluation only; nothing is trained on GDD.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
G=$HOME/data/gdd
for i in $(seq 1 1440); do
  hf download geodf/gdd --repo-type dataset --include "test/*" --include "*.csv" --include "README.md" --local-dir "$G" > logs/gdd_download.log 2>&1 && break
  grep -q "requires approval\|Access denied\|gated" logs/gdd_download.log || { sleep 30; continue; }
  sleep 60
done
[ -f "$G/test_indomain_detection.csv" ] || { log "" "gdd" "download FAILED (logs/gdd_download.log)"; exit 1; }
log "" "gdd" "downloaded: $(find "$G/test" -type f | wc -l) files, $(du -sh "$G" | cut -f1)"
R=$HOME/results_dinov3/vit
SAT=facebook/dinov3-vitl16-pretrain-sat493m; WEB=facebook/dinov3-vitl16-pretrain-lvd1689m
for split in indomain crossgenerator crossdomain; do
  for m in "cross_ $SAT $R/dinov3-vitl16-pretrain-sat493m/full_s42" "cross_web_ $WEB $R/dinov3-vitl16-pretrain-lvd1689m/web_s42" "cross_frozen_ $SAT $R/dinov3-vitl16-pretrain-sat493m/frozen_s42"; do
    set -- $m; out=${1}gdd_${split}_metrics.json
    [ -f "$out" ] && continue
    busy=$(pgrep -af "dinov3_forensics_train_v2.py|train_norm.py" | grep -v forkserver | head -1 | cut -c1-40)
    log "\`python eval_cross_dataset.py --dataset gdd --gdd-split $split --dinov3 $2 --ckpt $3/best_checkpoint.pt --out $out\`" "${1}gdd_$split" "started${busy:+ (a training run is active on this GPU: its s/epoch during this evaluation is not clean)}"
    python eval_cross_dataset.py --dataset gdd --gdd-split $split --root "$G" --dinov3 $2 --ckpt "$3/best_checkpoint.pt" --out "$out" > logs/${1}gdd_$split.log 2>&1
    rc=$?; [ -f "$out" ] && log "" "${1}gdd_$split" "exit=$rc, $out written" || log "" "${1}gdd_$split" "exit=$rc, FAILED (logs/${1}gdd_$split.log)"
  done
done
