#!/bin/bash
# Added 2026-11-27 (Joe approved): zero-shot, robustness and label-free evaluations with the seed-43/44
# checkpoints of the full and the web model (ZEROSHOT_EXTRAS_PROTOCOL.md, addendum 2026-11-27).
# Evaluation only; existing outputs are skipped, nothing is overwritten.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
R=$HOME/results_dinov3/vit; D=$HOME/data
SAT=facebook/dinov3-vitl16-pretrain-sat493m; WEB=facebook/dinov3-vitl16-pretrain-lvd1689m
ev() {   # ev NAME REPO CKPT args...
  local name=$1 repo=$2 ck=$3; shift 3
  [ -f "${name}_metrics.json" ] && return 0
  log "\`python eval_cross_dataset.py --dinov3 $repo --ckpt $ck --out ${name}_metrics.json $*\`" "$name" "started"
  python eval_cross_dataset.py --dinov3 "$repo" --ckpt "$ck" --out "${name}_metrics.json" "$@" > "logs/$name.log" 2>&1
  local rc=$?; [ -f "${name}_metrics.json" ] && log "" "$name" "exit=$rc, ${name}_metrics.json written" || log "" "$name" "exit=$rc, FAILED (logs/$name.log)"
}
log "\`./run_seed_evals.sh\`" "seed evals" "started"
for sd in 43 44; do
  for m in "cross_s${sd}_ $SAT $R/dinov3-vitl16-pretrain-sat493m/full_s$sd/best_checkpoint.pt" "cross_web_s${sd}_ $WEB $R/dinov3-vitl16-pretrain-lvd1689m/web_s$sd/best_checkpoint.pt"; do
    set -- $m; p=$1; repo=$2; ck=$3
    ev ${p}fmow $repo $ck --dataset fmow --root $D/fmow-fake-small
    ev ${p}rsfake $repo $ck --dataset rsfake --root $D/rsfake --subset-csv $D/rsfake/subset_seed42_n500.csv --per-gen 500
    ev ${p}uw $repo $ck --dataset uw --root "$D/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data"
    ev ${p}dmaer $repo $ck --dataset dmaer --root $D/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched
    ev ${p}allreals_dmaer $repo $ck --dataset dmaer --root $D/dm_aer/DM-AER-DeepFake-V1-sample
    for split in indomain crossgenerator crossdomain localization; do
      ev ${p}gdd_$split $repo $ck --dataset gdd --gdd-split $split --root $D/gdd
    done
  done
done
for sd in 43 44; do
  for m in "robust_full_s${sd}_ $SAT $R/dinov3-vitl16-pretrain-sat493m/full_s$sd/best_checkpoint.pt" "robust_web_s${sd}_ $WEB $R/dinov3-vitl16-pretrain-lvd1689m/web_s$sd/best_checkpoint.pt"; do
    set -- $m; p=$1; repo=$2; ck=$3
    for c in jpeg:90 jpeg:70 jpeg:50 jpeg:30 scale:0.75 scale:0.5; do
      ev ${p}${c/:/} $repo $ck --dataset overhead --root $D/overhead/test_jpg/test --degrade $c
    done
  done
done
log "\`python fewshot_adapt.py extract --sets own_seeds\`" "zeroshot_extras_seeds" "started"
python fewshot_adapt.py extract --sets own_seeds > logs/zeroshot_extras_seeds_extract.log 2>&1; rc=$?; log "" "zeroshot_extras_seeds extract" "exit=$rc"
[ $rc -eq 0 ] && { python zeroshot_extras.py --seeds > logs/zeroshot_extras_seeds.log 2>&1; log "\`python zeroshot_extras.py --seeds\`" "zeroshot_extras_seeds" "exit=$? (zeroshot_extras_seeds_metrics.json)"; }
log "" "seed evals" "finished"
