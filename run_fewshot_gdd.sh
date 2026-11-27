#!/bin/bash
# Added 2026-11-25 (Joe approved): few-shot on GDD + t-SNE figure, FEWSHOT_PROTOCOL.md addendum.
# Feature extraction on the laptop (workstation GPU is full with a training run); probes and figure here on CPU.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
L=${GRSL_LAPTOP-$(cat ~/.config/grsl/laptop 2>/dev/null)}   # address kept outside the repository
[ -n "$L" ] || { echo "no laptop address: set GRSL_LAPTOP or write it to ~/.config/grsl/laptop"; exit 1; }
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
V=results_dinov3/vit
log "rsync GDD test data + full_s42/web_s42 checkpoints to laptop" "fewshot_gdd" "started"
ssh -n $L "mkdir -p ~/data/gdd ~/$V/dinov3-vitl16-pretrain-sat493m/full_s42 ~/$V/dinov3-vitl16-pretrain-lvd1689m/web_s42 ~/data/features/fewshot_cls384" || exit 1
rsync -a ~/data/gdd/ $L:data/gdd/ || exit 1
rsync -a ~/$V/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt $L:$V/dinov3-vitl16-pretrain-sat493m/full_s42/ || exit 1
rsync -a ~/$V/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt $L:$V/dinov3-vitl16-pretrain-lvd1689m/web_s42/ || exit 1
log "\`python fewshot_adapt.py extract --sets gdd\` (on laptop, alongside its training run)" "fewshot_gdd" "started"
ssh -n $L 'cd ~/grsl && git fetch -q origin grsl-experiments && git checkout -q origin/grsl-experiments -- fewshot_adapt.py eval_cross_dataset.py && source ~/Desktop/.venv/bin/activate && python fewshot_adapt.py extract --sets gdd' > logs/fewshot_gdd_extract.log 2>&1
rc=$?; log "" "fewshot_gdd extract" "exit=$rc"; [ $rc -eq 0 ] || exit 1
rsync -a --ignore-existing $L:data/features/fewshot_cls384/ ~/data/features/fewshot_cls384/ || exit 1
python fewshot_adapt.py analyse --sets gdd > logs/fewshot_gdd_analyse.log 2>&1; log "\`python fewshot_adapt.py analyse --sets gdd\`" "fewshot_gdd" "exit=$? (fewshot_gdd_metrics.json)"
python tsne_features.py > logs/tsne_features.log 2>&1; log "\`python tsne_features.py\`" "tsne" "exit=$? (tsne_features.png)"
