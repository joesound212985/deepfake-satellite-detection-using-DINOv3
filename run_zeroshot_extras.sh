#!/bin/bash
# Added 2026-11-26 (Joe approved): label-free transfer experiments, ZEROSHOT_EXTRAS_PROTOCOL.md.
# Own-data features are extracted on the laptop (workstation GPU is full); the analysis runs here on CPU.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
L=${GRSL_LAPTOP-$(cat ~/.config/grsl/laptop 2>/dev/null)}   # address kept outside the repository
[ -n "$L" ] || { echo "no laptop address: set GRSL_LAPTOP or write it to ~/.config/grsl/laptop"; exit 1; }
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
log "\`python fewshot_adapt.py extract --sets own\` (on laptop, alongside its training run)" "zeroshot_extras" "started"
ssh -n $L 'cd ~/grsl && git fetch -q origin grsl-experiments && git checkout -q origin/grsl-experiments -- fewshot_adapt.py eval_cross_dataset.py && source ~/Desktop/.venv/bin/activate && python fewshot_adapt.py extract --sets own' > logs/zeroshot_extras_extract.log 2>&1
rc=$?; log "" "zeroshot_extras extract" "exit=$rc"; [ $rc -eq 0 ] || exit 1
rsync -a --ignore-existing $L:data/features/fewshot_cls384/ ~/data/features/fewshot_cls384/ || exit 1
python zeroshot_extras.py > logs/zeroshot_extras.log 2>&1; log "\`python zeroshot_extras.py\`" "zeroshot_extras" "exit=$? (zeroshot_extras_metrics.json)"
