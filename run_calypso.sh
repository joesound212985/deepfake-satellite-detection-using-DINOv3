#!/bin/bash
# Third machine (Calypso, 2 x RTX A6000 48 GB; a lab machine with a shared account), added 2026-11-28.
#   LANE=radio   (GPU 0)  radio seeds 43 and 44 (C-RADIO does not fit the workstation's 16 GB)
#   LANE=review  (GPU 1)  frozen seeds 43 and 44 (resnet and resnetlr seeds 43/44 run on the workstation, which is faster for them)
# Same settings as the seed-42 rows; existing switches only. One job per GPU.
# Everything stays inside ~/joe_grsl on that machine: HOME is pointed at ~/joe_grsl/home for the run, so results,
# logs and the model cache do not land in the shared home. Another user's server holds memory on both cards.
# Finished rows are skipped. Afterwards copy the run folders back to the workstation's ~/results_dinov3.
# These runs come from a different GPU: use their accuracy numbers, not their timing columns.
LANE=${LANE:-radio}
REAL=$HOME
source "$REAL/miniconda3/etc/profile.d/conda.sh"; conda activate joe_grsl
export HOME=$REAL/joe_grsl/home PYTHONUNBUFFERED=1
cd "$HOME/grsl"; mkdir -p logs
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
log "\`LANE=$LANE ./run_calypso.sh\` on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1))" "calypso $LANE" "started"
if [ "$LANE" = radio ]; then
  export CUDA_VISIBLE_DEVICES=0
  SKIP_SETUP=1 SEEDS="43 44" ROWS="radio" ./run_everything.sh >> logs/run_calypso_radio.out 2>&1
else
  export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1     # DINOv3 and ResNet-50 weights were copied from the workstation's cache
  SKIP_SETUP=1 SEEDS="43 44" ROWS="frozen" ./run_everything.sh >> logs/run_calypso_review.out 2>&1
fi
log "" "calypso $LANE" "finished"
