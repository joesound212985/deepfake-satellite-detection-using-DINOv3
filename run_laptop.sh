#!/bin/bash
# Second machine (laptop) share of the queue, split off on 2026-11-25 at Joe's request:
#   1. radio_s42                   (needs more than the workstation's 16 GB)
#   2. fullnorm_s42, webnorm_s42   (run_norm_check.sh, side experiment)
#   3. plain seeds 43 and 44       (run_everything.sh, same settings as on the workstation)
# Run after setup_env.sh and setup_data.sh. Finished rows are skipped, nothing is overwritten.
# Afterwards copy the run folders back to the workstation's ~/results_dinov3.
# These runs come from a different GPU: use their accuracy numbers, not their timing columns.
source ~/Desktop/.venv/bin/activate
cd "$(dirname "$0")"; mkdir -p logs
echo "| $(date -Is) | \`./run_laptop.sh\` on $(hostname) ($(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)) | laptop queue | started |" >> RUNLOG.md
# radio_s42 ran out of GPU memory on the workstation's 16 GB card (2026-11-25); same settings here.
SKIP_SETUP=1 SEEDS="42" ROWS="radio" ./run_everything.sh >> logs/run_laptop.out 2>&1
./run_norm_check.sh >> logs/run_laptop.out 2>&1
SKIP_SETUP=1 SEEDS="43 44" ROWS="plain" ./run_everything.sh >> logs/run_laptop.out 2>&1
echo "| $(date -Is) |  | laptop queue | finished |" >> RUNLOG.md
