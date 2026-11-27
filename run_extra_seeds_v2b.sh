#!/bin/bash
# Added 2026-11-26 (Joe: "move it"): plain_s44 moved from the laptop queue to this machine so that
# both machines finish sooner. Same script and settings as plain_s42 / plain_s43 (run_everything.sh,
# ROWS=plain); only the machine differs (plain_s42 was also trained here).
# Waits for run_extra_seeds_v2.sh (web/full seeds 43, 44). The name is chosen so that
# run_web_seeds.sh and run_fewshot_seeds.sh keep waiting until this run has finished too
# (16 GB GPU: one job at a time). Kill this script to cancel.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while pgrep -f '^/bin/bash ./run_(everything|extra_seeds_v2)\.sh' >/dev/null; do sleep 20; done
echo "| $(date -Is) | \`SKIP_SETUP=1 SEEDS=44 ROWS=plain ./run_everything.sh\` (moved from laptop) | plain_s44 | queue started |" >> RUNLOG.md
SKIP_SETUP=1 SEEDS="44" ROWS="plain" ./run_everything.sh >> logs/run_extra_seeds.out 2>&1
