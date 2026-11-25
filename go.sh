#!/bin/bash
# Short launcher: makes sure the whole queue is running in the background, then shows progress.
# Safe to run any time: finished rows and existing result files are skipped, never overwritten.
#   stage 1  run_everything.sh        six seed-42 rows + cross-dataset eval of full_s42 + table.csv
#   stage 2  run_extra_seeds_v2.sh    DM-AER + web_s42 cross-dataset eval, SimLBR with DM-AER, robustness /
#                                     two-class / figure / analysis_extras.json, then seeds 43/44 full + plain
#   stage 3  run_web_seeds.sh         web seeds 43/44, table.csv rebuilt
#   stage 4  run_norm_check.sh        side experiment: fullnorm_s42, webnorm_s42 (backbone input normalisation)
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
if pgrep -f "run_everything.sh|run_extra_seeds|run_web_seeds|run_norm_check" >/dev/null; then echo "queue already running"
else
  pgrep -f setup_data.sh >/dev/null && export SKIP_SETUP=1   # dataset download still running
  ROWS=${ROWS:-"full plain frozen resnet web radio"} nohup ./run_everything.sh >> logs/run_everything.out 2>&1 &
  sleep 3
  nohup ./run_extra_seeds_v2.sh >> logs/run_extra_seeds_v2.nohup 2>&1 &
  sleep 1
  nohup ./run_web_seeds.sh >> logs/run_web_seeds.nohup 2>&1 &
  sleep 1
  nohup ./run_norm_check.sh >> logs/run_norm_check.nohup 2>&1 &
  echo "queue started"; sleep 2
fi
python watch.py
