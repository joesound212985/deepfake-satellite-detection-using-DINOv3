#!/bin/bash
# Short launcher: makes sure the whole queue is running in the background (this workstation and the
# laptop), then shows live progress and temperatures for both machines.
# Safe to run any time: finished rows and existing result files are skipped, never overwritten.
#   stage 1  run_everything.sh        six seed-42 rows + cross-dataset eval of full_s42 + table.csv
#   stage 2  run_extra_seeds_v2.sh    DM-AER + web_s42 cross-dataset eval, SimLBR with DM-AER, robustness /
#                                     two-class / figure / analysis, official SimLBR, resnetlr_s42,
#                                     then seeds 43/44 in the order web, full
#   stage 3  run_web_seeds.sh         test-set re-scoring of all seeds + significance tests, table.csv rebuilt
#   laptop   run_laptop.sh            (second machine) fullnorm_s42, webnorm_s42, then plain seeds 43/44;
#                                     results are copied back into ~/results_dinov3 afterwards
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
if [ -f .laptop ]; then   # this machine is the laptop: only its own share, never the workstation queue
  pgrep -f "run_lapto[p]|dinov3_forensics_train_v[2]|train_nor[m]" >/dev/null && echo "laptop queue already running" \
    || { (setsid nohup ./run_laptop.sh > logs/run_laptop.nohup 2>&1 < /dev/null &); echo "laptop queue started"; }
  exec python watch.py
fi
if pgrep -f "run_everything.sh|run_extra_seeds|run_web_seeds|run_norm_check" >/dev/null; then echo "queue already running"
else
  pgrep -f setup_data.sh >/dev/null && export SKIP_SETUP=1   # dataset download still running
  ROWS=${ROWS:-"full plain frozen resnet web radio"} nohup ./run_everything.sh >> logs/run_everything.out 2>&1 &
  sleep 3
  nohup ./run_extra_seeds_v2.sh >> logs/run_extra_seeds_v2.nohup 2>&1 &
  sleep 1
  nohup ./run_web_seeds.sh >> logs/run_web_seeds.nohup 2>&1 &
  sleep 1
  # stage 4 and the plain seeds 43/44 moved to the laptop on 2026-11-25 (run_laptop.sh); NORM_HERE=1 runs stage 4 here
  [ -n "$NORM_HERE" ] && nohup ./run_norm_check.sh >> logs/run_norm_check.nohup 2>&1 &
  echo "queue started"; sleep 2
fi
# second machine: start its share (run_laptop.sh) if it is reachable and idle; GRSL_LAPTOP="" skips this
LAPTOP=${GRSL_LAPTOP-joe@100.72.70.64}
if [ -n "$LAPTOP" ]; then
  ssh -n -o BatchMode=yes -o ConnectTimeout=4 "$LAPTOP" 'pgrep -f "run_lapto[p]|dinov3_forensics_train_v[2]|train_nor[m]" >/dev/null && echo "laptop queue already running" || { cd ~/grsl && mkdir -p logs && (setsid nohup ./run_laptop.sh > logs/run_laptop.nohup 2>&1 < /dev/null &) ; echo "laptop queue started"; }' 2>/dev/null || echo "laptop not reachable"
fi
python watch.py   # live dashboard: sensors of both machines, every row, laptop training read over ssh
