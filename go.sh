#!/bin/bash
# Short launcher: starts the queue in the background, then shows progress.
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
if pgrep -f run_everything.sh >/dev/null; then echo "queue already running"
else
  pgrep -f setup_data.sh >/dev/null && export SKIP_SETUP=1   # dataset download still running
  ROWS=${ROWS:-"full plain frozen resnet web radio"} nohup ./run_everything.sh > logs/run_everything.out 2>&1 &
  echo "queue started"; sleep 3
fi
python watch.py
