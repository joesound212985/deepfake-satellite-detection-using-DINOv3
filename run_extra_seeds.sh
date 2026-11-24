#!/bin/bash
# Waits for the seed-42 queue to finish, then runs seeds 43 and 44 for full + plain
# (gives mean +/- std on the two main rows). Kill this script to cancel.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
while pgrep -f "run_everything.sh" >/dev/null; do sleep 120; done
SEEDS="43 44" ROWS="full plain" ./run_everything.sh > logs/run_extra_seeds.out 2>&1
