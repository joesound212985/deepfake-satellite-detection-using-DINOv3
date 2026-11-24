#!/bin/bash
# Waits for smoke+rescore to finish, then: all rows seed 42 -> full/plain seeds 43,44 -> remaining rows seeds 43,44.
cd ~/grsl
while ! grep -q ALL_DONE logs/driver.log; do sleep 30; done
./run_all.sh 42
ROWS="full plain" ./run_all.sh 43 44
./run_all.sh 43 44
echo "[$(date -Is)] QUEUE_DONE" >> RUNLOG.md
