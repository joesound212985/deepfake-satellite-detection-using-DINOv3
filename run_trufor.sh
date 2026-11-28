#!/bin/bash
# Added 2026-11-27 (Joe: "go for trufor"): TruFor on the test folder, TRUFOR_PROTOCOL.md. Waits for the seed evaluation queue (one GPU job at a time).
source ~/Desktop/.venv/bin/activate
cd ~/grsl
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
while pgrep -f "^/bin/bash ./run_seed_evals.sh" >/dev/null; do sleep 30; done
log "\`PYTHONPATH=~/external/trufor_env python trufor_eval.py\`" "trufor_overhead" "started"
PYTHONPATH=$HOME/external/trufor_env python trufor_eval.py > logs/trufor_overhead.log 2>&1
rc=$?; [ -f trufor_overhead_metrics.json ] && log "" "trufor_overhead" "exit=$rc, trufor_overhead_metrics.json written" || log "" "trufor_overhead" "exit=$rc, FAILED (logs/trufor_overhead.log)"
