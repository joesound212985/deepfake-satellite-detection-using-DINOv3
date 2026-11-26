#!/bin/bash
# Added 2026-11-26 (Joe approved): few-shot probes with the seed-43/44 backbones (FEWSHOT_PROTOCOL.md addendum).
# Waits until the whole workstation queue (training + stage-3 re-scoring) has finished, so no timing is disturbed.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
while pgrep -f "^/bin/bash ./run_(everything|extra_seeds_v2|web_seeds)" >/dev/null; do sleep 60; done
log "\`python fewshot_adapt.py extract --sets seeds\`" "fewshot_seeds" "started"
python fewshot_adapt.py extract --sets seeds > logs/fewshot_seeds_extract.log 2>&1; rc=$?; log "" "fewshot_seeds extract" "exit=$rc"
[ $rc -eq 0 ] || exit 1
python fewshot_adapt.py analyse --sets seeds > logs/fewshot_seeds_analyse.log 2>&1; log "\`python fewshot_adapt.py analyse --sets seeds\`" "fewshot_seeds" "exit=$? (fewshot_seeds_metrics.json)"
