#!/bin/bash
# Added 2026-11-25 (Joe approved): few-shot adaptation experiment, protocol in FEWSHOT_PROTOCOL.md.
# Waits for the frozen-checkpoint scoring and the web_s42 zero-shot files, then extracts features and
# fits the probes. Evaluation-style work only; no row of the main table is touched.
source ~/Desktop/.venv/bin/activate
cd ~/grsl
log() { echo "| $(date -Is) | $1 | $2 | $3 |" >> RUNLOG.md; }
while pgrep -f "^/bin/bash ./run_frozen_cross" >/dev/null; do sleep 30; done
for d in fmow rsfake uw dmaer; do
  while [ ! -f cross_web_${d}_metrics.per_image.json ] && pgrep -f "run_extra_seeds|run_everything.sh" >/dev/null; do sleep 30; done
done
log "\`python fewshot_adapt.py extract\`" "fewshot" "started"
python fewshot_adapt.py extract > logs/fewshot_extract.log 2>&1; log "" "fewshot extract" "exit=$?"
if [ ! -f fewshot_metrics.json ]; then
  python fewshot_adapt.py analyse > logs/fewshot_analyse.log 2>&1; log "\`python fewshot_adapt.py analyse\`" "fewshot" "exit=$? (fewshot_metrics.json)"
fi
