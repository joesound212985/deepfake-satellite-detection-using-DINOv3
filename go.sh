#!/bin/bash
# Short launcher: makes sure the whole queue is running in the background (this workstation and the
# laptop), then shows live progress and temperatures for both machines.
# Safe to run any time: finished rows and existing result files are skipped, never overwritten.
#   stage 1  run_everything.sh        six seed-42 rows + cross-dataset eval of full_s42 + table.csv
#   stage 2  run_extra_seeds_v2.sh    DM-AER + web_s42 cross-dataset eval, SimLBR with DM-AER, robustness /
#                                     two-class / figure / analysis, official SimLBR, resnetlr_s42,
#                                     then seeds 43/44 in the order web, full
#   stage 2b run_extra_seeds_v2b.sh   plain_s44 (moved here from the laptop on 2026-11-26), after stage 2
#   stage 3  run_web_seeds.sh         test-set re-scoring of all seeds + significance tests, table.csv rebuilt
#   stage 5  run_review_seeds.sh      (2026-11-28) seeds 43/44 of the frozen and the two ResNet-50 rows; waits for
#                                     run_seed_evals.sh (zero-shot / robustness / label-free of seeds 43/44) and
#                                     run_trufor.sh (TruFor on the test folder), which are started by hand
#   extras   (started by hand, not by this launcher) run_fewshot*.sh few-shot probes, run_gdd*.sh GDD scoring,
#            run_zeroshot_extras.sh label-free experiments, run_fewshot_seeds.sh (waits for the queue);
#            supplement/make_supplement.py rebuilds the supplementary material from the result files
#   laptop   run_laptop.sh            (second machine) fullnorm_s42, webnorm_s42, then plain_s43 (plain_s44: skip_tags.txt);
#                                     results are copied back into ~/results_dinov3 afterwards
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
if [ -f .laptop ]; then   # this machine is the laptop: only its own share, never the workstation queue
  pgrep -f "run_lapto[p]|dinov3_forensics_train_v[2]|train_nor[m]" >/dev/null && echo "laptop queue already running" \
    || { (setsid nohup ./run_laptop.sh > logs/run_laptop.nohup 2>&1 < /dev/null &); echo "laptop queue started"; }
  exec python watch.py
fi
if pgrep -f "run_everything.sh|run_extra_seeds|run_web_seeds|run_norm_check|run_seed_evals|run_trufor|run_review_seeds" >/dev/null; then echo "queue already running"
else
  pgrep -f setup_data.sh >/dev/null && export SKIP_SETUP=1   # dataset download still running
  ROWS=${ROWS:-"full plain frozen resnet web radio"} nohup ./run_everything.sh >> logs/run_everything.out 2>&1 &
  sleep 3
  nohup ./run_extra_seeds_v2.sh >> logs/run_extra_seeds_v2.nohup 2>&1 &
  sleep 1
  nohup ./run_extra_seeds_v2b.sh >> logs/run_extra_seeds_v2b.nohup 2>&1 &
  sleep 1
  nohup ./run_web_seeds.sh >> logs/run_web_seeds.nohup 2>&1 &
  sleep 1
  # stage 4 and plain_s43 run on the laptop (run_laptop.sh, since 2026-11-25); NORM_HERE=1 runs stage 4 here
  [ -n "$NORM_HERE" ] && nohup ./run_norm_check.sh >> logs/run_norm_check.nohup 2>&1 &
  echo "queue started"; sleep 2
fi
# stage 5: start it if it is not running and one of its runs is still missing (it waits for the GPU by itself)
R5=$HOME/results_dinov3; S5=0
for t in vit/dinov3-vitl16-pretrain-sat493m/frozen_s43 vit/dinov3-vitl16-pretrain-sat493m/frozen_s44 resnet/resnet50/resnet_s43 resnet/resnet50/resnet_s44 resnet/resnet50/resnetlr_s43 resnet/resnet50/resnetlr_s44; do
  [ -f "$R5/$t/test_metrics.json" ] || S5=1
done
if [ $S5 = 1 ] && ! pgrep -f "run_review_seeds.sh" >/dev/null; then
  (setsid nohup ./run_review_seeds.sh > logs/run_review_seeds.nohup 2>&1 < /dev/null &); echo "stage 5 (review seeds) started"
fi
# second machine: start its share (run_laptop.sh) if it is reachable and idle; GRSL_LAPTOP="" skips this
LAPTOP=${GRSL_LAPTOP-$(cat ~/.config/grsl/laptop 2>/dev/null)}   # address kept outside the repository
if [ -n "$LAPTOP" ]; then
  ssh -n -o BatchMode=yes -o ConnectTimeout=4 "$LAPTOP" 'pgrep -f "run_lapto[p]|dinov3_forensics_train_v[2]|train_nor[m]" >/dev/null && echo "laptop queue already running" || { cd ~/grsl && mkdir -p logs && (setsid nohup ./run_laptop.sh > logs/run_laptop.nohup 2>&1 < /dev/null &) ; echo "laptop queue started"; }' 2>/dev/null || echo "laptop not reachable"
fi
python watch.py   # live dashboard: sensors of both machines, every row, laptop training read over ssh
