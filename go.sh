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
#   stage 5  run_review_seeds.sh      (2026-11-28) resnet and resnetlr seeds 43/44 here, after the evaluations below
#   stage 6  run_lora.sh              (2026-11-28) LoRA variant lora_s42-44 + its zero-shot evaluations, after stage 5 (LORA_PROTOCOL.md)
#   by hand  run_gdd_evals.sh (GDD-trained models in both directions, after their checkpoints are copied back; GDD_TRAIN_PROTOCOL.md)
#   by hand  run_seed_evals.sh (zero-shot / robustness / label-free of seeds 43/44), run_trufor.sh (TruFor on the test folder)
#   extras   (started by hand, not by this launcher) run_fewshot*.sh few-shot probes, run_gdd*.sh GDD scoring,
#            run_zeroshot_extras.sh label-free experiments, run_fewshot_seeds.sh (waits for the queue);
#            supplement/make_supplement.py rebuilds the supplementary material from the result files
#   calypso  run_calypso.sh           (third machine, 2026-11-28) LANE=radio on GPU 0: radio seeds 43/44; LANE=review on GPU 1:
#                                     frozen seeds 43/44; started only when ~/joe_grsl/READY exists there
#            run_calypso_gdd.sh       the full model trained on GDD: gdd_s42 on GPU 1 after the frozen lane; started only when
#                                     ~/joe_grsl/home/data/gdd_layout exists there (seeds 43, 44 moved to the laptop 2026-11-28)
#   laptop   run_laptop.sh            (second machine) fullnorm_s42, webnorm_s42, then plain_s43 (plain_s44: skip_tags.txt); finished
#            run_laptop_gdd.sh        (2026-11-28) the full model trained on GDD: gdd_s43, gdd_s44, in tmux session grsl_gdd; started
#                                     only when ~/data/gdd_layout exists there; results are copied back into ~/results_dinov3 afterwards
source ~/Desktop/.venv/bin/activate
cd ~/grsl; mkdir -p logs
if [ -f .laptop ]; then   # this machine is the laptop: only its own share, never the workstation queue
  pgrep -f "run_lapto[p]|dinov3_forensics_train_v[2]|train_nor[m]" >/dev/null && echo "laptop queue already running" \
    || { [ -f ~/data/gdd_layout/train/metadata.csv ] && { tmux new-session -d -s grsl_gdd "./run_laptop_gdd.sh"; echo "laptop GDD seeds started"; } || echo "laptop: GDD data not there"; }
  exec python watch.py
fi
if pgrep -f "run_everything.sh|run_extra_seeds|run_web_seeds|run_norm_check|run_seed_evals|run_trufor|run_review_seeds|run_lora|run_gdd_evals" >/dev/null; then echo "queue already running"
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
# stage 5: workstation share of the extra seeds (resnet and resnetlr, seeds 43/44); it waits for the GPU by itself.
# The frozen and radio seeds run on Calypso (below).
S5=0; for t in resnet_s43 resnet_s44 resnetlr_s43 resnetlr_s44; do [ -f "$HOME/results_dinov3/resnet/resnet50/$t/test_metrics.json" ] || S5=1; done
if [ $S5 = 1 ] && ! pgrep -f "run_review_seeds.sh" >/dev/null; then
  (setsid nohup ./run_review_seeds.sh > logs/run_review_seeds.nohup 2>&1 < /dev/null &); echo "stage 5 (resnet seeds) started"
fi
# stage 6: LoRA variant (three seeds); waits for stage 5 and for the GPU by itself
S6=0; for t in lora_s42 lora_s43 lora_s44; do [ -f "$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/$t/test_metrics.json" ] || S6=1; done
if [ $S6 = 1 ] && ! pgrep -f "run_lora.sh" >/dev/null; then
  (setsid nohup ./run_lora.sh > logs/run_lora.nohup 2>&1 < /dev/null &); echo "stage 6 (LoRA seeds) started"
fi
# second machine: start its share (run_laptop_gdd.sh: gdd_s43, gdd_s44) if it is reachable, idle and has the GDD layout; GRSL_LAPTOP="" skips this
LAPTOP=${GRSL_LAPTOP-$(cat ~/.config/grsl/laptop 2>/dev/null)}   # address kept outside the repository
if [ -n "$LAPTOP" ]; then
  ssh -n -o BatchMode=yes -o ConnectTimeout=4 "$LAPTOP" 'pgrep -f "run_lapto[p]|dinov3_forensics_train_v[2]|train_nor[m]" >/dev/null && echo "laptop queue already running" || { V=~/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m; [ -f $V/gdd_s43/test_metrics.json ] && [ -f $V/gdd_s44/test_metrics.json ] && { echo "laptop GDD seeds finished"; exit 0; }; [ -f ~/data/gdd_layout/train/metadata.csv ] || { echo "laptop: GDD data not there"; exit 0; }; cd ~/grsl && mkdir -p logs && tmux new-session -d -s grsl_gdd "./run_laptop_gdd.sh" && echo "laptop GDD seeds started"; }' 2>/dev/null || echo "laptop not reachable"
fi
# third machine (Calypso, 2026-11-28): start its two lanes if it is reachable and set up; finished rows are skipped there; GRSL_CALYPSO="" skips this
CAL=${GRSL_CALYPSO-$(cat ~/.config/grsl/calypso 2>/dev/null)}   # address kept outside the repository
if [ -n "$CAL" ]; then
  timeout 15 ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$CAL" '[ -f ~/joe_grsl/READY ] || { echo "calypso not set up"; exit 0; }; R=~/joe_grsl/home/results_dinov3
    lane() { tmux has-session -t joe_grsl_$1 2>/dev/null && { echo "calypso $1 lane already running"; return; }; for t in $2; do [ -f $R/$t/test_metrics.json ] || { tmux new-session -d -s joe_grsl_$1 "LANE=$1 ~/joe_grsl/home/grsl/run_calypso.sh"; echo "calypso $1 lane started"; return; }; done; echo "calypso $1 lane finished"; }
    lane radio "radio/C-RADIOv4-SO400M/radio_s43 radio/C-RADIOv4-SO400M/radio_s44"
    lane review "vit/dinov3-vitl16-pretrain-sat493m/frozen_s43 vit/dinov3-vitl16-pretrain-sat493m/frozen_s44"
    V=vit/dinov3-vitl16-pretrain-sat493m
    gdd() { [ -f ~/joe_grsl/home/data/gdd_layout/train/metadata.csv ] || { echo "calypso gdd: data not there yet"; return; }; tmux has-session -t joe_grsl_gdd$1 2>/dev/null && { echo "calypso gdd lane (GPU $1) already running"; return; }; for s in $3; do [ -f $R/$V/gdd_s$s/test_metrics.json ] || { tmux new-session -d -s joe_grsl_gdd$1 "GPU=$1 AFTER=$2 SEEDS=\"$3\" ~/joe_grsl/home/grsl/run_calypso_gdd.sh"; echo "calypso gdd lane (GPU $1) started"; return; }; done; echo "calypso gdd lane (GPU $1) finished"; }
    gdd 1 joe_grsl_review "42"' 2>/dev/null | grep -v "^#" || echo "calypso not reachable"
fi
python watch.py   # live dashboard: sensors of both machines, every row, laptop training read over ssh
