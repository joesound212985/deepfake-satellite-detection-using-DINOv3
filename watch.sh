#!/bin/bash
# Live progress for the GRSL queue.  Usage:  ~/grsl/watch.sh      (Ctrl-C to exit)
cd "$(dirname "$0")"
RES=$HOME/results_dinov3/vit
while true; do
  clear
  echo "GRSL queue  $(date '+%F %T')   GPU: $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null)"
  echo "------------------------------------------------------------------------------------------"
  printf "%-10s %s\n" "run" "status"
  for tag in full_s42 plain_s42 frozen_s42 resnet_s42 radio_s42 noedge_s42 nobound_s42 nods_s42 web_s42 full_s43 plain_s43 full_s44 plain_s44; do
    ck=$RES/dinov3-vitl16-pretrain-sat493m; [[ $tag == web* ]] && ck=$RES/dinov3-vitl16-pretrain-lvd1689m; [[ $tag == resnet* ]] && ck=$HOME/results_dinov3/resnet/resnet50; [[ $tag == radio* ]] && ck=$HOME/results_dinov3/radio/C-RADIOv4-SO400M
    if [ -f "$ck/$tag/test_metrics.json" ]; then
      printf "%-10s DONE  acc=%s  macroF1=%s  IoU=%s\n" "$tag" \
        $(python3 -c "import json;m=json.load(open('$ck/$tag/test_metrics.json'));print(f\"{m['accuracy']:.4f} {m['macro_f1']:.4f} {m['localization_tampered']['mean_iou']:.4f}\")")
    elif [ -f "logs/$tag.log" ] && pgrep -f "RUN_TAG=$tag" >/dev/null 2>&1 || { [ -f "logs/$tag.log" ] && [ "$(find logs/$tag.log -mmin -2)" ]; }; then
      bar=$(tr '\r' '\n' < logs/$tag.log | grep -E "^\[train\] epoch|^\[eval\]" | tail -1 | cut -c1-95)
      last=$(tr '\r' '\n' < logs/$tag.log | grep -E "^\[epoch" | tail -1 | sed 's/train_loss=[^ ]* //; s/train_iou=[^ ]* //' | cut -c1-110)
      printf "%-10s RUNNING  %s\n" "$tag" "$bar"
      [ -n "$last" ] && printf "%-10s          %s\n" "" "$last"
    elif [ -f "logs/$tag.log" ]; then
      printf "%-10s stopped/incomplete\n" "$tag"
    else
      printf "%-10s queued\n" "$tag"
    fi
  done
  echo "------------------------------------------------------------------------------------------"
  echo "cross-dataset:"; for d in fmow rsfake uw; do [ -f cross_${d}_metrics.json ] && echo "  $d: done" || { [ -f logs/cross_$d.log ] && echo "  $d: $(tr '\r' '\n' < logs/cross_$d.log | grep -E '^\[eval\]' | tail -1 | cut -c1-80)" || echo "  $d: pending"; }; done
  echo "last RUNLOG lines:"; tail -3 RUNLOG.md | cut -c1-120
  sleep 5
done
