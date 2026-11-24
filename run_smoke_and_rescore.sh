#!/bin/bash
# Step 4 smoke test then Step 5 old-model rescoring, sequential (GPU memory).
cd ~/grsl
while ! grep -q COPY_DONE ~/data/overhead/copy.log; do sleep 10; done
echo "[$(date -Is)] smoke start" >> logs/driver.log
NUM_EPOCHS=1 RUN_TAG=smoke EVAL_SPLITS=val,test python dinov3_forensics_train_v2.py > logs/smoke.log 2>&1
echo "[$(date -Is)] smoke exit=$?" >> logs/driver.log
nvidia-smi --query-gpu=memory.used --format=csv >> logs/driver.log
echo "[$(date -Is)] rescore start" >> logs/driver.log
LEGACY_TOKEN_SLICE=1 RUN_EVAL_ONLY=1 EVAL_SPLITS=test RUN_TAG=old_model_rescored \
CKPT_LOAD=$HOME/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/best_checkpoint.pt \
python dinov3_forensics_train_v2.py > logs/old_model_rescored.log 2>&1
echo "[$(date -Is)] rescore exit=$?" >> logs/driver.log
echo ALL_DONE >> logs/driver.log
