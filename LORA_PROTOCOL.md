# LoRA variant: protocol

Written 2026-11-28 before any LoRA run was started (Joe, 2026-11-28: "yes lora"; a review asked for a
parameter-efficient variant next to full fine-tuning and the frozen backbone). No number from this experiment
existed when this file was committed.

## Question

Does low-rank adaptation of the encoder reach the accuracy of full fine-tuning on the benchmark, and does it
transfer differently to the external test sets?

## What is trained

Everything is the `full` row (DINOv3 ViT-L/16 satellite weights, edge-gated decoder, same losses, augmentation,
sampler, 50 epochs, batch size 3, 384 x 384, 10 % of the training set held out with VAL_SEED=42, best epoch by
validation macro F1) except the encoder:

- The 303 M pre-trained encoder weights stay frozen for the whole run.
- Low-rank adapters (LoRA, Hu et al. 2022) are added to the query and value projections of all 24 blocks
  (48 linear layers). Rank 16, alpha 16 (scale 1), no dropout. A starts as in `nn.Linear`, B starts at zero, so the
  run starts from the pre-trained encoder exactly. 1,572,864 adapter parameters.
- Adapter learning rate 1e-5: the adapters are in the decoder / classifier parameter group (LR_DECODER, weight
  decay 1e-4 like every other parameter).
- The adapters are kept frozen during the three warm-up epochs, as the backbone is in the `full` row, and are
  trained from epoch 4.

Rank, alpha, target layers and learning rate were fixed here and are not tuned afterwards. If the result is poor
it is reported as it is; any second setting would be a new, separately labelled experiment that needs Joe's word.

## Code

`dinov3_forensics_train_v2.py`, switch `LORA_RANK` (default 0: the adapter class is never built and every other
row runs as before) and `LORA_ALPHA` (default = rank). The adapters are added through forward hooks, so the
encoder keeps its parameter names. `best_checkpoint.pt` stores the merged weights W + (alpha / rank) B A and no
adapter keys, so the evaluation scripts load it as an ordinary model; the script's own final test is run after
loading that merged checkpoint. Checked on the CPU before the runs (plain versus merged versus hooked model:
largest output difference 1.4e-6; no gradient reaches the frozen encoder weights).

Command: `SEED=$S LORA_RANK=16 LORA_ALPHA=16 EVAL_SPLITS=val,test RUN_TAG=lora_s$S python dinov3_forensics_train_v2.py`

## Runs and reporting

- Seeds 42, 43, 44 (`run_lora.sh`, workstation, after the queue that is running). Results in
  `~/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/lora_s{42,43,44}`.
- Reported: the same test-set columns as the other rows of the main table (mean and sample standard deviation
  over three seeds), and the number of trained parameters.
- Zero-shot on the external sets (fmow-fake-small, RSFAKE subset, UW, DM-AER, GDD splits) with
  `eval_cross_dataset.py`, same rule and thresholds as for the other models, files `cross_lora_s*_*`.
- The test split is not used for any choice.
