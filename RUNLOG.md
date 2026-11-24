# RUNLOG

## 2026-11-23 setup

- Env: WSL2, RTX 4080 SUPER 16376 MiB (driver 621.14), Python 3.13.14, torch 2.14.1+cu130 (CUDA ok), transformers 5.19.0, timm 1.0.30, opencv-python-headless 5.0.0, scikit-learn 1.9.0, pandas 3.0.5, huggingface_hub 1.33.0. conda env `torch313` (existing, all deps present; nothing installed).
- Disk: WSL root 827 GB free; C: 524 GB free.
- HF: logged in. Tiny download OK for all four gated repos (sat493m, lvd1689m, TZHSW/RSFAKE, geodf/fmow-fake-small).
- Dataset: NOT at /mnt/c/Users/admin/Desktop/{train_jpg/train,test_jpg/test}. Found at /mnt/c/Users/admin/Desktop/train (10000 imgs + masks + metadata.csv, 949 MB) and /mnt/c/Users/admin/Desktop/test (3150, 277 MB). ~/train_jpg/train is empty, ~/test_jpg/test has 96 files (partial). Copying to ~/data/overhead/{train_jpg/train,test_jpg/test}.
- Old checkpoint: ~/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/best_checkpoint.pt (1.24 GB, dated 2025-12-03). Kept.
- MISSING: dinov3_forensics_train_v2.py, aggregate_runs.py, v2_changes.diff not found anywhere on ~ or /mnt/c. Only original ~/dinov3_map_forgery_2025.py exists (copied here as .original.py). Blocked on Joe for the v2 files.

## Class-count / label check (original script logic, metadata.csv) — PASSED

Train (10000):
| manipulation_type | n | label |
|---|---|---|
| pristine | 4964 | pristine |
| inpainted_struct | 1260 | tampered |
| inpainted_unstruct | 1237 | tampered |
| generated_fake | 850 | fake |
| generated_real | 840 | fake |
| generated_uncond | 849 | fake |
Totals: pristine 4964, tampered 2497, fake 2539. Matches authors' page. Letter has fake/tampered swapped (2496/2540) -> fix in letter.

Test (3150): pristine 1511, inpainted_struct 432 + inpainted_unstruct 456 = tampered 888, generated_fake 249 + generated_real 280 + generated_uncond 222 = fake 751. Matches letter.
No mislabeling: no generated_* string contains a tampered keyword. Note: a few inpainted images have manipulated_fraction = 1.0 (max in train inpainted_unstruct, test inpainted_struct); they stay "tampered" per dataset definition.

## 2026-11-23 files received

- Joe pasted v2 script (truncated mid-file), aggregate_runs.py (complete) and v2_changes.diff (complete). v2 rebuilt by `patch -p0` of v2_changes.diff onto the original ~/dinov3_map_forgery_2025.py: applied cleanly, py_compile OK, 23 "[v2]" markers.
- My change: TRAIN_ROOT/TEST_ROOT now read env vars, default ~/data/overhead/{train_jpg/train,test_jpg/test} (WSL copy). Committed.
- fmow-fake-small: README readable but train.zip/val.zip return "Access denied. This repository requires approval" -> Joe must request access on the HF page (approval pending/not yet granted). Dataset card says 30 real + 30 fake.
- RSFAKE: SPLIT csvs downloaded. Test split = 100k real (fmow) + 10 x 10k fakes. Fakes are per-generator zips (~2.7 GB each), so the whole FAKE/ set (27.6 GB) is being downloaded; a subset is then drawn from the official test csv. REAL/fmow is not distributed; needs fMoW-rgb (~3.5 TB raw / ~200 GB rgb) + crop.py.

## Literature notes (open questions)

- May et al. (arXiv:2305.05784, HTML v1): MCC "calculated using the predicted and ground truth masks", no formula, no statement of per-image vs pooled pixels; threshold calibrated per method (value not given). Their Table 1 counts: train 4,964 / 2,496 synthetic / 2,540 manipulated; test 1,511 / 886 synthetic / 753 manipulated. The released metadata.csv gives train 2,539 fake / 2,497 tampered and test 751 fake / 888 tampered, i.e. the paper's synthetic/manipulated columns are swapped relative to the released files. Letter should cite the released-file counts. Their localization baselines (MCC, calibrated, overall): EXIF 0.133, FSG 0.279, Noiseprint 0.274. No IoU reported.
- SimLBR (arXiv:2602.20412, HTML v1): uses DINOv3-Satellite (sat493m) backbone; trains on DiffusionSat-512/256 and evaluates on "held-out test sets from all ten models". Reports RSFake accuracy 88.88 overall (fake 78.87 / real 99.29); DINOv3 without LBR 73.90 (48.28 / 99.52). No AUC, no per-generator numbers, no statement of the official split or image count. Not directly comparable to a zero-shot transfer from the May et al. data.

## 2026-11-23 Step 4 smoke test — PASSED (mechanically)

| time | command | tag | outcome |
|---|---|---|---|
| 16:20-16:43 | `NUM_EPOCHS=1 RUN_TAG=smoke EVAL_SPLITS=val,test python dinov3_forensics_train_v2.py` | smoke | exit 0, test_metrics.json written |

- `[tokens] sequence=581, prefix (CLS+register)=5, patch grid=24x24` — as expected.
- params total 309,428,485 (encoder 303,129,600; heads 6,298,885). train_sec_per_epoch 476.7 s (3000 it, bs 3). ms_per_test_image 113.1 (4-view TTA). GPU memory observed 9,033 MiB during training (nvidia-smi sample, not peak API).
- 1-epoch metrics (not for the letter): acc 0.270, macro-F1 0.178; model predicts "fake" for almost everything after one epoch (cm rows pristine/tampered/fake: [91,16,1404],[37,10,841],[1,0,750]). mean IoU 0.290. No non-finite losses. Will watch per-epoch val F1 in the full run.
- Wall time of whole smoke run 23.5 min (1 train epoch + val eval + test eval + seg confusion + collage).

### Time plan
- ~9.5 min/epoch incl. per-epoch val eval -> ~8.2 h per 50-epoch run.
- Seed 42, 7 rows: ~57 h. Seeds 43+44 for full + plain decoder: +33 h (~90 h total, ~3.8 days). All 21 runs: ~172 h (~7.2 days).
- Queue order (run_queue.sh): seed 42 all rows -> full/plain s43,s44 -> remaining rows s43,s44. Each later stage is skippable by killing the queue; completed runs are kept.
| 2026-11-23T16:57:03-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=full_s42 python dinov3_forensics_train_v2.py` | full_s42 | started |

## 2026-11-23 Step 5 old model re-scored

| time | command | tag | outcome |
|---|---|---|---|
| 16:43-16:56 | `LEGACY_TOKEN_SLICE=1 RUN_EVAL_ONLY=1 EVAL_SPLITS=test RUN_TAG=old_model_rescored CKPT_LOAD=~/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/best_checkpoint.pt python dinov3_forensics_train_v2.py` | old_model_rescored (attempt 1) | exit 0 but INVALID: `[load] missing=408, unexpected=408`. transformers 5.x renamed `encoder.layer.N`->`encoder.model.layer.N`; strict=False silently left the backbone at pretrained weights (acc 0.337). Folder renamed `old_model_rescored_BADLOAD_keys_mismatch`. |
| 17:10-17:23 | same command, after fix | old_model_rescored | exit 0, `missing=0, unexpected=0` |

Fix (committed): `remap_state_dict_keys()` in dinov3_forensics_train_v2.py maps 4.x<->5.x key names (and strips `_orig_mod.`); the loader now raises if any key is still missing/unexpected. Same helper used by eval_cross_dataset.py. Checkpoint header: epoch 3, metric f1, score 0.9942 (val).

Old model (legacy token slicing, 4-view TTA), test set n=3150:
| metric | letter | re-scored |
|---|---|---|
| accuracy | 90.98% (2866/3150) | 90.95% (2865/3150) |
| macro-F1 | 0.907 | 0.907 |
| balanced acc | - | 0.9018 |
| tampered->pristine | 176/888 | 177/888 |
| mean IoU (tampered) | 0.659 | 0.511 |
| mean Dice / MCC | - | 0.627 / 0.606 |
cm rows true (pristine,tampered,fake) x cols pred: [[1447,46,18],[177,683,28],[9,7,735]].
Per-class P/R/F1/AP/AUC: pristine .886/.958/.921/.957/.971; tampered .928/.769/.841/.933/.959; fake .941/.979/.960/.989/.997.
**Case: classification matches (1-image difference, fp16/TTA nondeterminism); only IoU differs -> the letter's 0.659 came from the inflated x1.20 IoU function. Plain IoU of the old model is 0.511.**
ms_per_test_image 228.5 here is NOT representative (GPU shared with full_s42 training); use the value from a solo run.
| 2026-11-23T17:37:55-06:00 | killed at Joe's request | full_s42 | stopped in epoch 2; no test_metrics.json; queue not resumed |
| 2026-11-23T17:52:35-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=full_s42 python dinov3_forensics_train_v2.py` | full_s42 | started |
| 2026-11-23T17:58:57-06:00 |  | full_s42 | exit=137, NO test_metrics.json (see logs/full_s42.log) |
| 2026-11-23T17:58:57-06:00 | `SEED=42 ABL_EDGE_GATE=0 ABL_BOUNDARY=0 ABL_DEEP_SUP=0 EVAL_SPLITS=val,test RUN_TAG=plain_s42 python dinov3_forensics_train_v2.py` | plain_s42 | started |
| 2026-11-23T17:59:12-06:00 | killed at Joe's request (switching to minimum rows) | full_s42, plain_s42 | both stopped, no test_metrics.json |
| 2026-11-23T18:06:57-06:00 | `BACKBONE=resnet50 NUM_EPOCHS=1 RUN_TAG=smoke_resnet EVAL_SPLITS=val,test python dinov3_forensics_train_v2.py` | smoke_resnet | started |
| 2026-11-23T18:11:10-06:00 | killed at Joe's request | smoke_resnet | stopped mid epoch 1; model built OK, ~10 it/s; no test_metrics.json |
| 2026-11-23T18:11:29-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=full_s42 python dinov3_forensics_train_v2.py` | full_s42 | started |
| 2026-11-23T18:14:27-06:00 | killed at Joe's request | resume_test | resume test aborted before completion; resume code untested |
| 2026-11-23T18:31:00-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=full_s42 python dinov3_forensics_train_v2.py` | full_s42 | started |

## 2026-11-23 Moved to second machine (native Linux, user adminuser)

- GPU: NVIDIA GeForce RTX 4080 SUPER, 16376 MiB. Python 3.14.4 venv ~/Desktop/.venv, torch 2.14.1+cu126, transformers 5.22.2.
- Copied bundle: grsl/ -> ~/grsl, results_dinov3/ -> ~/results_dinov3, overhead.tar -> ~/data/overhead (20002 train files, 6302 test files).
- The bundle had no .git, so the earlier git history is not on this machine; new repo initialised from the bundle state.
- pip-installed into the venv: timm, opencv-python-headless, scikit-learn, pandas, requests, tifffile.

| time | command | tag | outcome |
|---|---|---|---|
| 2026-11-23T19:07:18-06:00 | `NUM_EPOCHS=1 RUN_TAG=smoke_machine2 EVAL_SPLITS=val,test python dinov3_forensics_train_v2.py` | smoke_machine2 | started |
| 2026-11-23T19:09:26-06:00 |  | smoke_machine2 | interrupted at 468/3000 iters of epoch 1 (session interruption, not a crash); ~9 it/s, losses finite; no test_metrics.json |
| 2026-11-23T19:12:10-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=full_s42 python dinov3_forensics_train_v2.py` | full_s42 | started |
| 2026-11-23T22:05:03-06:00 | `./run_extra_seeds.sh` (SEEDS="43 44" ROWS="full plain" after the seed-42 queue) | extra_seeds | queued, waiting for run_everything.sh to finish |
| 2026-11-23T22:36:48-06:00 | (stopped by Claude) | full_s42 | **DIVERGED, INVALID.** Epochs 1-17 normal (best val F1 0.9853 at epoch 17), epoch 18 worse (0.9721), epoch 19 iter ~694 "non-finite model outputs", every later batch skipped (train_loss=0, val_loss=nan, val F1 0.2228) plus CUDA OOM messages. last_epoch.pt: 416 of 450 weight tensors non-finite; GradScaler scale = 7.3e-40 (collapsed). best_checkpoint.pt (epoch 17) is finite. Queue and extra-seed waiter stopped; no other row started. Folder renamed full_s42_DIVERGED_ep19_nan, log copied to logs/full_s42_DIVERGED_ep19_nan.log. No test_metrics.json. |
| 2026-11-23T22:37:12-06:00 | `NUM_EPOCHS=5 RUN_TAG=diag_scaler EVAL_SPLITS=val python dinov3_forensics_train_v2.py` | diag_scaler | started: diagnostic only (logs every GradScaler back-off) |
| 2026-11-23T22:50:10-06:00 | killed at Joe's request | diag_scaler | stopped; diagnostic incomplete |
| 2026-11-23T22:54:38-06:00 | `NUM_EPOCHS=6 RUN_TAG=diag_scaler EVAL_SPLITS=val python dinov3_forensics_train_v2.py` (resumes from last_epoch.pt) | diag_scaler | restarted with Joe's approval: diagnostic only |
| 2026-11-23T22:57:05-06:00 | `python simlbr_baseline.py` then `--no-lbr` | simlbr | started (seeds 42 43 44; embeddings computed while diag_scaler shares the GPU) |
| 2026-11-23T23:15:27-06:00 |  | simlbr | done: simlbr_metrics.json, simlbr_nolbr_metrics.json (paper schedule, 10 epochs = 180 optimizer steps on 9000 images; best epoch 9-10, i.e. still improving) and simlbr_ep300_metrics.json, simlbr_nolbr_ep300_metrics.json (300 epochs, disclosed deviation). Re-implementation from the paper text; unstated details chosen by us are listed in the script header and in each json. |
| 2026-11-23T23:24:10-06:00 | `python simlbr_baseline.py` and `--no-lbr` | simlbr | re-run with the settings of the authors' released code (github.com/mvrl/SimLBR ac90f86: 256 px input, 1024-512-256-1 head with dropout 0.3, val-accuracy selection). Earlier paper-text versions moved to superseded/. |
