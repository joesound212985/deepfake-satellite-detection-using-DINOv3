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
| 2026-11-23T23:58:46-06:00 |  | diag_scaler | done (6 epochs). GradScaler scale: 2048, 1024, 1024 (frozen epochs, 1-2 back-offs each) then 32, 64, 16 (unfrozen, 5/1/3 back-offs); non-finite gradients first appear in encoder layers 0-1. Cause of the full_s42 NaNs: fp16 backward overflow -> scale ratchets down to underflow. |
| 2026-11-23T23:58:46-06:00 | `NUM_EPOCHS=6 RUN_TAG=verify_bf16 EVAL_SPLITS=val,test python dinov3_forensics_train_v2.py` | verify_bf16 | started: check of the bf16 fix (not a reported result) |
| 2026-11-24T00:59:46-06:00 |  | verify_bf16 | exit 0. 6 epochs with AMP_DTYPE=bf16: 0 skipped steps, 0 non-finite warnings (fp16 diagnostic: 13 back-offs in the same 6 epochs). val F1 per epoch 0.284, 0.342, 0.408, 0.653, 0.936, 0.959 (fp16 full_s42: 0.280, 0.342, 0.407, 0.651, 0.926, 0.955). Full test evaluation and test_metrics.json written without error. Check run only, NOT a reported result. |

### Fix for the full_s42 NaNs (diff: logs/fix_amp_resume.diff)
- Cause: fp16 autocast. Backward pass overflows in encoder layers 0-1 once the backbone is unfrozen; each overflow makes GradScaler skip the step and halve its scale; the scale never recovers and underflows (7.3e-40 at epoch 19), then the weights go NaN.
- Change: model forward/backward in bfloat16 (AMP_DTYPE, default bf16; fp16 restores the old behaviour, fp32 disables AMP); losses, softmax and sigmoid computed in float32; no GradScaler for bf16, with an explicit skip of any step whose gradient norm is non-finite; training raises as soon as the weights are non-finite or an epoch had no usable batch; amp_dtype recorded in test_metrics.json.
- Resume: RNG state tensors are moved back to the CPU before restoring (was failing with "RNG state must be a torch.ByteTensor" after the weights had already been loaded); a failed resume now raises (RESUME=0 to start fresh). The resume path has still not been exercised end to end.
- Model, losses, learning rates, schedule, augmentation and data are unchanged.
| 2026-11-24T00:59:46-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=full_s42 python dinov3_forensics_train_v2.py` | full_s42 | started |
| 2026-11-24T00:59:48-06:00 | `./run_extra_seeds.sh` | extra_seeds | re-queued behind the seed-42 rows |
| 2026-11-24T09:06:57-06:00 | `CUDA_VISIBLE_DEVICES="" python count_flops.py` | flops | wrote flops.json (CPU, batch 1, 384x384, single view). torch FlopCounterMode counts 2 per multiply-add; gmacs = gflops/2 |
| 2026-11-24T09:06:57-06:00 | `pip install einops open_clip_torch` | env | C-RADIO remote code needs both; they were missing on this machine and radio_s42 would have failed at import. torch/timm/transformers versions unchanged |
| 2026-11-24T09:48:18-06:00 |  | full_s42 | exit=0, test_metrics.json written |
| 2026-11-24T09:48:18-06:00 | `SEED=42 ABL_EDGE_GATE=0 ABL_BOUNDARY=0 ABL_DEEP_SUP=0 EVAL_SPLITS=val,test RUN_TAG=plain_s42 python dinov3_forensics_train_v2.py` | plain_s42 | started |
| 2026-11-24T18:08:02-06:00 |  | plain_s42 | exit=0, test_metrics.json written |
| 2026-11-24T18:08:02-06:00 | `SEED=42 FREEZE_BACKBONE=1 EVAL_SPLITS=val,test RUN_TAG=frozen_s42 python dinov3_forensics_train_v2.py` | frozen_s42 | started |
| 2026-11-24T22:57:37-06:00 |  | frozen_s42 | exit=0, test_metrics.json written |
| 2026-11-24T22:57:37-06:00 | `SEED=42 DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m EVAL_SPLITS=val,test RUN_TAG=web_s42 python dinov3_forensics_train_v2.py` | web_s42 | started |
| 2026-11-25T07:46:57-06:00 |  | web_s42 | exit=0, test_metrics.json written |
| 2026-11-25T07:46:57-06:00 | `SEED=42 BACKBONE=resnet50 EVAL_SPLITS=val,test RUN_TAG=resnet_s42 python dinov3_forensics_train_v2.py` | resnet_s42 | started |
| 2026-11-25T07:51:12-06:00 | `./run_web_seeds.sh` | web_seeds | queued behind extra seeds: web_s43, web_s44, cross-dataset eval of web_s42 (Joe's request after web_s42 scored best) |
| 2026-11-25T08:04:44-06:00 | download DM-AER-DeepFake-V1-sample.zip (Google Drive link in github.com/rijulgupta-dm/deepfake-satellite-images, CC BY-NC-SA 4.0) | dm_aer | ~/data/dm_aer: 1000 StyleGAN2 fakes (512x512) + 1000 Million-AID reals (137-4432 px, 367 below 384 px), official validation sample, used whole |
| 2026-11-25T08:04:44-06:00 | `./run_extra_seeds_v2.sh` (replaces run_extra_seeds.sh, which was only waiting) | queue | after radio_s42 + full cross-eval: cross_dmaer (full_s42), cross_web_{fmow,rsfake,uw,dmaer} (web_s42), SimLBR re-run with DM-AER to simlbr_dmaer_metrics.json / simlbr_nolbr_dmaer_metrics.json, then seeds 43/44 full+plain. run_web_seeds.sh restarted with wait pattern covering the v2 script |
| 2026-11-25T08:12:47-06:00 | edit eval_cross_dataset.py, simlbr_baseline.py | dm_aer | DM-AER resolution confound (fakes all 512x512, reals 137-4432 px): primary evaluation now size-matched (`--dmaer-matched`: the 494 reals with shorter side >= 512, every image centre-cropped to 512x512 before the 384 resize; 1000 fakes unchanged). All-reals run kept as a sensitivity check (cross_allreals_dmaer / cross_web_allreals_dmaer / SimLBR cross_dmaer_allreals). Decided by Joe before any DM-AER result existed |
| 2026-11-25T08:12:47-06:00 | restart `./run_extra_seeds_v2.sh` (was only waiting) | queue | picks up the matched DM-AER runs; training untouched |
| 2026-11-25T08:12:47-06:00 | edit setup_data.sh, setup_env.sh, go.sh, watch.py | scripts | setup_data.sh now fetches DM-AER sample (section 6; verified all datasets "ok"); setup_env.sh installs einops, open_clip_torch, tifffile; go.sh starts all three queue stages if none is running; watch.py lists seeds 43/44, web seeds and all cross-dataset files |
| 2026-11-25T08:25:42-06:00 | fix aggregate_runs.py | bug fix | it averaged every test_metrics.json under results_dinov3 by (checkpoint, variant), so smoke, verify_bf16 and old_model_rescored would have been merged into the "full" row of table.csv. Now only folders named <row>_s<seed> count, keyed by row name; skipped folders are printed. No table.csv had been written yet |
| 2026-11-25T08:25:42-06:00 | edit eval_cross_dataset.py | extras | new `--dataset overhead` (main test split: two-class + three-class scores, tampered-only IoU) and `--degrade jpeg:Q / scale:S` (perturbation at native resolution before the 384 resize). CPU dry run on 24 images OK |
| 2026-11-25T08:25:42-06:00 | new analyze_extras.py, make_qualitative.py | extras | two-class test comparison with SimLBR, RSFAKE per-generator, robustness table, bootstrap CIs (1000 resamples, seed 42), consistency check against test_metrics.json; qualitative figure with a fixed selection rule |
| 2026-11-25T08:25:42-06:00 | new train_norm.py, run_norm_check.sh | side experiment (Joe approved 2026-11-25) | fullnorm_s42 / webnorm_s42: identical training, encoder input normalised with the backbone's preprocessor_config mean/std (sat: 0.430,0.411,0.296 / 0.213,0.156,0.143; web: ImageNet). dinov3_forensics_train_v2.py NOT modified (encoder call patched at run time; verified on CPU against manual normalisation). Runs last, after web seeds |
| 2026-11-25T08:25:42-06:00 | restart `./run_extra_seeds_v2.sh`, start `./run_norm_check.sh` | queue | extras (14 test-set evaluations, figure, analysis) run in the GPU gap after the cross-dataset evaluations, before seeds 43/44 |
| 2026-11-25T08:36:20-06:00 | new simlbr_official.py; `pip install lightning wandb kornia torchmetrics` (no existing package changed) | simlbr_official (Joe's request) | SimLBR is now trained and evaluated with the authors' released code (~/third_party/SimLBR, unmodified checkout of ac90f86): their datasets/transform, model + LBR, training entry point with val_acc checkpoint selection, and evaluate_subset. Run-time adaptations only: (1) our data exposed in their AIGC folder layout by symlinks (TIFF and size-matched DM-AER crops written as PNG), (2) backbone weights from Hugging Face instead of the Meta download key, same CLS token, (3) our subset names appended to their list. Release-script hyperparameters (batch 200, 5 epochs), seeds 42/43/44, LBR and no-LBR, backbones sat and web. Their repo has no licence file, so it is cloned at run time and not copied into this repo. CPU dry run (8 images per class, 1 epoch) passed end to end; no real numbers yet |
| 2026-11-25T08:36:20-06:00 | restart `./run_extra_seeds_v2.sh` (was only waiting) | queue | official SimLBR runs added to the GPU gap before seeds 43/44; analyze_extras.py reports them next to the model's two-class scores |
| 2026-11-25T08:42:01-06:00 | edit analyze_extras.py, eval_cross_dataset.py (`--no-edge-gate`), run_web_seeds.sh, watch.py | significance (Joe's request) | exact McNemar test on three-class correctness of the same test images: web vs full, full vs plain, web vs plain, per seed, plus mean accuracy difference over seeds; raw p-values, not corrected for multiple comparisons. Needs per-image predictions, so the seven remaining full/plain/web checkpoints are re-scored on the test set after the web seeds finish. Code checked on synthetic predictions; plain checkpoint load checked on CPU (needs --no-edge-gate, a wrong call raises a key-mismatch error). run_web_seeds.sh restarted (was only waiting) |
| 2026-11-25T09:03:50-06:00 | edit run_extra_seeds_v2.sh (restarted; was only waiting), watch.py, README.md | resnetlr_s42 (Joe approved) | resnet_s42 at epoch 18: val acc 0.66, best val F1 0.544 (DINOv3 rows ~0.99 at that point), consistent with the backbone LR 2e-6 being far too low for a CNN. Added a second ResNet-50 row with LR_BACKBONE=1e-4 (existing switch; value fixed before seeing any result with it, not tuned), everything else identical, selection on validation as always. Both rows are reported. Runs after the evaluations, before seeds 43/44 (about 3 h) |
| 2026-11-25T09:13:06-06:00 | edit run_extra_seeds_v2.sh, run_web_seeds.sh, watch.py, go.sh; both waiting scripts restarted | queue order (Joe's request) | seeds 43/44 now run in the order web, full, plain (was full+plain, then web). Same runs and settings; only the order changed so the best seed-42 row gets its three seeds first. run_web_seeds.sh now finds the web seeds done and goes straight to the test-set re-scoring, significance tests and table rebuild |
| 2026-11-25T11:08:58-06:00 |  | resnet_s42 | exit=0, test_metrics.json written |
| 2026-11-25T11:08:58-06:00 | `SEED=42 BACKBONE=radio EVAL_SPLITS=val,test RUN_TAG=radio_s42 python dinov3_forensics_train_v2.py` | radio_s42 | started |
| 2026-11-25T11:16:44-06:00 | edit run_extra_seeds_v2.sh (waiter restarted; was only waiting), go.sh, watch.py, README.md; new run_laptop.sh; run_norm_check.sh waiter stopped | queue split (Joe's request) | plain_s43, plain_s44, fullnorm_s42, webnorm_s42 removed from this machine's queue; to run on Joe's laptop (RTX 5090, WSL) via run_laptop.sh, not yet started there. Same script and settings. When copied back: re-run the plain_s43/s44 test re-scoring + analyze_extras.py + aggregate_runs.py here; note different GPU in RESULTS.md and do not use their timing columns. Undo: EXTRA_ROWS="web full plain" and NORM_HERE=1 ./go.sh |
| 2026-11-25T11:34:39-06:00 |  | radio_s42 | exit=1, NO test_metrics.json (see logs/radio_s42.log) |
| 2026-11-25T11:34:39-06:00 | `python eval_cross_dataset.py --dataset fmow --root /home/adminuser/data/fmow-fake-small` | cross_fmow | started |
| 2026-11-25T11:34:57-06:00 |  | cross_fmow | exit=0, cross_fmow_metrics.json written |
| 2026-11-25T11:34:57-06:00 | `python eval_cross_dataset.py --dataset rsfake --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_rsfake | started |
| 2026-11-25T11:49:45-06:00 |  | cross_rsfake | exit=0, cross_rsfake_metrics.json written |
| 2026-11-25T11:49:45-06:00 | `python eval_cross_dataset.py --dataset uw --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_uw | started |
| 2026-11-25T12:01:39-06:00 |  | cross_uw | exit=0, cross_uw_metrics.json written |
| 2026-11-25T12:01:39-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-25T12:01:39-06:00] run_everything.sh finished (SEEDS=42 ROWS=full plain frozen resnet web radio)
| 2026-11-25T12:02:31-06:00 | `python eval_cross_dataset.py --dataset dmaer --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out cross_dmaer_metrics.json --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_dmaer | started |
| 2026-11-25T12:04:51-06:00 |  | cross_dmaer | exit=0, cross_dmaer_metrics.json written |
| 2026-11-25T12:04:51-06:00 | `python eval_cross_dataset.py --dataset dmaer --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out cross_allreals_dmaer_metrics.json --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample` | cross_allreals_dmaer | started |
| 2026-11-25T12:07:55-06:00 |  | cross_allreals_dmaer | exit=0, cross_allreals_dmaer_metrics.json written |
| 2026-11-25T12:07:55-06:00 | `python eval_cross_dataset.py --dataset fmow --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_fmow_metrics.json --root /home/adminuser/data/fmow-fake-small` | cross_web_fmow | started |
| 2026-11-25T12:08:14-06:00 |  | cross_web_fmow | exit=0, cross_web_fmow_metrics.json written |
| 2026-11-25T12:08:14-06:00 | `python eval_cross_dataset.py --dataset rsfake --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_rsfake_metrics.json --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_web_rsfake | started |
| 2026-11-25T12:17:14-06:00 | (diagnosis, no command) | radio_s42 | FAILED by GPU memory, not a script bug: first CUDA out-of-memory right after epoch 3, when the backbone unfreezes (FREEZE_WARMUP_EPOCHS=3); process at 15.18 GiB of 15.57 GiB; 5994 batches skipped, divergence guard stopped the run in epoch 5. No test_metrics.json. Partial folder kept untouched. Not re-run with changed settings (rule 2/5). Option: run unchanged on the laptop (24 GB). |
| 2026-11-25T12:19:28-06:00 | `python eval_cross_dataset.py --dataset overhead --root ~/data/overhead/test_jpg/test --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt full_s42/best_checkpoint.pt --out overhead_full_metrics.json` (run early, by hand, to validate the evaluation code; queue will skip it) | overhead_full | started |
| 2026-11-25T12:26:58-06:00 |  | cross_web_rsfake | exit=0, cross_web_rsfake_metrics.json written |
| 2026-11-25T12:26:58-06:00 | `python eval_cross_dataset.py --dataset uw --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_uw_metrics.json --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_web_uw | started |
| 2026-11-25T12:28:12-06:00 | `./run_frozen_cross.sh` (new) | cross_frozen_{fmow,rsfake,uw,dmaer} | Joe approved: score existing frozen_s42 checkpoint zero-shot on the four external sets. Evaluation only. Runs alongside the queue's evaluation steps (no timed training row is running). |
| 2026-11-25T12:28:12-06:00 | `./run_fewshot.sh` (new: fewshot_adapt.py, FEWSHOT_PROTOCOL.md) | fewshot | Joe approved the few-shot adaptation experiment. Protocol written before any number. Dry run on 80 images per set (scratch dir, not a result) only to test the code; protocol not changed afterwards. New experiment outside the original handoff: list under deviations in RESULTS.md. |
| 2026-11-25T12:28:12-06:00 | (correction) | fmow-fake-small | set has 120 scored images (60 real, 60 manipulated: train+val); an earlier file count of 90/80 per split included the mask/auxiliary files. |
| 2026-11-25T12:28:12-06:00 | (note) | laptop | Joe is installing Linux on the laptop (not WSL). RADIO (radio_s42 out of memory here) to be run there unchanged, with run_laptop.sh rows. |
| 2026-11-25T12:29:58-06:00 |  | overhead_full | exit=0. Evaluation-code check PASSED: three-class accuracy 0.9000, macro-F1 0.8975, mean IoU 0.6746, confusion matrix identical to full_s42/test_metrics.json. Two-class on the test set: AUC 0.9656, accuracy 0.9121. |
| 2026-11-25T12:30:03-06:00 | `python eval_cross_dataset.py --dataset fmow --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt frozen_s42/best_checkpoint.pt --out cross_frozen_fmow_metrics.json --root /home/adminuser/data/fmow-fake-small` | cross_frozen_fmow | started |
| 2026-11-25T12:30:36-06:00 |  | cross_frozen_fmow | exit=0, cross_frozen_fmow_metrics.json written |
| 2026-11-25T12:30:36-06:00 | `python eval_cross_dataset.py --dataset rsfake --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt frozen_s42/best_checkpoint.pt --out cross_frozen_rsfake_metrics.json --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_frozen_rsfake | started |
| 2026-11-25T12:51:57-06:00 |  | cross_web_uw | exit=0, cross_web_uw_metrics.json written |
| 2026-11-25T12:51:57-06:00 | `python eval_cross_dataset.py --dataset dmaer --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_dmaer_metrics.json --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_web_dmaer | started |
| 2026-11-25T12:56:53-06:00 |  | cross_web_dmaer | exit=0, cross_web_dmaer_metrics.json written |
| 2026-11-25T12:56:53-06:00 | `python eval_cross_dataset.py --dataset dmaer --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_allreals_dmaer_metrics.json --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample` | cross_web_allreals_dmaer | started |
| 2026-11-25T13:01:41-06:00 |  | cross_frozen_rsfake | exit=0, cross_frozen_rsfake_metrics.json written |
| 2026-11-25T13:01:41-06:00 | `python eval_cross_dataset.py --dataset uw --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt frozen_s42/best_checkpoint.pt --out cross_frozen_uw_metrics.json --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_frozen_uw | started |
| 2026-11-25T13:03:08-06:00 |  | cross_web_allreals_dmaer | exit=0, cross_web_allreals_dmaer_metrics.json written |
| 2026-11-25T13:03:08-06:00 | `python simlbr_baseline.py --out simlbr_dmaer_metrics.json` | simlbr_dmaer | started |
| 2026-11-25T13:04:02-06:00 |  | simlbr_dmaer | exit=0 |
| 2026-11-25T13:04:02-06:00 | `python simlbr_baseline.py --no-lbr --out simlbr_nolbr_dmaer_metrics.json` | simlbr_nolbr_dmaer | started |
| 2026-11-25T13:04:10-06:00 |  | simlbr_nolbr_dmaer | exit=0 |
| 2026-11-25T13:04:10-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out robust_full_jpeg90_metrics.json --degrade jpeg:90` | robust_full_jpeg90 | started |
| 2026-11-25T13:14:18-06:00 |  | robust_full_jpeg90 | exit=0, robust_full_jpeg90_metrics.json written |
| 2026-11-25T13:14:18-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out robust_full_jpeg70_metrics.json --degrade jpeg:70` | robust_full_jpeg70 | started |
| 2026-11-25T13:24:27-06:00 |  | robust_full_jpeg70 | exit=0, robust_full_jpeg70_metrics.json written |
| 2026-11-25T13:24:27-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out robust_full_jpeg50_metrics.json --degrade jpeg:50` | robust_full_jpeg50 | started |
| 2026-11-25T13:26:37-06:00 |  | cross_frozen_uw | exit=0, cross_frozen_uw_metrics.json written |
| 2026-11-25T13:26:37-06:00 | `python eval_cross_dataset.py --dataset dmaer --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt frozen_s42/best_checkpoint.pt --out cross_frozen_dmaer_metrics.json --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_frozen_dmaer | started |
| 2026-11-25T13:27:34-06:00 | rsync to laptop joe@joe-Alienware-18-Area-51-AA18250 (RTX 5090 Laptop GPU 24 GB, Ubuntu 26.04.1, Python 3.14.4, driver 595.99.02) over Tailscale | laptop setup | copied ~/data/overhead (26,304 files, count verified), HF weights (DINOv3 sat + web, C-RADIOv4, resnet50) so no dataset or weight is re-downloaded; git clone of grsl-experiments at 808df9b into ~/grsl; venv ~/Desktop/.venv with the workstation's exact pip freeze (torch 2.14.1+cu126), install in progress. Nothing started yet. |
| 2026-11-25T13:31:32-06:00 |  | cross_frozen_dmaer | exit=0, cross_frozen_dmaer_metrics.json written |
| 2026-11-25T13:32:52-06:00 |  | robust_full_jpeg50 | exit=0, robust_full_jpeg50_metrics.json written |
| 2026-11-25T13:32:52-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out robust_full_jpeg30_metrics.json --degrade jpeg:30` | robust_full_jpeg30 | started |
| 2026-11-25T13:33:18-06:00 | laptop: `pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130` | laptop env | DEVIATION between machines: the workstation's torch 2.14.1+cu126 has no kernels for the RTX 5090 (sm_120; "no kernel image is available"). Laptop uses the same torch/torchvision versions built for CUDA 13.0; every other package is the workstation's exact pip freeze (pip check clean). GPU test passed (bf16 autocast conv). |
| 2026-11-25T13:35:47-06:00 | laptop: `setsid nohup ./run_laptop.sh` (code da6820f); `sudo powerprofilesctl set performance` | laptop queue | radio_s42 started 13:33 on the laptop. First ~1.5 min of epoch 1 ran in the 'balanced' power profile (GPU 54 W, 2.2 it/s); switched to 'performance' (168 W, 7+ it/s). Laptop timings (s/epoch, ms/image) are NOT comparable with the workstation rows. Desktop session logged out at Joe's request; run survived (linger on). |
| 2026-11-25T13:37:41-06:00 |  | robust_full_jpeg30 | exit=0, robust_full_jpeg30_metrics.json written |
| 2026-11-25T13:37:41-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out robust_full_scale0.75_metrics.json --degrade scale:0.75` | robust_full_scale0.75 | started |
| 2026-11-25T13:39:33-06:00 | edit watch.py, go.sh; laptop: `echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo` | dashboard / laptop | watch.py shows sensors of both machines and the laptop's live training (ssh); go.sh also starts run_laptop.sh if the laptop is idle. Laptop CPU turbo off at Joe's request (CPU 95 -> 59 C; training speed not lower: 8.9 it/s). radio_s42 row is now a laptop row in the dashboard. |
| 2026-11-25T13:42:30-06:00 |  | robust_full_scale0.75 | exit=0, robust_full_scale0.75_metrics.json written |
| 2026-11-25T13:42:30-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out robust_full_scale0.5_metrics.json --degrade scale:0.5` | robust_full_scale0.5 | started |
| 2026-11-25T13:44:28-06:00 | laptop: `git stash; git pull; git stash pop` (my mistake), fixed with `tail -c +1 --pid=28422 -f /proc/28422/fd/1 > logs/radio_s42.log` | radio_s42 log (laptop) | The stash replaced the tracked, open log file, so the running process kept writing to a deleted inode (training itself unaffected; GPU stayed busy). The complete log was recovered from the process's file descriptor and is being followed into logs/radio_s42.log until the run ends. Rule from now on: never stash/checkout tracked files on the laptop while a run is live; update single files with `git checkout origin/grsl-experiments -- <file>`. Also: laptop platform profile performance -> balanced-performance at Joe's request (fan noise): fans ~4400 -> ~2700 rpm, GPU still 168 W. |
| 2026-11-25T13:45:23-06:00 | `python fewshot_adapt.py extract` | fewshot | started |
| 2026-11-25T13:49:18-06:00 |  | robust_full_scale0.5 | exit=0, robust_full_scale0.5_metrics.json written |
| 2026-11-25T13:49:18-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out overhead_web_metrics.json ` | overhead_web | started |
| 2026-11-25T13:59:24-06:00 |  | overhead_web | exit=0, overhead_web_metrics.json written |
| 2026-11-25T13:59:25-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out robust_web_jpeg90_metrics.json --degrade jpeg:90` | robust_web_jpeg90 | started |
| 2026-11-25T14:02:45-06:00 | `hf download geodf/gdd --repo-type dataset --include 'test/*' '*.csv' README.md --local-dir ~/data/gdd` | gdd | GDD (Arndt et al., arXiv 2611.31487, ORNL; licence CC BY 4.0 on the dataset card; gated: auto). Joe approved adding it as a fifth external test set. Test splits + CSVs only (no training on it). |
| 2026-11-25T14:09:32-06:00 |  | robust_web_jpeg90 | exit=0, robust_web_jpeg90_metrics.json written |
| 2026-11-25T14:09:32-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out robust_web_jpeg70_metrics.json --degrade jpeg:70` | robust_web_jpeg70 | started |
| 2026-11-25T14:19:16-06:00 |  | gdd | downloaded: 8250 files, 7.8G |
| 2026-11-25T14:19:16-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split indomain --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out cross_gdd_indomain_metrics.json` | cross_gdd_indomain | started |
| 2026-11-25T14:19:36-06:00 |  | robust_web_jpeg70 | exit=0, robust_web_jpeg70_metrics.json written |
| 2026-11-25T14:19:36-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out robust_web_jpeg50_metrics.json --degrade jpeg:50` | robust_web_jpeg50 | started |
| 2026-11-25T14:26:25-06:00 |  | cross_gdd_indomain | exit=0, cross_gdd_indomain_metrics.json written |
| 2026-11-25T14:26:25-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split indomain --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_gdd_indomain_metrics.json` | cross_web_gdd_indomain | started |
| 2026-11-25T14:27:12-06:00 |  | fewshot extract | exit=0 |
| 2026-11-25T14:27:42-06:00 | `python fewshot_adapt.py analyse` | fewshot | exit=0 (fewshot_metrics.json) |
| 2026-11-25T14:31:32-06:00 |  | cross_web_gdd_indomain | exit=0, cross_web_gdd_indomain_metrics.json written |
| 2026-11-25T14:31:32-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split indomain --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/frozen_s42/best_checkpoint.pt --out cross_frozen_gdd_indomain_metrics.json` | cross_frozen_gdd_indomain | started |
| 2026-11-25T14:31:58-06:00 |  | robust_web_jpeg50 | exit=0, robust_web_jpeg50_metrics.json written |
| 2026-11-25T14:31:58-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out robust_web_jpeg30_metrics.json --degrade jpeg:30` | robust_web_jpeg30 | started |
| 2026-11-25T14:36:12-06:00 |  | cross_frozen_gdd_indomain | exit=0, cross_frozen_gdd_indomain_metrics.json written |
| 2026-11-25T14:36:12-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split crossgenerator --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out cross_gdd_crossgenerator_metrics.json` | cross_gdd_crossgenerator | started |
| 2026-11-25T14:41:53-06:00 |  | robust_web_jpeg30 | exit=0, robust_web_jpeg30_metrics.json written |
| 2026-11-25T14:41:53-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out robust_web_scale0.75_metrics.json --degrade scale:0.75` | robust_web_scale0.75 | started |
| 2026-11-25T14:42:25-06:00 |  | cross_gdd_crossgenerator | exit=0, cross_gdd_crossgenerator_metrics.json written |
| 2026-11-25T14:42:25-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split crossgenerator --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_gdd_crossgenerator_metrics.json` | cross_web_gdd_crossgenerator | started |
| 2026-11-25T14:48:51-06:00 |  | cross_web_gdd_crossgenerator | exit=0, cross_web_gdd_crossgenerator_metrics.json written |
| 2026-11-25T14:48:51-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split crossgenerator --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/frozen_s42/best_checkpoint.pt --out cross_frozen_gdd_crossgenerator_metrics.json` | cross_frozen_gdd_crossgenerator | started |
| 2026-11-25T14:51:36-06:00 |  | robust_web_scale0.75 | exit=0, robust_web_scale0.75_metrics.json written |
| 2026-11-25T14:51:36-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out robust_web_scale0.5_metrics.json --degrade scale:0.5` | robust_web_scale0.5 | started |
| 2026-11-25T14:55:06-06:00 |  | cross_frozen_gdd_crossgenerator | exit=0, cross_frozen_gdd_crossgenerator_metrics.json written |
| 2026-11-25T14:55:06-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split crossdomain --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out cross_gdd_crossdomain_metrics.json` | cross_gdd_crossdomain | started |
| 2026-11-25T15:01:31-06:00 |  | cross_gdd_crossdomain | exit=0, cross_gdd_crossdomain_metrics.json written |
| 2026-11-25T15:01:31-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split crossdomain --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_gdd_crossdomain_metrics.json` | cross_web_gdd_crossdomain | started |
| 2026-11-25T15:01:32-06:00 |  | robust_web_scale0.5 | exit=0, robust_web_scale0.5_metrics.json written |
| 2026-11-25T15:01:41-06:00 | `python make_qualitative.py` | qualitative | exit=0 |
| 2026-11-25T15:01:41-06:00 | `python simlbr_official.py --backbone sat` | simlbr_official_sat | started |
| 2026-11-25T15:05:56-06:00 |  | cross_web_gdd_crossdomain | exit=0, cross_web_gdd_crossdomain_metrics.json written |
| 2026-11-25T15:05:56-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split crossdomain --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/frozen_s42/best_checkpoint.pt --out cross_frozen_gdd_crossdomain_metrics.json` | cross_frozen_gdd_crossdomain | started |
| 2026-11-25T15:10:38-06:00 |  | cross_frozen_gdd_crossdomain | exit=0, cross_frozen_gdd_crossdomain_metrics.json written |
| 2026-11-25T15:11:01-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split localization --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s42/best_checkpoint.pt --out cross_gdd_localization_metrics.json` | cross_gdd_localization | started |
| 2026-11-25T15:13:28-06:00 |  | cross_gdd_localization | exit=0, cross_gdd_localization_metrics.json written |
| 2026-11-25T15:13:28-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split localization --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s42/best_checkpoint.pt --out cross_web_gdd_localization_metrics.json` | cross_web_gdd_localization | started |
| 2026-11-25T15:15:35-06:00 |  | cross_web_gdd_localization | exit=0, cross_web_gdd_localization_metrics.json written |
| 2026-11-25T15:15:35-06:00 | `python eval_cross_dataset.py --dataset gdd --gdd-split localization --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/frozen_s42/best_checkpoint.pt --out cross_frozen_gdd_localization_metrics.json` | cross_frozen_gdd_localization | started |
| 2026-11-25T15:17:47-06:00 |  | cross_frozen_gdd_localization | exit=0, cross_frozen_gdd_localization_metrics.json written |
| 2026-11-25T16:06:04-06:00 |  | simlbr_official_sat | exit=0 |
| 2026-11-25T16:06:04-06:00 | `python simlbr_official.py --backbone web` | simlbr_official_web | started |
| 2026-11-25T17:02:30-06:00 | laptop: logind drop-in /etc/systemd/logind.conf.d/90-lid-ignore.conf (HandleLidSwitch*=ignore), reloaded logind | laptop-setup | lid close no longer suspends |
| 2026-11-25T17:04:30-06:00 | watch.py: results panel now shows GDD zero-shot rows, few-shot AUC table, official SimLBR line | dashboard | ok |
| 2026-11-25T17:04:46-06:00 |  | simlbr_official_web | exit=0 |
| 2026-11-25T17:05:22-06:00 | `python analyze_extras.py` | analysis_extras | exit=0 (analysis_extras.json) |
| 2026-11-25T17:05:22-06:00 | `SEED=42 BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s42 python dinov3_forensics_train_v2.py` | resnetlr_s42 | started |
| 2026-11-25T20:25:25-06:00 |  | resnetlr_s42 | exit=0, test_metrics.json written |
| 2026-11-25T20:25:25-06:00 | `SEED=43 DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m EVAL_SPLITS=val,test RUN_TAG=web_s43 python dinov3_forensics_train_v2.py` | web_s43 | started |
| 2026-11-25T21:14:20-06:00 | watch.py: reloads itself when updated, results panel on the laptop too (pulls result files every 10 min), overall finish time in header | dashboard | ok |
| 2026-11-25T23:44:42-06:00 | rsync GDD test data + full_s42/web_s42 checkpoints to laptop | fewshot_gdd | started |
| 2026-11-25T23:46:57-06:00 | `python fewshot_adapt.py extract --sets gdd` (on laptop, alongside its training run) | fewshot_gdd | started |
| 2026-11-25T23:48:15-06:00 | laptop: platform profile performance, CPU turbo back on (Joe asked for more speed) | laptop-setup | applied during radio_s42 epoch 43 |
| 2026-11-25T23:49:21-06:00 | laptop: CPU hit 101 C with turbo on (plus GDD extraction workers); turbo off again -> 61 C; profile left at performance | laptop-setup | ok |
| 2026-11-25T23:53:54-06:00 | laptop: platform profile quiet (Joe asked for silent mode); turbo stays off | laptop-setup | applied during radio_s42 |
| 2026-11-26T00:02:00-06:00 |  | fewshot_gdd extract | exit=0 |
| 2026-11-26T00:02:44-06:00 | `python fewshot_adapt.py analyse --sets gdd` | fewshot_gdd | exit=0 (fewshot_gdd_metrics.json) |
| 2026-11-26T00:02:53-06:00 | `python tsne_features.py` | tsne | exit=0 (tsne_features.png) |
| 2026-11-26T05:14:16-06:00 |  | web_s43 | exit=0, test_metrics.json written |
| 2026-11-26T05:14:16-06:00 | `SEED=44 DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m EVAL_SPLITS=val,test RUN_TAG=web_s44 python dinov3_forensics_train_v2.py` | web_s44 | started |
| 2026-11-26T09:52:21-06:00 | `python analyze_cross_detail.py` | cross_detail | exit=0 (cross_detail.json, CROSS_DETAIL.md) |
| 2026-11-26T09:52:21-06:00 | `./run_fewshot_seeds.sh` (waits for the training queue) | fewshot_seeds | queued |
| 2026-11-26T09:56:26-06:00 | `python make_supplement.py` + headless browser print | supplement | SUPPLEMENT.md/.html/.pdf draft (7 pages, 8 tables, 2 figures) |
| 2026-11-26T10:04:40-06:00 | supplement draft removed from the repo (kept locally, git-ignored) at Joe's request | supplement | ok |
| 2026-11-26T10:39:47-06:00 | supplement files and generator moved to supplement/ (run: python supplement/make_supplement.py) | supplement | ok |
| 2026-11-26T14:04:02-06:00 |  | web_s44 | exit=0, test_metrics.json written |
| 2026-11-26T14:04:02-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-26T14:04:02-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=web)
| 2026-11-26T14:04:02-06:00 | `SEED=43 EVAL_SPLITS=val,test RUN_TAG=full_s43 python dinov3_forensics_train_v2.py` | full_s43 | started |
| 2026-11-26T14:33:29-06:00 | `python fewshot_adapt.py extract --sets own` (on laptop, alongside its training run) | zeroshot_extras | started |
| 2026-11-26T14:34:37-06:00 | watch.py/go.sh: few-shot table with GDD rows, status marks for new analyses, comments updated | dashboard | ok |
| 2026-11-26T15:05:22-06:00 |  | zeroshot_extras extract | exit=0 |
| 2026-11-26T15:05:47-06:00 | `python zeroshot_extras.py` | zeroshot_extras | exit=0 (zeroshot_extras_metrics.json) |
| 2026-11-26 20:30 | laptop: platform_profile quiet -> performance (turbo stays off), user request | – | done |
| 2026-11-26 20:34 | `./run_extra_seeds_v2b.sh` started (waiter): plain_s44 moved from laptop to workstation, runs after full_s44; stage 3 + few-shot seeds wait for it | plain_s44 | waiting |
| 2026-11-26 20:35 | go.sh + watch.py updated: plain_s44 shown as a workstation row, go.sh starts run_extra_seeds_v2b.sh after a restart | dashboard | done |
| 2026-11-26 20:37 | watch.py: more colour and detail (own-test summary by variant, validation history, fan/RAM/disk, finish clock times) | dashboard | done |
| 2026-11-26 20:53 | watch.py: optional panel for the separate spacecraft project (~/spacecraft/status.py), workstation only | dashboard | done |
| 2026-11-26 21:28 | `python supplement/make_supplement.py`: label-free section added (Tables S9-S11), PDF/DOCX rebuilt (git-ignored) | supplement | done |
| 2026-11-26T22:54:41-06:00 |  | full_s43 | exit=0, test_metrics.json written |
| 2026-11-26T22:54:41-06:00 | `SEED=44 EVAL_SPLITS=val,test RUN_TAG=full_s44 python dinov3_forensics_train_v2.py` | full_s44 | started |
| 2026-11-26 23:01 | laptop: platform_profile performance -> quiet (Joe's request); night light 1700 K on both machines | – | done |
| 2026-11-26 23:23 | watch.py: views split into tabs (Runs, Own test, Zero-shot, Few-shot, Spacecraft), keys 1-5 / arrows / q | dashboard | done |
| 2026-11-26 23:26 | laptop: dashboard window restarted with tabs; brightness 100%, night light off (Joe's request) | – | done |
| 2026-11-26 23:38 | laptop: speaker muted, brightness 1/255, night light on 1700 K (Joe's request) | – | done |
| 2026-11-27 01:20 | laptop: AlienFX case-light commands sent to AW-ELC controller (hidraw, not saved to firmware); GTA trailer screen saver installed then removed (its idle check was wrong: it never closed on input) | laptop settings | removed; `~/gta` kept, autostart entry disabled |
| 2026-11-27 01:36 | laptop: `~/stocks/saver.sh` (world stock market screen saver: local page + Yahoo Finance data, shows after 2 min idle, closes on input, speaks headline with Piper voice in `~/stocks/venv`, CPU only) added to autostart; speaker unmuted at 40% only while it shows; brightness 180/255 | laptop settings | running; training queue unaffected (2 queue scripts alive) |
| 2026-11-27 01:36 | workstation: night light set to 1000 K | workstation settings | done |
| 2026-11-27 01:37 | read-only analysis of zero-shot results started (agents; no GPU, nothing written to the repo) after Joe asked to improve zero-shot accuracy | zeroshot_improve | running |
| 2026-11-27 01:45 | laptop: stock screen saver stopped and removed from autostart at Joe's request; speaker muted again; files left in `~/stocks` (incl. voice models) and `~/gta`, nothing running from them | laptop settings | done; training queue unaffected |
| 2026-11-27 01:47 | laptop: brightness back to 1/255 (Joe's request) | laptop settings | done |
| 2026-11-27 | zero-shot analysis (agents, read-only, no GPU) finished: `ZEROSHOT_SOTA.md` (literature) and `ZEROSHOT_IMPROVE_PLAN_DRAFT.md` (diagnosis + draft protocol + adversarial review) saved, uncommitted; nothing run, awaiting Joe | zeroshot_improve | draft only |
| 2026-11-27T07:48:26-06:00 |  | full_s44 | exit=0, test_metrics.json written |
| 2026-11-27T07:48:26-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T07:48:26-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=full)
| 2026-11-27T07:48:40-06:00 | `SKIP_SETUP=1 SEEDS=44 ROWS=plain ./run_everything.sh` (moved from laptop) | plain_s44 | queue started |
| 2026-11-27T07:48:40-06:00 | `SEED=44 ABL_EDGE_GATE=0 ABL_BOUNDARY=0 ABL_DEEP_SUP=0 EVAL_SPLITS=val,test RUN_TAG=plain_s44 python dinov3_forensics_train_v2.py` | plain_s44 | started |
| 2026-11-27 10:20 | `~/HANDOFF.md` written (state of both projects, rules, machines, open decisions) for a new chat | handoff | done |
| 2026-11-27 10:38 | handoff log moved to `~/Desktop/Handoff_Logs/v_1_2026-11-27.md`; later handoffs are saved there as v_2, v_3 ... (Joe's request) | handoff | done |
| 2026-11-27 10:40 | `ZEROSHOT_AUG_PROTOCOL.md` written (one run `fullaug_s42` with resampling-factor augmentation, switch `AUG_SHIFT=1`); no code written, nothing run; scratch diagnostics copied to `~/grsl_zeroshot_diag_scratch_2026-11-27/` | zeroshot_aug | protocol draft, run approval pending (Joe) |
| 2026-11-27 10:42 | Joe decided to skip the resampling-augmentation run (`fullaug_s42`); `ZEROSHOT_AUG_PROTOCOL.md` marked NOT RUN; no code written, nothing run; the letter keeps the zero-shot results as they are | zeroshot_aug | skipped |
| 2026-11-27 13:43 | laptop address removed from go.sh, watch.py, run_fewshot_gdd.sh, run_zeroshot_extras.sh (Joe: "remove that address", "push the GRSL fix"); now read from `~/.config/grsl/laptop` outside the repo; pushed as 4324913 together with 0b78b5e (seed 43/44 logs); older commits still contain it (history not rewritten) | github | pushed |
| 2026-11-27 13:54 | new chat resumed from handoff v_10; queues checked (workstation plain_s44 epoch 38/50, laptop plain_s43 epoch 47/50). Supplement Table S8 fix: column "3-class accuracy" was reading `accuracy_argmax` (two-class, 0.9124 for full_s42) instead of `three_class.accuracy` (0.9000, equal to test_metrics.json); 1-line change in supplement/make_supplement.py, supplement not rebuilt yet. Noted: `scale0.75` rows equal the clean rows because test images are 512x512 and 0.75 x 512 = 384 = model input | supplement | fixed, uncommitted |
| 2026-11-27 13:58 | paper: result figures added (confusion matrices, robustness, few-shot curves, t-SNE, qualitative; seed-42 files only) by `~/Desktop/GRSL_Letter/make_figs_results.py` -> `figs_v3/`; `OJCS_Paper_v_3.docx` built and rendered (8 pages); build_letter.py: figures wider than a column span the page | paper | done |
| 2026-11-27 14:04 | paper: acknowledgment added (dataset creators; DINOv3, C-RADIO, SimLBR authors; no funding, no AI-tools sentence); `OJCS_Paper_v_4.docx` built and rendered | paper | done |
| 2026-11-27 14:05 | paper: main-findings sentences in the introduction written from table.csv, cross_*_metrics.json, fewshot*_metrics.json (decoder clause still pending on plain seeds 43/44); `OJCS_Paper_v_5.docx` built and rendered | paper | done |
| 2026-11-27 15:04 | laptop queue finished 14:59; `radio_s42`, `fullnorm_s42`, `webnorm_s42`, `plain_s43` copied to `~/results_dinov3` (rsync --ignore-existing, 103 files, sha256 equal on both machines); workstation's failed radio folder renamed `radio_s42_FAILED_workstation`, its log `logs/radio_s42_FAILED_workstation.log`; laptop logs, RUNLOG, table.csv and git state saved in `logs/laptop/`; the laptop's RUNLOG lines follow | laptop copy | done |
<!-- lines merged from the laptop's RUNLOG.md (verbatim, laptop clock) -->





















| 2026-11-25T13:33:31-06:00 | `./run_laptop.sh` on joe-Alienware-18-Area-51-AA18250 (NVIDIA GeForce RTX 5090 Laptop GPU) | laptop queue | started |
| 2026-11-25T13:33:34-06:00 | `SEED=42 BACKBONE=radio EVAL_SPLITS=val,test RUN_TAG=radio_s42 python dinov3_forensics_train_v2.py` | radio_s42 | started |
| 2026-11-26T02:24:44-06:00 |  | radio_s42 | exit=0, test_metrics.json written |
| 2026-11-26T02:24:45-06:00 |  | cross_fmow | SKIPPED: fmow-fake-small not accessible ([90mHint: set HF_DEBUG=1 as environment variable for full traceback.[0m) |
| 2026-11-26T02:24:45-06:00 | `python prepare_rsfake_subset.py --root /home/joe/data/rsfake --per-gen 500 --seed 42` | rsfake_subset | started |
| 2026-11-26T02:24:46-06:00 |  | cross_rsfake | SKIPPED: subset csv missing |
| 2026-11-26T02:24:46-06:00 |  | cross_uw | SKIPPED: /home/joe/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data not found |
| 2026-11-26T02:24:46-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-26T02:24:46-06:00] run_everything.sh finished (SEEDS=42 ROWS=radio)
| 2026-11-26T02:24:46-06:00 | `SEED=42 EVAL_SPLITS=val,test RUN_TAG=fullnorm_s42 python train_norm.py` | fullnorm_s42 | started |
| 2026-11-26T15:08:11-06:00 |  | fullnorm_s42 | exit=0, test_metrics.json written |
| 2026-11-26T15:08:11-06:00 | `SEED=42 DINOV3_CKPT=facebook/dinov3-vitl16-pretrain-lvd1689m EVAL_SPLITS=val,test RUN_TAG=webnorm_s42 python train_norm.py` | webnorm_s42 | started |
| 2026-11-26T20:34:41-06:00 | laptop: plain_s44 moved to the workstation (Joe: "move it"); skip_tags.txt + 1-line skip check in run_everything.sh | plain_s44 | will be skipped here |
| 2026-11-27T03:12:14-06:00 |  | webnorm_s42 | exit=0, test_metrics.json written |
| 2026-11-27T03:12:14-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv rebuilt with fullnorm/webnorm rows |
[2026-11-27T03:12:17-06:00] run_norm_check.sh finished
| 2026-11-27T03:12:17-06:00 | `SEED=43 ABL_EDGE_GATE=0 ABL_BOUNDARY=0 ABL_DEEP_SUP=0 EVAL_SPLITS=val,test RUN_TAG=plain_s43 python dinov3_forensics_train_v2.py` | plain_s43 | started |
| 2026-11-27T14:59:37-06:00 |  | plain_s43 | exit=0, test_metrics.json written |
| 2026-11-27T14:59:38-06:00 |  | cross_fmow | SKIPPED: fmow-fake-small not accessible ([90mHint: set HF_DEBUG=1 as environment variable for full traceback.[0m) |
| 2026-11-27T14:59:38-06:00 | `python prepare_rsfake_subset.py --root /home/joe/data/rsfake --per-gen 500 --seed 42` | rsfake_subset | started |
| 2026-11-27T14:59:38-06:00 |  | cross_rsfake | SKIPPED: subset csv missing |
| 2026-11-27T14:59:38-06:00 |  | cross_uw | SKIPPED: /home/joe/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data not found |
| 2026-11-27T14:59:38-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T14:59:38-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=plain)
| 2026-11-27T14:59:38-06:00 |  | laptop queue | finished |
<!-- end of laptop lines -->
| 2026-11-27T16:07:09-06:00 |  | plain_s44 | exit=0, test_metrics.json written |
| 2026-11-27T16:07:09-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T16:07:09-06:00] run_everything.sh finished (SEEDS=44 ROWS=plain)
| 2026-11-27T16:07:09-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T16:07:09-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=web)
| 2026-11-27T16:07:09-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/plain_s42/best_checkpoint.pt --out overhead_plain_s42_metrics.json --no-edge-gate` | overhead_plain_s42 | started |
| 2026-11-27T16:11:38-06:00 |  | overhead_plain_s42 | exit=0, overhead_plain_s42_metrics.json written |
| 2026-11-27T16:11:38-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out overhead_full_s43_metrics.json ` | overhead_full_s43 | started |
| 2026-11-27T16:16:26-06:00 |  | overhead_full_s43 | exit=0, overhead_full_s43_metrics.json written |
| 2026-11-27T16:16:26-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/plain_s43/best_checkpoint.pt --out overhead_plain_s43_metrics.json --no-edge-gate` | overhead_plain_s43 | started |
| 2026-11-27T16:20:53-06:00 |  | overhead_plain_s43 | exit=0, overhead_plain_s43_metrics.json written |
| 2026-11-27T16:20:53-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out overhead_web_s43_metrics.json ` | overhead_web_s43 | started |
| 2026-11-27T16:25:43-06:00 |  | overhead_web_s43 | exit=0, overhead_web_s43_metrics.json written |
| 2026-11-27T16:25:43-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out overhead_full_s44_metrics.json ` | overhead_full_s44 | started |
| 2026-11-27T16:30:31-06:00 |  | overhead_full_s44 | exit=0, overhead_full_s44_metrics.json written |
| 2026-11-27T16:30:31-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/plain_s44/best_checkpoint.pt --out overhead_plain_s44_metrics.json --no-edge-gate` | overhead_plain_s44 | started |
| 2026-11-27T16:35:01-06:00 |  | overhead_plain_s44 | exit=0, overhead_plain_s44_metrics.json written |
| 2026-11-27T16:35:01-06:00 | `python eval_cross_dataset.py --dataset overhead --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out overhead_web_s44_metrics.json ` | overhead_web_s44 | started |
| 2026-11-27T16:39:53-06:00 |  | overhead_web_s44 | exit=0, overhead_web_s44_metrics.json written |
| 2026-11-27T16:40:32-06:00 | `python analyze_extras.py` | analysis_extras | exit=0 (significance tests added) |
| 2026-11-27T16:40:32-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv rebuilt with web seeds |
| 2026-11-27T16:40:44-06:00 | `python fewshot_adapt.py extract --sets seeds` | fewshot_seeds | started |
| 2026-11-27T17:05:48-06:00 |  | fewshot_seeds extract | exit=0 |
| 2026-11-27T17:06:54-06:00 | `python fewshot_adapt.py analyse --sets seeds` | fewshot_seeds | exit=0 (fewshot_seeds_metrics.json) |
| 2026-11-27 17:14 | queues ended 17:06 without error (plain_s44, re-scoring of all seeds, McNemar tests, table.csv, few-shot seeds 43/44). `python make_results_md.py` -> `RESULTS.md` (new script; every number read from result files). Paper: `~/Desktop/GRSL_Letter/OJCS_Paper_v_6.docx` (11 pages; all results, discussion, conclusion filled from the files through `paper_results_oj.py`; Fig. 2 now from test_metrics.json; figures in `figs_v6/`). Supplement regenerated (new title; robustness table shows three-class accuracy), copy `supplement_draft/Supplement_v_2_2026-11-27.docx`, old files kept in `supplement/old_2026-11-26/`. `python update_readme_results.py` re-run (README results block; not committed, not pushed). `go.sh` / `watch.py` stopped on the workstation at Joe's request; laptop off since 15:05 | results, paper | done; nothing committed |
| 2026-11-27 17:15 | handoff log `~/Desktop/Handoff_Logs/v_11_2026-11-27.md` written (final state of the first paper, spacecraft planning state) | handoff | done |
| 2026-11-27 17:30 | supplement finished: DRAFT line removed, Tables S12-S15 added (few-shot over training seeds, per-seed results, McNemar per seed, input normalisation) in `supplement/make_supplement.py`; `supplement_draft/Supplement_v_3_2026-11-27.docx` (12 pages, rendered, checked). Paper `OJCS_Paper_v_7.docx`: one sentence at the start of Results pointing to the supplementary material | supplement, paper | done |
| 2026-11-27 17:37 | final results committed and pushed to origin/grsl-experiments at Joe's word ("pushed the latest results to github", read as an instruction; remote was still at b9a896f); `logs/laptop/` and `supplement/old_*/` kept off GitHub through .gitignore | github | pushed |
| 2026-11-27 18:40 | review comments (pasted by Joe) checked against the result files; paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_8.docx` (11 pages, rendered, checked): frozen-backbone sentence in V-B corrected (IoU 0.462 is above both ResNet-50 rows), seed-42 versus three-seed few-shot values stated in V-F and under Table VII, repeated parameter counts removed from IV-E, fmow-fake-small count explained (local copy: train and val folders, 30 real + 30 manipulated each, 120 images, no duplicate files). Edits in `paper_results_oj.py` and `paper_content_oj.py`; no result changed. Open for Joe: Dong Xu affiliation, footnote placeholders, branch name | paper | done |
| 2026-11-27 19:40 | second set of review comments (pasted by Joe): paper reframed as an empirical study, text only, no result changed: abstract, introduction (model called a baseline), new discussion paragraph on when fine-tuning pays (numbers read from analysis_extras.json, table.csv, fewshot_metrics.json), limitations (single generator family as likely cause of zero-shot failure; no forensic localizer or LoRA baseline). `OJCS_Paper_v_10.docx` (11 pages, rendered, checked); v_9 is the same except one overstated clause ("usable mask") and should not be used. Title unchanged, waits for Joe. No new experiment run | paper | done |
| 2026-11-27 19:35 | correction: the two lines above were written at about 18:35 and 19:32, not 18:40 and 19:40. Joe gave `OJCS_Paper_v_10_review.docx` (direct edit of the Word file: new title "... A Benchmark Study", abstract and introduction reframed, footnote filled: corresponding author R. Chadha, Joe at the Institute for Data Science and Informatics, D. Xu at the Health Informatics Institute, University of South Florida). Merged into `paper_content_oj.py` / `paper_results_oj.py`; `OJCS_Paper_v_11.docx` (11 pages, rendered, checked) equals the review file in text except two additions kept from v_10: discussion paragraph on when fine-tuning pays, two limitation sentences. Still pending: manuscript date, Xu's ZIP and e-mail | paper | done |
| 2026-11-27T19:38:58-06:00 | `./run_seed_evals.sh` | seed evals | started |
| 2026-11-27T19:38:58-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_fmow_metrics.json --dataset fmow --root /home/adminuser/data/fmow-fake-small` | cross_s43_fmow | started |
| 2026-11-27T19:39:22-06:00 |  | cross_s43_fmow | exit=0, cross_s43_fmow_metrics.json written |
| 2026-11-27T19:39:22-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_rsfake_metrics.json --dataset rsfake --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_s43_rsfake | started |
| 2026-11-27T19:54:16-06:00 |  | cross_s43_rsfake | exit=0, cross_s43_rsfake_metrics.json written |
| 2026-11-27T19:54:16-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_uw_metrics.json --dataset uw --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_s43_uw | started |
| 2026-11-27T20:06:17-06:00 |  | cross_s43_uw | exit=0, cross_s43_uw_metrics.json written |
| 2026-11-27T20:06:17-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_s43_dmaer | started |
| 2026-11-27T20:08:39-06:00 |  | cross_s43_dmaer | exit=0, cross_s43_dmaer_metrics.json written |
| 2026-11-27T20:08:39-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_allreals_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample` | cross_s43_allreals_dmaer | started |
| 2026-11-27T20:11:43-06:00 |  | cross_s43_allreals_dmaer | exit=0, cross_s43_allreals_dmaer_metrics.json written |
| 2026-11-27T20:11:43-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_gdd_indomain_metrics.json --dataset gdd --gdd-split indomain --root /home/adminuser/data/gdd` | cross_s43_gdd_indomain | started |
| 2026-11-27T20:14:03-06:00 |  | cross_s43_gdd_indomain | exit=0, cross_s43_gdd_indomain_metrics.json written |
| 2026-11-27T20:14:03-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_gdd_crossgenerator_metrics.json --dataset gdd --gdd-split crossgenerator --root /home/adminuser/data/gdd` | cross_s43_gdd_crossgenerator | started |
| 2026-11-27T20:17:06-06:00 |  | cross_s43_gdd_crossgenerator | exit=0, cross_s43_gdd_crossgenerator_metrics.json written |
| 2026-11-27T20:17:06-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_gdd_crossdomain_metrics.json --dataset gdd --gdd-split crossdomain --root /home/adminuser/data/gdd` | cross_s43_gdd_crossdomain | started |
| 2026-11-27T20:20:10-06:00 |  | cross_s43_gdd_crossdomain | exit=0, cross_s43_gdd_crossdomain_metrics.json written |
| 2026-11-27T20:20:10-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out cross_s43_gdd_localization_metrics.json --dataset gdd --gdd-split localization --root /home/adminuser/data/gdd` | cross_s43_gdd_localization | started |
| 2026-11-27T20:21:46-06:00 |  | cross_s43_gdd_localization | exit=0, cross_s43_gdd_localization_metrics.json written |
| 2026-11-27T20:21:46-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_fmow_metrics.json --dataset fmow --root /home/adminuser/data/fmow-fake-small` | cross_web_s43_fmow | started |
| 2026-11-27T20:22:07-06:00 |  | cross_web_s43_fmow | exit=0, cross_web_s43_fmow_metrics.json written |
| 2026-11-27T20:22:07-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_rsfake_metrics.json --dataset rsfake --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_web_s43_rsfake | started |
| 2026-11-27T20:36:52-06:00 |  | cross_web_s43_rsfake | exit=0, cross_web_s43_rsfake_metrics.json written |
| 2026-11-27T20:36:52-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_uw_metrics.json --dataset uw --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_web_s43_uw | started |
| 2026-11-27T20:48:48-06:00 |  | cross_web_s43_uw | exit=0, cross_web_s43_uw_metrics.json written |
| 2026-11-27T20:48:48-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_web_s43_dmaer | started |
| 2026-11-27T20:51:09-06:00 |  | cross_web_s43_dmaer | exit=0, cross_web_s43_dmaer_metrics.json written |
| 2026-11-27T20:51:09-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_allreals_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample` | cross_web_s43_allreals_dmaer | started |
| 2026-11-27T20:54:13-06:00 |  | cross_web_s43_allreals_dmaer | exit=0, cross_web_s43_allreals_dmaer_metrics.json written |
| 2026-11-27T20:54:13-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_gdd_indomain_metrics.json --dataset gdd --gdd-split indomain --root /home/adminuser/data/gdd` | cross_web_s43_gdd_indomain | started |
| 2026-11-27T20:56:33-06:00 |  | cross_web_s43_gdd_indomain | exit=0, cross_web_s43_gdd_indomain_metrics.json written |
| 2026-11-27T20:56:33-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_gdd_crossgenerator_metrics.json --dataset gdd --gdd-split crossgenerator --root /home/adminuser/data/gdd` | cross_web_s43_gdd_crossgenerator | started |
| 2026-11-27T20:59:37-06:00 |  | cross_web_s43_gdd_crossgenerator | exit=0, cross_web_s43_gdd_crossgenerator_metrics.json written |
| 2026-11-27T20:59:37-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_gdd_crossdomain_metrics.json --dataset gdd --gdd-split crossdomain --root /home/adminuser/data/gdd` | cross_web_s43_gdd_crossdomain | started |
| 2026-11-27T21:02:41-06:00 |  | cross_web_s43_gdd_crossdomain | exit=0, cross_web_s43_gdd_crossdomain_metrics.json written |
| 2026-11-27T21:02:41-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out cross_web_s43_gdd_localization_metrics.json --dataset gdd --gdd-split localization --root /home/adminuser/data/gdd` | cross_web_s43_gdd_localization | started |
| 2026-11-27T21:04:17-06:00 |  | cross_web_s43_gdd_localization | exit=0, cross_web_s43_gdd_localization_metrics.json written |
| 2026-11-27T21:04:17-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_fmow_metrics.json --dataset fmow --root /home/adminuser/data/fmow-fake-small` | cross_s44_fmow | started |
| 2026-11-27T21:04:37-06:00 |  | cross_s44_fmow | exit=0, cross_s44_fmow_metrics.json written |
| 2026-11-27T21:04:37-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_rsfake_metrics.json --dataset rsfake --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_s44_rsfake | started |
| 2026-11-27T21:19:22-06:00 |  | cross_s44_rsfake | exit=0, cross_s44_rsfake_metrics.json written |
| 2026-11-27T21:19:22-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_uw_metrics.json --dataset uw --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_s44_uw | started |
| 2026-11-27T21:31:20-06:00 |  | cross_s44_uw | exit=0, cross_s44_uw_metrics.json written |
| 2026-11-27T21:31:20-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_s44_dmaer | started |
| 2026-11-27T21:33:41-06:00 |  | cross_s44_dmaer | exit=0, cross_s44_dmaer_metrics.json written |
| 2026-11-27T21:33:41-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_allreals_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample` | cross_s44_allreals_dmaer | started |
| 2026-11-27T21:36:45-06:00 |  | cross_s44_allreals_dmaer | exit=0, cross_s44_allreals_dmaer_metrics.json written |
| 2026-11-27T21:36:45-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_gdd_indomain_metrics.json --dataset gdd --gdd-split indomain --root /home/adminuser/data/gdd` | cross_s44_gdd_indomain | started |
| 2026-11-27T21:39:05-06:00 |  | cross_s44_gdd_indomain | exit=0, cross_s44_gdd_indomain_metrics.json written |
| 2026-11-27T21:39:05-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_gdd_crossgenerator_metrics.json --dataset gdd --gdd-split crossgenerator --root /home/adminuser/data/gdd` | cross_s44_gdd_crossgenerator | started |
| 2026-11-27T21:42:08-06:00 |  | cross_s44_gdd_crossgenerator | exit=0, cross_s44_gdd_crossgenerator_metrics.json written |
| 2026-11-27T21:42:08-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_gdd_crossdomain_metrics.json --dataset gdd --gdd-split crossdomain --root /home/adminuser/data/gdd` | cross_s44_gdd_crossdomain | started |
| 2026-11-27T21:45:12-06:00 |  | cross_s44_gdd_crossdomain | exit=0, cross_s44_gdd_crossdomain_metrics.json written |
| 2026-11-27T21:45:12-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out cross_s44_gdd_localization_metrics.json --dataset gdd --gdd-split localization --root /home/adminuser/data/gdd` | cross_s44_gdd_localization | started |
| 2026-11-27T21:46:48-06:00 |  | cross_s44_gdd_localization | exit=0, cross_s44_gdd_localization_metrics.json written |
| 2026-11-27T21:46:48-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_fmow_metrics.json --dataset fmow --root /home/adminuser/data/fmow-fake-small` | cross_web_s44_fmow | started |
| 2026-11-27T21:47:07-06:00 |  | cross_web_s44_fmow | exit=0, cross_web_s44_fmow_metrics.json written |
| 2026-11-27T21:47:07-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_rsfake_metrics.json --dataset rsfake --root /home/adminuser/data/rsfake --subset-csv /home/adminuser/data/rsfake/subset_seed42_n500.csv --per-gen 500` | cross_web_s44_rsfake | started |
| 2026-11-27T22:01:52-06:00 |  | cross_web_s44_rsfake | exit=0, cross_web_s44_rsfake_metrics.json written |
| 2026-11-27T22:01:52-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_uw_metrics.json --dataset uw --root /home/adminuser/data/uw_fake_sat/anti-deepfake/anti-deepfake-data and code/data` | cross_web_s44_uw | started |
| 2026-11-27T22:13:47-06:00 |  | cross_web_s44_uw | exit=0, cross_web_s44_uw_metrics.json written |
| 2026-11-27T22:13:47-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample --dmaer-matched` | cross_web_s44_dmaer | started |
| 2026-11-27T22:16:08-06:00 |  | cross_web_s44_dmaer | exit=0, cross_web_s44_dmaer_metrics.json written |
| 2026-11-27T22:16:08-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_allreals_dmaer_metrics.json --dataset dmaer --root /home/adminuser/data/dm_aer/DM-AER-DeepFake-V1-sample` | cross_web_s44_allreals_dmaer | started |
| 2026-11-27T22:19:11-06:00 |  | cross_web_s44_allreals_dmaer | exit=0, cross_web_s44_allreals_dmaer_metrics.json written |
| 2026-11-27T22:19:11-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_gdd_indomain_metrics.json --dataset gdd --gdd-split indomain --root /home/adminuser/data/gdd` | cross_web_s44_gdd_indomain | started |
| 2026-11-27T22:21:31-06:00 |  | cross_web_s44_gdd_indomain | exit=0, cross_web_s44_gdd_indomain_metrics.json written |
| 2026-11-27T22:21:31-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_gdd_crossgenerator_metrics.json --dataset gdd --gdd-split crossgenerator --root /home/adminuser/data/gdd` | cross_web_s44_gdd_crossgenerator | started |
| 2026-11-27T22:24:35-06:00 |  | cross_web_s44_gdd_crossgenerator | exit=0, cross_web_s44_gdd_crossgenerator_metrics.json written |
| 2026-11-27T22:24:35-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_gdd_crossdomain_metrics.json --dataset gdd --gdd-split crossdomain --root /home/adminuser/data/gdd` | cross_web_s44_gdd_crossdomain | started |
| 2026-11-27T22:26:48-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:26:48-06:00] run_everything.sh finished (SEEDS=42 ROWS=full plain frozen resnet web radio)
| 2026-11-27T22:27:39-06:00 |  | cross_web_s44_gdd_crossdomain | exit=0, cross_web_s44_gdd_crossdomain_metrics.json written |
| 2026-11-27T22:27:39-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out cross_web_s44_gdd_localization_metrics.json --dataset gdd --gdd-split localization --root /home/adminuser/data/gdd` | cross_web_s44_gdd_localization | started |
| 2026-11-27T22:29:16-06:00 |  | cross_web_s44_gdd_localization | exit=0, cross_web_s44_gdd_localization_metrics.json written |
| 2026-11-27T22:29:16-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out robust_full_s43_jpeg90_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:90` | robust_full_s43_jpeg90 | started |
| 2026-11-27T22:29:25-06:00 | `python analyze_extras.py` | analysis_extras | exit=0 (analysis_extras.json) |
| 2026-11-27T22:29:25-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:29:25-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=web)
| 2026-11-27T22:29:25-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:29:25-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=full)
| 2026-11-27T22:29:28-06:00 | `SKIP_SETUP=1 SEEDS=44 ROWS=plain ./run_everything.sh` (moved from laptop) | plain_s44 | queue started |
| 2026-11-27T22:29:28-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:29:28-06:00] run_everything.sh finished (SEEDS=44 ROWS=plain)
| 2026-11-27 22:30 | GitHub branch `grsl-experiments` renamed to `ojcs-experiments` at Joe's word (name chosen by Joe): c9b584d (74d4581 plus the branch name in README.md, watch.py, run_zeroshot_extras.sh, run_fewshot_gdd.sh) pushed as the new branch, old branch deleted on GitHub; merged into local master; local commit 54446cc (protocols) is NOT pushed. Paper source `letter_content.py` updated, takes effect in the next paper version | github | renamed |
| 2026-11-27T22:30:49-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:30:49-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=web)
| 2026-11-27T22:31:27-06:00 | `python analyze_extras.py` | analysis_extras | exit=0 (significance tests added) |
| 2026-11-27T22:31:27-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv rebuilt with web seeds |
| 2026-11-27 22:34 | TruFor on the test split prepared, NOT run: `TRUFOR_PROTOCOL.md` and `trufor_eval.py` committed locally (94137c2, not pushed); authors' code cloned to `~/external/trufor_src/TruFor` (commit ae54475, licence: non-profit research use with attribution, not redistributed); weights `~/external/trufor_src/weights/trufor.pth.tar` (zip MD5 7bee48f3476c75616c3c5721ab256ff8, equal to the published value); yacs installed to `~/external/trufor_env` (training environment unchanged). The smoke test was blocked by the permission check, so no TruFor code has been executed and no number exists; waits for Joe. Extra training seeds (frozen, ResNet-50, C-RADIO) and LoRA: not started, wait for Joe | trufor | prepared |
| 2026-11-27T22:34:05-06:00 |  | robust_full_s43_jpeg90 | exit=0, robust_full_s43_jpeg90_metrics.json written |
| 2026-11-27T22:34:05-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out robust_full_s43_jpeg70_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:70` | robust_full_s43_jpeg70 | started |
| 2026-11-27T22:38:51-06:00 |  | robust_full_s43_jpeg70 | exit=0, robust_full_s43_jpeg70_metrics.json written |
| 2026-11-27T22:38:51-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out robust_full_s43_jpeg50_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:50` | robust_full_s43_jpeg50 | started |
| 2026-11-27T22:43:37-06:00 |  | robust_full_s43_jpeg50 | exit=0, robust_full_s43_jpeg50_metrics.json written |
| 2026-11-27T22:43:37-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out robust_full_s43_jpeg30_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:30` | robust_full_s43_jpeg30 | started |
| 2026-11-27T22:48:24-06:00 |  | robust_full_s43_jpeg30 | exit=0, robust_full_s43_jpeg30_metrics.json written |
| 2026-11-27T22:48:24-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out robust_full_s43_scale0.75_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.75` | robust_full_s43_scale0.75 | started |
| 2026-11-27T22:48:35-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:48:35-06:00] run_everything.sh finished (SEEDS=42 ROWS=full plain frozen resnet web radio)
| 2026-11-27T22:51:12-06:00 | `python analyze_extras.py` | analysis_extras | exit=0 (analysis_extras.json) |
| 2026-11-27T22:51:12-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:51:12-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=web)
| 2026-11-27T22:51:12-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:51:12-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=full)
| 2026-11-27T22:51:16-06:00 | `SKIP_SETUP=1 SEEDS=44 ROWS=plain ./run_everything.sh` (moved from laptop) | plain_s44 | queue started |
| 2026-11-27T22:51:16-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:51:16-06:00] run_everything.sh finished (SEEDS=44 ROWS=plain)
| 2026-11-27T22:52:37-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-27T22:52:37-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=web)
| 2026-11-27T22:53:13-06:00 |  | robust_full_s43_scale0.75 | exit=0, robust_full_s43_scale0.75_metrics.json written |
| 2026-11-27T22:53:13-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s43/best_checkpoint.pt --out robust_full_s43_scale0.5_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.5` | robust_full_s43_scale0.5 | started |
| 2026-11-27T22:53:15-06:00 | `python analyze_extras.py` | analysis_extras | exit=0 (significance tests added) |
| 2026-11-27T22:53:15-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv rebuilt with web seeds |
| 2026-11-27T22:58:01-06:00 |  | robust_full_s43_scale0.5 | exit=0, robust_full_s43_scale0.5_metrics.json written |
| 2026-11-27T22:58:01-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out robust_web_s43_jpeg90_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:90` | robust_web_s43_jpeg90 | started |
| 2026-11-27T23:02:51-06:00 |  | robust_web_s43_jpeg90 | exit=0, robust_web_s43_jpeg90_metrics.json written |
| 2026-11-27T23:02:51-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out robust_web_s43_jpeg70_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:70` | robust_web_s43_jpeg70 | started |
| 2026-11-27T23:07:39-06:00 |  | robust_web_s43_jpeg70 | exit=0, robust_web_s43_jpeg70_metrics.json written |
| 2026-11-27T23:07:39-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out robust_web_s43_jpeg50_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:50` | robust_web_s43_jpeg50 | started |
| 2026-11-27T23:12:28-06:00 |  | robust_web_s43_jpeg50 | exit=0, robust_web_s43_jpeg50_metrics.json written |
| 2026-11-27T23:12:28-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out robust_web_s43_jpeg30_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:30` | robust_web_s43_jpeg30 | started |
| 2026-11-27T23:17:16-06:00 |  | robust_web_s43_jpeg30 | exit=0, robust_web_s43_jpeg30_metrics.json written |
| 2026-11-27T23:17:16-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out robust_web_s43_scale0.75_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.75` | robust_web_s43_scale0.75 | started |
| 2026-11-27T23:22:04-06:00 |  | robust_web_s43_scale0.75 | exit=0, robust_web_s43_scale0.75_metrics.json written |
| 2026-11-27T23:22:04-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s43/best_checkpoint.pt --out robust_web_s43_scale0.5_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.5` | robust_web_s43_scale0.5 | started |
| 2026-11-27T23:26:52-06:00 |  | robust_web_s43_scale0.5 | exit=0, robust_web_s43_scale0.5_metrics.json written |
| 2026-11-27T23:26:52-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out robust_full_s44_jpeg90_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:90` | robust_full_s44_jpeg90 | started |
| 2026-11-27T23:31:41-06:00 |  | robust_full_s44_jpeg90 | exit=0, robust_full_s44_jpeg90_metrics.json written |
| 2026-11-27T23:31:41-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out robust_full_s44_jpeg70_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:70` | robust_full_s44_jpeg70 | started |
| 2026-11-27T23:36:29-06:00 |  | robust_full_s44_jpeg70 | exit=0, robust_full_s44_jpeg70_metrics.json written |
| 2026-11-27T23:36:29-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out robust_full_s44_jpeg50_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:50` | robust_full_s44_jpeg50 | started |
| 2026-11-27T23:41:18-06:00 |  | robust_full_s44_jpeg50 | exit=0, robust_full_s44_jpeg50_metrics.json written |
| 2026-11-27T23:41:18-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out robust_full_s44_jpeg30_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:30` | robust_full_s44_jpeg30 | started |
| 2026-11-27T23:46:06-06:00 |  | robust_full_s44_jpeg30 | exit=0, robust_full_s44_jpeg30_metrics.json written |
| 2026-11-27T23:46:06-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out robust_full_s44_scale0.75_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.75` | robust_full_s44_scale0.75 | started |
| 2026-11-27 23:47 | Joe: "go for trufor". Six-image smoke test passed (scratchpad, not a result) after allowing the checkpoint's numpy scalar types in the safe loader; `run_trufor.sh` waits for the seed evaluation queue, then runs `trufor_eval.py` on the 3,150 test images | trufor | queued |
| 2026-11-27T23:50:56-06:00 |  | robust_full_s44_scale0.75 | exit=0, robust_full_s44_scale0.75_metrics.json written |
| 2026-11-27T23:50:56-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-sat493m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/full_s44/best_checkpoint.pt --out robust_full_s44_scale0.5_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.5` | robust_full_s44_scale0.5 | started |
| 2026-11-27T23:55:47-06:00 |  | robust_full_s44_scale0.5 | exit=0, robust_full_s44_scale0.5_metrics.json written |
| 2026-11-27T23:55:47-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out robust_web_s44_jpeg90_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:90` | robust_web_s44_jpeg90 | started |
| 2026-11-27 23:56 | Joe: "start the experiments for grsl", "update go.sh". `run_review_seeds.sh` started (waits for run_seed_evals.sh and run_trufor.sh, then frozen_s43, resnet_s43, frozen_s44, resnet_s44, resnetlr_s43, resnetlr_s44; existing switches only). go.sh (replaced as a new file, the running copy was not edited) and watch.py know stage 5 and the seed evaluations. Not started: C-RADIO seeds (need a larger GPU), LoRA (model change, waits for Joe's choices) | stage 5 | queued |
| 2026-11-28T00:00:38-06:00 |  | robust_web_s44_jpeg90 | exit=0, robust_web_s44_jpeg90_metrics.json written |
| 2026-11-28T00:00:38-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out robust_web_s44_jpeg70_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:70` | robust_web_s44_jpeg70 | started |
| 2026-11-28 00:03 | third machine Calypso (2 x RTX A6000 48 GB, a lab machine with a shared account, used with its owner's permission; other users' jobs hold part of the GPU memory) set up for the extra seeds. Everything of ours is in `~/joe_grsl` there: code and `data/overhead` copied by rsync (26,304 files, count equal), conda env `joe_grsl` with the workstation's package versions (torch 2.14.1+cu126, transformers 5.22.2, timm 1.0.31, python 3.14). `run_calypso.sh` written, not started; go.sh starts it only when ~/joe_grsl/READY exists. go.sh / watch.py: Calypso step and panel; its address is kept outside the repository | calypso | set up |
| 2026-11-28T00:05:28-06:00 |  | robust_web_s44_jpeg70 | exit=0, robust_web_s44_jpeg70_metrics.json written |
| 2026-11-28T00:05:28-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out robust_web_s44_jpeg50_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:50` | robust_web_s44_jpeg50 | started |
| 2026-11-28T00:10:17-06:00 |  | robust_web_s44_jpeg50 | exit=0, robust_web_s44_jpeg50_metrics.json written |
| 2026-11-28T00:10:17-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out robust_web_s44_jpeg30_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade jpeg:30` | robust_web_s44_jpeg30 | started |
| 2026-11-28T00:15:15-06:00 |  | robust_web_s44_jpeg30 | exit=0, robust_web_s44_jpeg30_metrics.json written |
| 2026-11-28T00:15:15-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out robust_web_s44_scale0.75_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.75` | robust_web_s44_scale0.75 | started |
| 2026-11-28T00:20:06-06:00 |  | robust_web_s44_scale0.75 | exit=0, robust_web_s44_scale0.75_metrics.json written |
| 2026-11-28T00:20:06-06:00 | `python eval_cross_dataset.py --dinov3 facebook/dinov3-vitl16-pretrain-lvd1689m --ckpt /home/adminuser/results_dinov3/vit/dinov3-vitl16-pretrain-lvd1689m/web_s44/best_checkpoint.pt --out robust_web_s44_scale0.5_metrics.json --dataset overhead --root /home/adminuser/data/overhead/test_jpg/test --degrade scale:0.5` | robust_web_s44_scale0.5 | started |
| 2026-11-28T00:24:56-06:00 |  | robust_web_s44_scale0.5 | exit=0, robust_web_s44_scale0.5_metrics.json written |
| 2026-11-28T00:24:56-06:00 | `python fewshot_adapt.py extract --sets own_seeds` | zeroshot_extras_seeds | started |
| 2026-11-28 00:27 | Joe: "do the real runs now", "more queued jobs to calypso", "maximum power and speed". Calypso smoke test (radio, 1 epoch, tag smoke_calypso, not a result): training and validation finished without error (505 s/epoch), stopped during the final test evaluation at Joe's word. Started on Calypso (tmux): lane radio on GPU 0 (radio_s43, radio_s44), lane review on GPU 1 (frozen_s43, frozen_s44, resnetlr_s43, resnetlr_s44; DINOv3 and ResNet-50 weights copied from the workstation's cache, HF offline, no token on that machine). frozen_s43 was restarted once after a few minutes when the lane list changed (partial folder kept there as frozen_s43_ABORTED_restart_0128). Workstation: `run_review_seeds.sh` now resnet_s43, resnet_s44 only, waiting behind run_seed_evals.sh and run_trufor.sh. Runs on Calypso come from another GPU: accuracy numbers only, not timing. Calypso's own run log: ~/joe_grsl/home/grsl/RUNLOG.md there | extra seeds | running |
| 2026-11-28 00:34 | Joe: "can you speed up calypso". Both Calypso GPUs are at 95-99 % utilisation and about 295 W each, so nothing can be gained there without changing settings (not done). Measured there: radio 5.8 it/s, frozen 7.0 it/s (frozen is slower than on the workstation). Rebalanced instead: workstation takes resnet_s43/44 and resnetlr_s43/44 after TruFor; Calypso GPU 1 keeps frozen_s43/44 (frozen_s43 restarted once more, partial folder kept there as frozen_s43_ABORTED_restart_0140); Calypso GPU 0 radio_s43/44 untouched | extra seeds | rebalanced |
| 2026-11-28T00:37:38-06:00 |  | zeroshot_extras_seeds extract | exit=0 |
| 2026-11-28T00:38:06-06:00 | `python zeroshot_extras.py --seeds` | zeroshot_extras_seeds | exit=0 (zeroshot_extras_seeds_metrics.json) |
| 2026-11-28T00:38:06-06:00 |  | seed evals | finished |
| 2026-11-28T00:38:13-06:00 | `PYTHONPATH=~/external/trufor_env python trufor_eval.py` | trufor_overhead | started |
| 2026-11-28T00:39:50-06:00 |  | trufor_overhead | exit=0, trufor_overhead_metrics.json written |
| 2026-11-28T00:40:48-06:00 | `./run_review_seeds.sh` | review seeds | started (resnet, resnetlr; seeds 43 44) |
| 2026-11-28T00:40:48-06:00 | `SEED=43 BACKBONE=resnet50 EVAL_SPLITS=val,test RUN_TAG=resnet_s43 python dinov3_forensics_train_v2.py` | resnet_s43 | started |
| 2026-11-28 00:42 | queues ended without a FAILED line: seed-43/44 zero-shot (36 files), robustness (24), label-free (`zeroshot_extras_seeds_metrics.json`), held-out generator (`fewshot_logo_metrics.json`), TruFor (`trufor_overhead_metrics.json`, 92 s). `make_results_md.py` extended (three new sections) and `RESULTS.md` rebuilt with --force (597 lines; earlier sections unchanged). Paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_12.docx` (12 pages, rendered, checked): Table V with three-seed zero-shot AUC next to the seed-42 intervals, robustness and label-free text and Table VII as three-seed means, new Table VIII and paragraph (test generator left out of the shots), TruFor row in Table IV and paragraph, abstract / introduction / limitations / conclusion adjusted, branch name ojcs-experiments; every number read from the result files in `paper_results_oj.py`. Supplement not yet updated (still v_3). Table III unchanged until the extra seeds finish | results, paper | done |
| 2026-11-28 00:44 | pushed to origin/ojcs-experiments at Joe's word ("push to git"): now at a194647 (protocol commits 54446cc, 94137c2, 38c0c21, 735a6c8 unchanged; the four later local commits were folded into one after the private description of the third machine was removed from run_calypso.sh and from one RUNLOG line; remote history was not rewritten). Not pushed: paper and supplement drafts (kept off GitHub), the live training logs | github | pushed |
| 2026-11-28 02:16 | paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_14.docx` (12 pages, rendered, checked): v_12 plus one Discussion sentence (the detector as a primary filter; ms per image read from table.csv), at Joe's word. The number 13 was not used (Joe named v_14). Table III still has the single-run rows; the extra seeds are still training | paper | done |
| 2026-11-28T04:01:14-06:00 |  | resnet_s43 | exit=0, test_metrics.json written |
| 2026-11-28T04:01:14-06:00 | `SEED=44 BACKBONE=resnet50 EVAL_SPLITS=val,test RUN_TAG=resnet_s44 python dinov3_forensics_train_v2.py` | resnet_s44 | started |
| 2026-11-28T07:21:36-06:00 |  | resnet_s44 | exit=0, test_metrics.json written |
| 2026-11-28T07:21:36-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv written |
[2026-11-28T07:21:36-06:00] run_everything.sh finished (SEEDS=43 44 ROWS=resnet)
| 2026-11-28T07:21:36-06:00 | `SEED=43 BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s43 python dinov3_forensics_train_v2.py` | resnetlr_s43 | started |
- 2026-11-28 09:00 Joe says he changed the Tailscale SSH rule from check to accept, so the 12-hour browser approval for Calypso should no longer be needed (not yet confirmed by a fresh connection after noon). Status 09:00: resnet_s43, resnet_s44, frozen_s43 finished; resnetlr_s43, frozen_s44, radio_s43 running; radio on Calypso takes about 15.6 min/epoch wall time.
| 2026-11-28T10:13:57-06:00 | `hf download geodf/gdd --repo-type dataset --include 'train/*' --include 'val/*' --local-dir ~/data/gdd` | gdd train download | started (Joe 2026-11-28: 'yes second family, train on GDD'); test splits were already there |
| 2026-11-28T10:16:42-06:00 | commit 7a758b8 | LoRA + GDD training | protocols LORA_PROTOCOL.md and GDD_TRAIN_PROTOCOL.md committed before any run (Joe: 'yes lora and yes second family, train on GDD'). Training script: new switch LORA_RANK (default 0 = unchanged behaviour; diff touches 4 existing lines: best-checkpoint save/load go through helpers that are identical without adapters, encoder-freeze condition and freeze_backbone argument gain 'or LORA_RANK > 0'). CPU check: B=0 output bit-identical to plain model; no gradient on encoder weights; merged vs hooked max diff 1.4e-6. Local commit, not pushed. |
| 2026-11-28T10:16:53-06:00 | `git push origin master:ojcs-experiments` | push | at Joe's word ('and then push'): remote at bb4e4f8 |
| 2026-11-28T10:41:21-06:00 | rsync from Calypso, sha256 of 24 files equal | frozen_s43 | copied into ~/results_dinov3 (folder moved into place only after the checksum comparison); log and Calypso RUNLOG snapshot in logs/calypso/ |
| 2026-11-28T10:42:20-06:00 |  | resnetlr_s43 | exit=0, test_metrics.json written |
| 2026-11-28T10:42:20-06:00 | `SEED=44 BACKBONE=resnet50 LR_BACKBONE=1e-4 EVAL_SPLITS=val,test RUN_TAG=resnetlr_s44 python dinov3_forensics_train_v2.py` | resnetlr_s44 | started |
| 2026-11-28T10:48:10-06:00 | `python prepare_gdd_train.py --check` | gdd layout | GDD train + val download finished (exit 0). First check stopped: 238 file stems occur twice in the training set (fully-regenerated / inpaint-splice pairs, different images). prepare_gdd_train.py now lists both images of such a pair under a link <folder>__<file name> (476 train, 128 test images), nothing dropped; addendum in GDD_TRAIN_PROTOCOL.md written before any run. Check after the change: train 11,410 and test 2,000 images, 0 unreadable or mismatched, 0 empty masks. Not yet copied to Calypso |
| 2026-11-28T10:48:38-06:00 | `python aggregate_runs.py --csv table.csv` | aggregate | table.csv rebuilt with frozen_s43 and resnetlr_s43 (frozen 2 seeds, resnet 3, resnetlr 2, radio 1), for paper v_15 at Joe's word |
| 2026-11-28 10:50 | paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_15.docx` (12 pages, rendered, pages 4, 5, 8, 12 checked) at Joe's word, with the seeds finished so far: Table III frozen 2 seeds, ResNet-50 3 and 2, C-RADIO 1 (text, Table II, abstract, limitations follow the counts read from table.csv); Tables VII and VIII swapped so they are numbered in order of citation; frozen column of Table V labelled seed 42; RTX A6000 named in Implementation. Supplement `supplement_draft/Supplement_v_15_2026-11-28.docx` (14 pages, rendered, pages 1, 11, 12 checked; number 15 to match the paper): current title, Table S13 with the new seeds, new Tables S16 (zero-shot per seed), S17/S18 (held-out generator), S19 (TruFor). Still open for the next version: frozen_s44, resnetlr_s44, radio_s43/44, LoRA, GDD training | paper, supplement | done |
| 2026-11-28T10:50:57-06:00 | `git push origin master:ojcs-experiments` | push | at Joe's word ('push'): remote at f625bb4 |
| 2026-11-28T11:05:34-06:00 | `python real_anchor.py` | real_anchor | protocol REAL_ANCHOR_PROTOCOL.md and script (both supplied by Joe) committed before any number (Joe: 'approve it and time to run it'); started, CPU only |
| 2026-11-28T11:16:37-06:00 |  | real_anchor | first start stopped by me after 11 min (29 CPU threads, load 100, training slowed from 12.7 to 11.3 it/s); nothing had been written; restarted unchanged with 6 threads |
| 2026-11-28 11:16 | paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_17.docx` (12 pages, rendered, page 1 checked): Joe's OJCS_Paper_v_16.docx differs from v_15 only in the footnote (Xu's ZIP code and e-mail filled in by Joe); these were carried into paper_content_oj.py and v_17 built from the source; its text is identical to v_16. No new result in it | paper | done |
| 2026-11-28T11:16:58-06:00 | `git push origin master:ojcs-experiments` | push | at Joe's word ('and push to git'): remote at aa65ff6 |
| 2026-11-28 11:17 | supplement `supplement_draft/Supplement_v_17_2026-11-28.docx` (14 pages, rendered) at Joe's word, numbered to match paper v_17; rebuilt from the same result files, text identical to Supplement v_15 | supplement | done |
| 2026-11-28T11:26:19-06:00 | `python real_anchor.py` (6 threads) | real_anchor | exit=0, real_anchor_metrics.json written 11:25. Pre-declared headline (anchor_matched_filter, k = 64, AUC above the zero-shot AUC of the fine-tuned model): sat_orig on 5 of 8 sets, three-seed mean of full on 3 of 8 (fmow has no k = 64; same counts with the zero-shot AUC of the whole set or of the held-out half). Below the fixed rule of six, so by the protocol it is reported as a negative result, not as a contribution. Measured false-alarm rate at the nominal 5 % threshold, k = 64: 0.19-0.23 (sat_orig), 0.22-0.27 (full) |
| 2026-11-28 11:40 | paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_18.docx` (12 pages, rendered, page 9 checked) at Joe's word: v_17 plus one paragraph in Section V-G on the real-anchored matched filter as a negative result (numbers read from real_anchor_metrics.json; 5 and 3 of 8 sets, false-alarm rate 19-27 %), and the experiment named in Section IV-D, the first sentence of Results and the limitations. No reference added for the matched filter. Supplement `supplement_draft/Supplement_v_18_2026-11-28.docx` (15 pages, rendered, pages 12, 13 checked): new Section IX with Tables S20 (AUC by k, four feature sets) and S21 (the comparison fixed in advance, k = 64) | paper, supplement | done |
| 2026-11-28 11:43 | paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_19.docx` (12 pages, rendered, page 12 checked) at Joe's word: v_18 plus reference [37] (Robey et al., A CFAR adaptive matched filter detector, IEEE Trans. Aerosp. Electron. Syst. 28(1), 208-216, 1992, doi 10.1109/7.135446; details checked by web search) cited in the matched-filter paragraph | paper | done |
| 2026-11-28 11:51 | paper `~/Desktop/GRSL_Letter/OJCS_Paper_v_20.docx` (12 pages, rendered) after a review Joe pasted: abstract states that the frozen-backbone figure rests on two seeds; Section V-C titled 'Comparison With SimLBR and TruFor'; Discussion compares the best frozen-feature detector (web features, 0.798) with the fine-tuned model of the same backbone (0.981, was the satellite model's 0.966) | paper | done |
| 2026-11-28T11:51:36-06:00 | `git push origin master:ojcs-experiments` | push | at Joe's word ('push to git'): remote at b97477e (real_anchor_metrics.json, supplement builder, RUNLOG) |
| 2026-11-28 11:53 | supplement `supplement_draft/Supplement_v_20_2026-11-28.docx` (17 pages, rendered, page 15 checked) at Joe's word, to go with paper v_20: new Section X with three-seed tables S22 (zero-shot detection and false-alarm rates), S23 (robustness), S24 (label-free alternatives), pointers to them under Tables S1, S8 and S9-S11; 'Zhao et al.' in place of 'UW fake satellite'; American spelling; Table S13 caption no longer says 'finished so far' | supplement | done |
| 2026-11-28T12:07:40-06:00 | rsync to Calypso | gdd copy | started at Joe's word ('yes' to copying the GDD data there and starting gdd_s42 when GPU 1 is free): train, test/indomain, CSVs, prepare_gdd_train.py, run_calypso_gdd.sh, protocol |
- 2026-11-28 12:27 Joe: when all the work on Calypso is done, disconnect from Calypso. To do at the end (after the last run folder is copied back and checked): close our tmux sessions there and log the second Tailscale out; ask Joe before removing ~/joe_grsl and the conda env.
| 2026-11-28T12:47:41-06:00 | tmux joe_grsl_gdd1 on Calypso: `GPU=1 AFTER=joe_grsl_review SEEDS="42" run_calypso_gdd.sh` | gdd_s42 | GDD data on Calypso (21,070 files, names and sizes equal; layout lists identical to the workstation's by md5; read check 0 unreadable). Lane waits for frozen_s44. Seeds 43 and 44 go to the laptop (Joe turned it on), so they are not on Calypso's list |
| 2026-11-28T12:52:29-06:00 | tmux grsl_gdd on the laptop: `SEEDS="43 44" ./run_laptop_gdd.sh` | gdd_s43, gdd_s44 | GDD training set copied from the workstation (17,820 files, names and sizes equal; the dataset is gated and the laptop's account has no approval), layout lists identical by md5, read check 0 unreadable; started |
| 2026-11-28T12:53:35-06:00 | go.sh, watch.py | launcher, dashboard | updated at Joe's word: stage 6 (LoRA), Calypso GDD lane with seed 42 only, laptop GDD seeds 43/44 (run_laptop_gdd.sh, tmux grsl_gdd); dashboard shows the GDD seeds under the machine that runs them. go.sh checked with bash -n, watch.py --once runs |
