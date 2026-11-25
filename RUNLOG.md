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
