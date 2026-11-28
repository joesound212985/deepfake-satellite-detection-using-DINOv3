# Results
Generated 2026-11-28 00:42 by `make_results_md.py` from the files named in each section. Rows with one seed are single runs; the others are mean ± standard deviation over the listed seeds (sample standard deviation, as written by `aggregate_runs.py`).


## 1. Environment

| Run | GPU | torch | transformers | AMP | Best epoch (validation) | s / epoch | ms / test image |
|---|---|---|---|---|---|---|---|
| frozen_s42 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 50 | 310 | 88.2 |
| full_s42 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 49 | 595 | 88.8 |
| full_s43 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 36 | 598 | 88.5 |
| full_s44 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 39 | 601 | 88.6 |
| fullnorm_s42 | NVIDIA GeForce RTX 5090 Laptop GPU | 2.14.1+cu130 | 5.22.2 | bf16 | 30 | 836 | 309.7 |
| plain_s42 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 39 | 563 | 82.6 |
| plain_s43 | NVIDIA GeForce RTX 5090 Laptop GPU | 2.14.1+cu130 | 5.22.2 | bf16 | 27 | 792 | 128.7 |
| plain_s44 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 45 | 562 | 82.9 |
| radio_s42 | NVIDIA GeForce RTX 5090 Laptop GPU | 2.14.1+cu130 | 5.22.2 | bf16 | 34 | 865 | 147.3 |
| resnet_s42 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 50 | 225 | 39.2 |
| resnetlr_s42 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 31 | 222 | 39.2 |
| web_s42 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 44 | 596 | 88.6 |
| web_s43 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 17 | 596 | 88.4 |
| web_s44 | NVIDIA GeForce RTX 4080 SUPER | 2.14.1+cu126 | 5.22.2 | bf16 | 24 | 597 | 89.3 |
| webnorm_s42 | NVIDIA GeForce RTX 5090 Laptop GPU | 2.14.1+cu130 | 5.22.2 | bf16 | 40 | 811 | 135.1 |

Source: `test_metrics.json` of each run. Timings depend on the machine and on what else ran on it; runs on the laptop GPU are not comparable in time with runs on the workstation GPU.

Settings of the full model (`full_s42/test_metrics.json`, `config`): edge_gate = True, boundary_loss = True, deep_supervision = True, freeze_backbone = False, mask_aug = False, legacy_token_slice = False, lr_backbone = 2e-06, lr_decoder = 1e-05, lambda_seg = 0.8, w_bound = 0.25, ds_weight = 0.3, img_size = 384, batch_size = 3, epochs = 50, val_split = 0.1, val_seed = 42, best_metric = f1, freeze_warmup_epochs = 3, tta = True, amp_dtype = bf16, heavy_tta = False, trained_in_this_run = True.

| Model | GFLOPs | GMACs | Parameters (M) |
|---|---|---|---|
| full / frozen (DINOv3-sat ViT-L, edge gate) | 1162.4 | 581.2 | 309.43 |
| plain (DINOv3-sat ViT-L, no edge gate) | 1162.3 | 581.2 | 309.43 |
| web (DINOv3 web ViT-L) | 1162.4 | 581.2 | 309.43 |
| resnet50 | 834.6 | 417.3 | 29.81 |
| radio (C-RADIOv4-SO400M) | 1293.4 | 646.7 | 437.57 |

Source: `flops.json` (input 1x3x384x384; gflops: 2 per multiply-add (torch FlopCounterMode + sdpa handler); gmacs = gflops/2; uncounted ops = 0).


## 2. Test set

3150 test images: pristine 1511, tampered 888, fake 751; tamper masks on 888 tampered images (`full_s42/test_metrics.json`).


## 3. Main table (own test set)

| Row | Seeds | Accuracy | Macro-F1 | Balanced acc. | Tampered recall | IoU | Dice | MCC | Params (M) | ms / image |
|---|---|---|---|---|---|---|---|---|---|---|
| Full (satellite) | 42, 43, 44 | 0.903 ± 0.003 | 0.902 ± 0.004 | 0.900 ± 0.004 | 0.774 ± 0.020 | 0.671 ± 0.003 | 0.762 ± 0.003 | 0.742 ± 0.004 | 309.4 | 89 |
| Plain decoder (satellite) | 42, 43, 44 | 0.904 ± 0.003 | 0.902 ± 0.004 | 0.900 ± 0.004 | 0.771 ± 0.015 | 0.674 ± 0.001 | 0.766 ± 0.002 | 0.744 ± 0.002 | 309.4 | 98 |
| Web pretraining | 42, 43, 44 | 0.935 ± 0.002 | 0.935 ± 0.002 | 0.926 ± 0.002 | 0.805 ± 0.011 | 0.706 ± 0.005 | 0.795 ± 0.006 | 0.779 ± 0.005 | 309.4 | 89 |
| Frozen backbone | 42 | 0.643 | 0.512 | 0.593 | 0.015 | 0.462 | 0.572 | 0.536 | 309.4 | 88 |
| ResNet-50, backbone LR 2e-6 | 42 | 0.665 | 0.636 | 0.658 | 0.306 | 0.312 | 0.410 | 0.361 | 29.8 | 39 |
| ResNet-50, backbone LR 1e-4 | 42 | 0.762 | 0.752 | 0.753 | 0.514 | 0.357 | 0.442 | 0.413 | 29.8 | 39 |
| C-RADIOv4-SO400M | 42 | 0.835 | 0.819 | 0.826 | 0.572 | 0.587 | 0.679 | 0.660 | 437.6 | 147 |
| Full, backbone normalisation (side experiment) | 42 | 0.894 | 0.889 | 0.891 | 0.748 | 0.674 | 0.765 | 0.748 | 309.4 | 310 |
| Web, backbone normalisation (side experiment) | 42 | 0.932 | 0.931 | 0.921 | 0.780 | 0.724 | 0.812 | 0.796 | 309.4 | 135 |

Source: `table.csv`. IoU, Dice and MCC are computed per tampered image and averaged over the 888 tampered test images. `fullnorm` / `webnorm` feed the backbone its own mean / standard-deviation normalisation instead of inputs in [0, 1]; they were run on the laptop GPU, as were `radio_s42` and `plain_s43`.

| Run | Accuracy | Macro-F1 | Tampered recall | IoU | Best epoch | GPU |
|---|---|---|---|---|---|---|
| full_s42 | 0.9000 | 0.8975 | 0.7658 | 0.6746 | 49 | NVIDIA GeForce RTX 4080 SUPER |
| full_s43 | 0.9060 | 0.9059 | 0.7973 | 0.6687 | 36 | NVIDIA GeForce RTX 4080 SUPER |
| full_s44 | 0.9041 | 0.9021 | 0.7601 | 0.6695 | 39 | NVIDIA GeForce RTX 4080 SUPER |
| plain_s42 | 0.9010 | 0.8979 | 0.7646 | 0.6751 | 39 | NVIDIA GeForce RTX 4080 SUPER |
| plain_s43 | 0.9044 | 0.9026 | 0.7601 | 0.6723 | 27 | NVIDIA GeForce RTX 5090 Laptop GPU |
| plain_s44 | 0.9076 | 0.9068 | 0.7883 | 0.6732 | 45 | NVIDIA GeForce RTX 4080 SUPER |
| web_s42 | 0.9368 | 0.9372 | 0.8176 | 0.7021 | 44 | NVIDIA GeForce RTX 4080 SUPER |
| web_s43 | 0.9359 | 0.9359 | 0.7995 | 0.7035 | 17 | NVIDIA GeForce RTX 4080 SUPER |
| web_s44 | 0.9324 | 0.9327 | 0.7973 | 0.7122 | 24 | NVIDIA GeForce RTX 4080 SUPER |

Per-seed values, from each run's `test_metrics.json`.


### Per-class results and confusion matrices

| Row | Class | Precision | Recall | F1 | One-vs-rest AUC |
|---|---|---|---|---|---|
| Full (satellite) | pristine | 0.888 ± 0.006 | 0.938 ± 0.009 | 0.912 ± 0.001 | 0.967 ± 0.002 |
| Full (satellite) | tampered | 0.906 ± 0.011 | 0.774 ± 0.020 | 0.835 ± 0.007 | 0.956 ± 0.001 |
| Full (satellite) | fake | 0.931 ± 0.011 | 0.987 ± 0.001 | 0.958 ± 0.006 | 0.997 ± 0.000 |
| Web pretraining | pristine | 0.896 ± 0.003 | 0.985 ± 0.002 | 0.938 ± 0.002 | 0.980 ± 0.001 |
| Web pretraining | tampered | 0.972 ± 0.005 | 0.805 ± 0.011 | 0.880 ± 0.006 | 0.971 ± 0.002 |
| Web pretraining | fake | 0.985 ± 0.003 | 0.989 ± 0.007 | 0.987 ± 0.002 | 0.999 ± 0.000 |

Mean ± standard deviation over seeds 42, 43, 44, from each run's `test_metrics.json`.

| full_s42: true \ predicted | pristine | tampered | fake |
|---|---|---|---|
| pristine | 1414 | 66 | 31 |
| tampered | 173 | 680 | 35 |
| fake | 6 | 4 | 741 |

| web_s42: true \ predicted | pristine | tampered | fake |
|---|---|---|---|
| pristine | 1488 | 19 | 4 |
| tampered | 157 | 726 | 5 |
| fake | 10 | 4 | 737 |

| Tampered-area bucket (full_s42) | Images | Classified correctly | Mean IoU |
|---|---|---|---|
| tampered_small | 144 | 0.285 | 0.418 |
| tampered_large | 744 | 0.859 | 0.724 |

Small / large is split at a manipulated fraction of 0.0625 (`full_s42/test_metrics.json`, `area_buckets`).


## 4. Significance between rows

exact McNemar (two-sided binomial on discordant images), three-class correctness, per seed. 3 comparisons x up to 3 seeds; p-values are not corrected for multiple comparisons. Source: `analysis_extras.json`; per-image files `overhead_*_metrics.per_image.json` (re-evaluation with `eval_cross_dataset.py`, which can differ from `test_metrics.json` in the fourth decimal).

| Pair | Seed | Acc. A | Acc. B | Only A correct | Only B correct | p (McNemar) / accuracy difference A − B |
|---|---|---|---|---|---|---|
| web vs full | 42 | 0.9371 | 0.9000 | 171 | 54 | 2.4e-15 |
| web vs full | 43 | 0.9356 | 0.9054 | 155 | 60 | 7.1e-11 |
| web vs full | 44 | 0.9321 | 0.9038 | 151 | 62 | 9.4e-10 |
| web vs full | mean diff. |  |  |  |  | 0.0319 ± 0.0047 |
| full vs plain | 42 | 0.9000 | 0.9003 | 51 | 52 | 1 |
| full vs plain | 43 | 0.9054 | 0.9048 | 53 | 51 | 0.92 |
| full vs plain | 44 | 0.9038 | 0.9083 | 38 | 52 | 0.17 |
| full vs plain | mean diff. |  |  |  |  | -0.0014 ± 0.0027 |
| web vs plain | 42 | 0.9371 | 0.9003 | 171 | 55 | 4.9e-15 |
| web vs plain | 43 | 0.9356 | 0.9048 | 148 | 51 | 3.7e-12 |
| web vs plain | 44 | 0.9321 | 0.9083 | 142 | 67 | 2.3e-07 |
| web vs plain | mean diff. |  |  |  |  | 0.0305 ± 0.0065 |


## 5. Earlier checkpoint re-scored

`old_model_rescored/test_metrics.json`: accuracy 0.9095, macro-F1 0.9070, balanced accuracy 0.9018, IoU 0.5108, Dice 0.6271, MCC 0.6056 (config: legacy_token_slice = True). This checkpoint predates the validation split, so its selection is not comparable with the rows of the main table. The numbers of the earlier letter draft are not in any result file and are not repeated here.


## 6. Two-class scoring and SimLBR

| Model | AUC real vs manipulated | Accuracy @0.5 | Detection rate | False-alarm rate |
|---|---|---|---|---|
| Full (satellite), seed 42 | 0.966 [0.959, 0.971] | 0.912 [0.903, 0.921] | 0.893 | 0.068 |
| Web pretraining, seed 42 | 0.981 [0.976, 0.985] | 0.940 [0.931, 0.948] | 0.899 | 0.015 |
| SimLBR, satellite features, latent blending (3 seeds) | 0.710 ± 0.002 | 0.622 ± 0.002 | 0.794 ± 0.012 | 0.565 ± 0.017 |
| SimLBR, satellite features, no blending (3 seeds) | 0.741 ± 0.002 | 0.645 ± 0.003 | 0.812 ± 0.006 | 0.536 ± 0.009 |
| SimLBR, web features, latent blending (3 seeds) | 0.773 ± 0.003 | 0.697 ± 0.003 | 0.642 ± 0.009 | 0.242 ± 0.017 |
| SimLBR, web features, no blending (3 seeds) | 0.798 ± 0.002 | 0.722 ± 0.002 | 0.682 ± 0.013 | 0.235 ± 0.015 |

Own test set, tampered and fake merged. Intervals: percentile 95%, 1000 resamples by image. SimLBR is trained on our training set with the authors' released code and settings (`simlbr_official_*_metrics.json`). Source: `analysis_extras.json`, `two_class_test`.

| Model | RSFAKE-1M subset | UW fake satellite | DM-AER | fmow-fake-small |
|---|---|---|---|---|
| SimLBR sat, latent blending | 0.678 ± 0.006 | 0.575 ± 0.032 | 0.827 ± 0.015 | 0.590 ± 0.005 |
| SimLBR sat, no blending | 0.666 ± 0.013 | 0.621 ± 0.007 | 0.862 ± 0.005 | 0.563 ± 0.006 |
| SimLBR web, latent blending | 0.623 ± 0.012 | 0.691 ± 0.037 | 0.761 ± 0.013 | 0.427 ± 0.008 |
| SimLBR web, no blending | 0.660 ± 0.003 | 0.699 ± 0.008 | 0.781 ± 0.010 | 0.395 ± 0.005 |

SimLBR AUC on outside datasets, three seeds (`analysis_extras.json`, `simlbr_official_cross`).


## 7. Robustness (seed 42)

| Model | Degradation | Three-class accuracy | Macro-F1 | Tampered recall | AUC | False-alarm rate | Tamper IoU |
|---|---|---|---|---|---|---|---|
| Full (satellite) | clean | 0.9000 | 0.8975 | 0.7658 | 0.9656 | 0.0675 | 0.6746 |
| Full (satellite) | jpeg90 | 0.8971 | 0.8947 | 0.7624 | 0.9651 | 0.0721 | 0.6743 |
| Full (satellite) | jpeg70 | 0.9048 | 0.9023 | 0.7579 | 0.9688 | 0.0496 | 0.6911 |
| Full (satellite) | jpeg50 | 0.8844 | 0.8811 | 0.7207 | 0.9606 | 0.0622 | 0.6476 |
| Full (satellite) | jpeg30 | 0.8648 | 0.8616 | 0.7117 | 0.9491 | 0.1052 | 0.6046 |
| Full (satellite) | scale0.75 | 0.9000 | 0.8975 | 0.7658 | 0.9656 | 0.0675 | 0.6746 |
| Full (satellite) | scale0.5 | 0.7352 | 0.7004 | 0.4752 | 0.9194 | 0.0212 | 0.5472 |
| Web pretraining | clean | 0.9371 | 0.9375 | 0.8176 | 0.9807 | 0.0152 | 0.7021 |
| Web pretraining | jpeg90 | 0.9356 | 0.9358 | 0.8108 | 0.9804 | 0.0159 | 0.6990 |
| Web pretraining | jpeg70 | 0.9349 | 0.9352 | 0.8131 | 0.9809 | 0.0146 | 0.7078 |
| Web pretraining | jpeg50 | 0.9114 | 0.9110 | 0.7534 | 0.9739 | 0.0225 | 0.6660 |
| Web pretraining | jpeg30 | 0.8971 | 0.8960 | 0.7151 | 0.9645 | 0.0357 | 0.6211 |
| Web pretraining | scale0.75 | 0.9371 | 0.9375 | 0.8176 | 0.9807 | 0.0152 | 0.7021 |
| Web pretraining | scale0.5 | 0.8517 | 0.8435 | 0.6745 | 0.9600 | 0.0199 | 0.6286 |

Degradation applied at test time before the 384 x 384 resize. The test images are 512 x 512, so scale 0.75 produces exactly the network input size and coincides with the clean condition. Source: `analysis_extras.json`, `robustness` (`overhead_*` and `robust_*_metrics.json`). The supplement draft of 2026-11-26 printed the two-class accuracy (0.9124 for full_s42) under the heading three-class accuracy; the three-class value is 0.9000, equal to `test_metrics.json`.


## 8. Zero-shot on outside datasets (seed-42 checkpoints)

| Checkpoint | Dataset | Real | Manip. | AUC [95% CI] | Det. @0.5 | FA @0.5 | Det. @5% FA (oracle threshold) |
|---|---|---|---|---|---|---|---|
| Full (satellite) | RSFAKE-1M subset | 5000 | 5000 | 0.584 [0.573, 0.595] | 0.108 | 0.040 | 0.123 |
| Full (satellite) | UW fake satellite | 4032 | 4032 | 0.500 [0.487, 0.513] | 0.033 | 0.034 | 0.044 |
| Full (satellite) | DM-AER (size-matched) | 494 | 1000 | 0.780 [0.753, 0.803] | 0.112 | 0.006 | 0.272 |
| Full (satellite) | GDD in-domain | 750 | 750 | 0.536 [0.507, 0.565] | 0.085 | 0.016 | 0.120 |
| Full (satellite) | GDD cross-generator | 1000 | 1000 | 0.648 [0.625, 0.673] | 0.076 | 0.016 | 0.120 |
| Full (satellite) | GDD cross-domain | 1000 | 1000 | 0.706 [0.684, 0.728] | 0.076 | 0.000 | 0.169 |
| Full (satellite) | GDD local edits | 500 | 500 | 0.568 [0.532, 0.603] | 0.060 | 0.012 | 0.114 |
| Full (satellite) | fmow-fake-small | 60 | 60 | 0.450 [0.350, 0.556] | 0.000 | 0.000 | 0.067 |
| Web pretraining | RSFAKE-1M subset | 5000 | 5000 | 0.668 [0.657, 0.678] | 0.105 | 0.025 | 0.156 |
| Web pretraining | UW fake satellite | 4032 | 4032 | 0.527 [0.514, 0.539] | 0.036 | 0.040 | 0.041 |
| Web pretraining | DM-AER (size-matched) | 494 | 1000 | 0.738 [0.713, 0.763] | 0.184 | 0.014 | 0.317 |
| Web pretraining | GDD in-domain | 750 | 750 | 0.481 [0.450, 0.510] | 0.047 | 0.013 | 0.129 |
| Web pretraining | GDD cross-generator | 1000 | 1000 | 0.596 [0.571, 0.621] | 0.073 | 0.014 | 0.135 |
| Web pretraining | GDD cross-domain | 1000 | 1000 | 0.634 [0.610, 0.658] | 0.073 | 0.004 | 0.170 |
| Web pretraining | GDD local edits | 500 | 500 | 0.560 [0.525, 0.596] | 0.054 | 0.012 | 0.132 |
| Web pretraining | fmow-fake-small | 60 | 60 | 0.432 [0.331, 0.536] | 0.000 | 0.000 | 0.083 |
| Frozen backbone | RSFAKE-1M subset | 5000 | 5000 | 0.543 [0.531, 0.554] | 0.270 | 0.207 | 0.051 |
| Frozen backbone | UW fake satellite | 4032 | 4032 | 0.716 [0.705, 0.726] | 0.336 | 0.119 | 0.069 |
| Frozen backbone | DM-AER (size-matched) | 494 | 1000 | 0.759 [0.730, 0.786] | 0.372 | 0.138 | 0.159 |
| Frozen backbone | GDD in-domain | 750 | 750 | 0.515 [0.485, 0.548] | 0.317 | 0.164 | 0.193 |
| Frozen backbone | GDD cross-generator | 1000 | 1000 | 0.541 [0.515, 0.567] | 0.224 | 0.158 | 0.080 |
| Frozen backbone | GDD cross-domain | 1000 | 1000 | 0.550 [0.525, 0.575] | 0.224 | 0.151 | 0.089 |
| Frozen backbone | GDD local edits | 500 | 500 | 0.505 [0.469, 0.542] | 0.142 | 0.160 | 0.036 |
| Frozen backbone | fmow-fake-small | 60 | 60 | 0.359 [0.266, 0.460] | 0.050 | 0.100 | 0.017 |

No adaptation. Bootstrap: {"resamples": 2000, "seed": 0, "within_class": true, "interval": "percentile 95%"}. Source: `cross_*_metrics.json`, `cross_detail.json`.

| Checkpoint | DM-AER AUC, size-matched reals | DM-AER AUC, all reals [95% CI] | Reals in the all-reals variant |
|---|---|---|---|
| Full (satellite) | 0.780 | 0.665 [0.640, 0.687] | 1000 |
| Web pretraining | 0.738 | 0.657 [0.633, 0.679] | 1000 |

| Dataset | Group | n | Full: AUC vs reals | Full: det. @0.5 | Web: AUC vs reals | Web: det. @0.5 |
|---|---|---|---|---|---|---|
| RSFAKE-1M subset | generated_RSSD_768 | 500 | 0.268 | 0.002 | 0.482 | 0.002 |
| RSFAKE-1M subset | generated_SDFRS | 500 | 0.816 | 0.280 | 0.856 | 0.222 |
| RSFAKE-1M subset | generated_crsdiff | 500 | 0.764 | 0.150 | 0.793 | 0.156 |
| RSFAKE-1M subset | generated_diffusion_sat | 500 | 0.574 | 0.038 | 0.732 | 0.042 |
| RSFAKE-1M subset | generated_diffusion_sat_256 | 500 | 0.804 | 0.224 | 0.890 | 0.312 |
| RSFAKE-1M subset | generated_geosynth | 500 | 0.387 | 0.006 | 0.439 | 0.002 |
| RSFAKE-1M subset | generated_geosynth_canny | 500 | 0.512 | 0.042 | 0.577 | 0.046 |
| RSFAKE-1M subset | generated_geosynth_sam | 500 | 0.407 | 0.016 | 0.451 | 0.002 |
| RSFAKE-1M subset | generated_mapsat | 500 | 0.527 | 0.058 | 0.621 | 0.088 |
| RSFAKE-1M subset | generated_rsinpaint | 500 | 0.778 | 0.264 | 0.837 | 0.174 |
| UW fake satellite | fake/Beijing_style | 2016 | 0.395 | 0.007 | 0.527 | 0.041 |
| UW fake satellite | fake/Seattle_style | 2016 | 0.605 | 0.058 | 0.526 | 0.031 |
| DM-AER (size-matched) | validation/FAKE | 1000 | 0.780 | 0.112 | 0.738 | 0.184 |
| GDD in-domain | gen/diffusionsat | 150 | 0.599 | 0.007 | 0.552 | 0.000 |
| GDD in-domain | gen/flux2dev | 150 | 0.179 | 0.000 | 0.147 | 0.000 |
| GDD in-domain | gen/flux2kleinbase4b | 150 | 0.196 | 0.000 | 0.059 | 0.000 |
| GDD in-domain | gen/neds | 150 | 0.894 | 0.340 | 0.806 | 0.160 |
| GDD in-domain | gen/r3gan | 150 | 0.812 | 0.080 | 0.840 | 0.073 |
| GDD cross-generator | gen/geosynth | 334 | 0.709 | 0.006 | 0.660 | 0.024 |
| GDD cross-generator | gen/rspaint | 333 | 0.836 | 0.222 | 0.861 | 0.195 |
| GDD cross-generator | gen/sd35medium | 333 | 0.400 | 0.000 | 0.268 | 0.000 |
| GDD cross-domain | gen/geosynth | 334 | 0.770 | 0.006 | 0.709 | 0.024 |
| GDD cross-domain | gen/rspaint | 333 | 0.877 | 0.222 | 0.893 | 0.195 |
| GDD cross-domain | gen/sd35medium | 333 | 0.471 | 0.000 | 0.301 | 0.000 |
| GDD local edits | manip/fully-regenerated | 84 | 0.852 | 0.214 | 0.849 | 0.179 |
| GDD local edits | manip/inpaint-splice | 84 | 0.571 | 0.095 | 0.604 | 0.083 |
| GDD local edits | manip/object-copy-move | 83 | 0.500 | 0.024 | 0.439 | 0.000 |
| GDD local edits | manip/object-splice | 83 | 0.442 | 0.024 | 0.502 | 0.036 |
| GDD local edits | manip/simple-copy-move | 83 | 0.486 | 0.000 | 0.467 | 0.012 |
| GDD local edits | manip/simple-splice | 83 | 0.554 | 0.000 | 0.496 | 0.012 |
| fmow-fake-small | inpaint | 20 | 0.450 | 0.000 | 0.427 | 0.000 |
| fmow-fake-small | object-splice | 20 | 0.361 | 0.000 | 0.323 | 0.000 |
| fmow-fake-small | simple-splice | 20 | 0.540 | 0.000 | 0.548 | 0.000 |

Per generator or manipulation type; each group is scored against all real images of its dataset (`cross_detail.json`, `per_group`).

| Checkpoint | Images with mask | IoU | Dice | MCC |
|---|---|---|---|---|
| Full (satellite) | 500 | 0.104 | 0.129 | 0.108 |
| Web pretraining | 500 | 0.177 | 0.213 | 0.211 |
| Frozen backbone | 500 | 0.105 | 0.139 | 0.119 |

Zero-shot tamper localisation on GDD local edits (`cross_*gdd_localization_metrics.json`).


## 9. Few-shot adaptation

| Dataset | Features | Zero-shot | k = 4 | k = 16 | k = 64 | k = 256 | Whole pool |
|---|---|---|---|---|---|---|---|
| RSFAKE-1M subset | fine-tuned satellite, seed 42 | 0.586 | 0.792 ± 0.106 | 0.910 ± 0.030 | 0.965 ± 0.013 | 0.985 ± 0.002 | 0.997 |
| RSFAKE-1M subset | fine-tuned web, seed 42 | 0.668 | 0.710 ± 0.102 | 0.822 ± 0.067 | 0.895 ± 0.014 | 0.941 ± 0.008 | 0.986 |
| RSFAKE-1M subset | original satellite | – | 0.885 ± 0.042 | 0.936 ± 0.015 | 0.958 ± 0.007 | 0.979 ± 0.002 | 0.996 |
| RSFAKE-1M subset | original web | – | 0.695 ± 0.077 | 0.810 ± 0.058 | 0.900 ± 0.018 | 0.948 ± 0.003 | 0.986 |
| UW fake satellite | fine-tuned satellite, seed 42 | 0.491 | 0.860 ± 0.073 | 0.965 ± 0.036 | 0.994 ± 0.009 | 1.000 ± 0.000 | 1.000 |
| UW fake satellite | fine-tuned web, seed 42 | 0.517 | 0.835 ± 0.063 | 0.951 ± 0.039 | 0.995 ± 0.007 | 1.000 ± 0.000 | 1.000 |
| UW fake satellite | original satellite | – | 0.933 ± 0.042 | 0.982 ± 0.020 | 0.996 ± 0.003 | 0.999 ± 0.000 | 1.000 |
| UW fake satellite | original web | – | 0.934 ± 0.034 | 0.980 ± 0.025 | 0.995 ± 0.005 | 1.000 ± 0.000 | 1.000 |
| DM-AER (size-matched) | fine-tuned satellite, seed 42 | 0.760 | 0.864 ± 0.051 | 0.947 ± 0.019 | 0.974 ± 0.005 | – | 0.994 |
| DM-AER (size-matched) | fine-tuned web, seed 42 | 0.732 | 0.885 ± 0.047 | 0.952 ± 0.015 | 0.973 ± 0.006 | – | 0.992 |
| DM-AER (size-matched) | original satellite | – | 0.885 ± 0.041 | 0.958 ± 0.009 | 0.981 ± 0.005 | – | 0.997 |
| DM-AER (size-matched) | original web | – | 0.808 ± 0.067 | 0.959 ± 0.013 | 0.992 ± 0.004 | – | 0.999 |
| GDD in-domain | fine-tuned satellite, seed 42 | 0.529 | 0.677 ± 0.075 | 0.819 ± 0.078 | 0.949 ± 0.015 | 0.980 ± 0.003 | 0.985 |
| GDD in-domain | fine-tuned web, seed 42 | 0.467 | 0.608 ± 0.062 | 0.724 ± 0.072 | 0.864 ± 0.038 | 0.940 ± 0.005 | 0.951 |
| GDD in-domain | original satellite | – | 0.724 ± 0.066 | 0.825 ± 0.023 | 0.927 ± 0.018 | 0.977 ± 0.002 | 0.983 |
| GDD in-domain | original web | – | 0.616 ± 0.060 | 0.708 ± 0.046 | 0.850 ± 0.032 | 0.932 ± 0.009 | 0.948 |
| GDD cross-generator | fine-tuned satellite, seed 42 | 0.636 | 0.758 ± 0.106 | 0.926 ± 0.026 | 0.960 ± 0.006 | 0.977 ± 0.002 | 0.981 |
| GDD cross-generator | fine-tuned web, seed 42 | 0.581 | 0.589 ± 0.084 | 0.760 ± 0.044 | 0.866 ± 0.020 | 0.926 ± 0.005 | 0.948 |
| GDD cross-generator | original satellite | – | 0.695 ± 0.085 | 0.864 ± 0.019 | 0.914 ± 0.009 | 0.948 ± 0.003 | 0.961 |
| GDD cross-generator | original web | – | 0.592 ± 0.057 | 0.752 ± 0.062 | 0.866 ± 0.015 | 0.921 ± 0.006 | 0.942 |
| GDD cross-domain | fine-tuned satellite, seed 42 | 0.702 | 0.847 ± 0.086 | 0.938 ± 0.023 | 0.977 ± 0.008 | 0.989 ± 0.001 | 0.991 |
| GDD cross-domain | fine-tuned web, seed 42 | 0.630 | 0.636 ± 0.079 | 0.793 ± 0.066 | 0.914 ± 0.027 | 0.969 ± 0.002 | 0.981 |
| GDD cross-domain | original satellite | – | 0.710 ± 0.061 | 0.856 ± 0.032 | 0.950 ± 0.006 | 0.978 ± 0.001 | 0.984 |
| GDD cross-domain | original web | – | 0.653 ± 0.086 | 0.773 ± 0.054 | 0.891 ± 0.016 | 0.955 ± 0.002 | 0.973 |
| GDD local edits | fine-tuned satellite, seed 42 | 0.546 | 0.577 ± 0.092 | 0.677 ± 0.029 | 0.734 ± 0.017 | – | 0.766 |
| GDD local edits | fine-tuned web, seed 42 | 0.539 | 0.537 ± 0.069 | 0.634 ± 0.031 | 0.701 ± 0.013 | – | 0.745 |
| GDD local edits | original satellite | – | 0.572 ± 0.061 | 0.636 ± 0.033 | 0.680 ± 0.009 | – | 0.717 |
| GDD local edits | original web | – | 0.548 ± 0.038 | 0.603 ± 0.032 | 0.654 ± 0.026 | – | 0.730 |
| fmow-fake-small | fine-tuned satellite, seed 42 | 0.518 | 0.562 ± 0.161 | 0.600 ± 0.106 | – | – | 0.584 |
| fmow-fake-small | fine-tuned web, seed 42 | 0.406 | 0.477 ± 0.211 | 0.474 ± 0.157 | – | – | 0.361 |
| fmow-fake-small | original satellite | – | 0.485 ± 0.135 | 0.442 ± 0.120 | – | – | 0.390 |
| fmow-fake-small | original web | – | 0.466 ± 0.183 | 0.482 ± 0.149 | – | – | 0.397 |

AUC on the held-out half; k labelled images per class; mean ± standard deviation over 10 draws; whole pool is one fit. Source: `fewshot_metrics.json`, `fewshot_gdd_metrics.json` (protocol `FEWSHOT_PROTOCOL.md`).

| Dataset | Features | k = 4 | k = 16 | k = 64 | k = 256 | Whole pool |
|---|---|---|---|---|---|---|
| RSFAKE-1M subset | fine-tuned satellite | 0.790 ± 0.003 | 0.911 ± 0.001 | 0.964 ± 0.002 | 0.984 ± 0.002 | 0.998 ± 0.000 |
| RSFAKE-1M subset | fine-tuned web | 0.720 ± 0.020 | 0.823 ± 0.016 | 0.891 ± 0.006 | 0.944 ± 0.004 | 0.988 ± 0.002 |
| UW fake satellite | fine-tuned satellite | 0.833 ± 0.025 | 0.959 ± 0.005 | 0.992 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| UW fake satellite | fine-tuned web | 0.834 ± 0.007 | 0.955 ± 0.004 | 0.992 ± 0.002 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| DM-AER (size-matched) | fine-tuned satellite | 0.871 ± 0.007 | 0.945 ± 0.005 | 0.972 ± 0.003 | – | 0.993 ± 0.001 |
| DM-AER (size-matched) | fine-tuned web | 0.889 ± 0.010 | 0.951 ± 0.007 | 0.975 ± 0.004 | – | 0.993 ± 0.001 |
| GDD in-domain | fine-tuned satellite | 0.677 ± 0.012 | 0.826 ± 0.012 | 0.948 ± 0.001 | 0.979 ± 0.001 | 0.984 ± 0.001 |
| GDD in-domain | fine-tuned web | 0.603 ± 0.005 | 0.712 ± 0.013 | 0.862 ± 0.003 | 0.942 ± 0.003 | 0.954 ± 0.002 |
| GDD cross-generator | fine-tuned satellite | 0.771 ± 0.021 | 0.929 ± 0.008 | 0.963 ± 0.002 | 0.978 ± 0.001 | 0.983 ± 0.001 |
| GDD cross-generator | fine-tuned web | 0.607 ± 0.017 | 0.768 ± 0.010 | 0.873 ± 0.008 | 0.931 ± 0.007 | 0.953 ± 0.006 |
| GDD cross-domain | fine-tuned satellite | 0.845 ± 0.011 | 0.939 ± 0.008 | 0.979 ± 0.002 | 0.990 ± 0.001 | 0.992 ± 0.001 |
| GDD cross-domain | fine-tuned web | 0.668 ± 0.028 | 0.803 ± 0.009 | 0.912 ± 0.010 | 0.967 ± 0.006 | 0.980 ± 0.004 |
| GDD local edits | fine-tuned satellite | 0.570 ± 0.010 | 0.673 ± 0.006 | 0.727 ± 0.007 | – | 0.762 ± 0.004 |
| GDD local edits | fine-tuned web | 0.543 ± 0.006 | 0.640 ± 0.006 | 0.700 ± 0.003 | – | 0.745 ± 0.008 |
| fmow-fake-small | fine-tuned satellite | 0.558 ± 0.008 | 0.601 ± 0.002 | – | – | 0.577 ± 0.009 |
| fmow-fake-small | fine-tuned web | 0.508 ± 0.032 | 0.547 ± 0.064 | – | – | 0.480 ± 0.106 |

The same probe on the backbones of seeds 42, 43 and 44: mean ± standard deviation over the three seeds of each seed's mean over 10 draws. Source: `fewshot_metrics.json`, `fewshot_gdd_metrics.json`, `fewshot_seeds_metrics.json`.

| Dataset | Model | k = 4 | k = 16 | k = 64 | k = 256 |
|---|---|---|---|---|---|
| RSFAKE-1M subset | fine-tuned satellite, seed 42 | 0.542 ± 0.024 | 0.547 ± 0.017 | 0.563 ± 0.011 | 0.564 ± 0.007 |
| RSFAKE-1M subset | fine-tuned web, seed 42 | 0.588 ± 0.034 | 0.598 ± 0.033 | 0.610 ± 0.012 | 0.620 ± 0.007 |
| UW fake satellite | fine-tuned satellite, seed 42 | 0.517 ± 0.046 | 0.541 ± 0.035 | 0.549 ± 0.025 | 0.566 ± 0.006 |
| UW fake satellite | fine-tuned web, seed 42 | 0.527 ± 0.034 | 0.549 ± 0.026 | 0.557 ± 0.017 | 0.572 ± 0.001 |
| DM-AER (size-matched) | fine-tuned satellite, seed 42 | 0.674 ± 0.096 | 0.680 ± 0.091 | 0.708 ± 0.040 | – |
| DM-AER (size-matched) | fine-tuned web, seed 42 | 0.635 ± 0.071 | 0.646 ± 0.047 | 0.667 ± 0.024 | – |
| GDD in-domain | fine-tuned satellite, seed 42 | 0.507 ± 0.044 | 0.530 ± 0.034 | 0.550 ± 0.009 | 0.550 ± 0.007 |
| GDD in-domain | fine-tuned web, seed 42 | 0.467 ± 0.032 | 0.511 ± 0.020 | 0.535 ± 0.011 | 0.534 ± 0.007 |
| GDD cross-generator | fine-tuned satellite, seed 42 | 0.577 ± 0.023 | 0.580 ± 0.016 | 0.587 ± 0.007 | 0.586 ± 0.006 |
| GDD cross-generator | fine-tuned web, seed 42 | 0.548 ± 0.026 | 0.558 ± 0.015 | 0.570 ± 0.011 | 0.574 ± 0.004 |
| GDD cross-domain | fine-tuned satellite, seed 42 | 0.604 ± 0.029 | 0.621 ± 0.023 | 0.626 ± 0.016 | 0.630 ± 0.008 |
| GDD cross-domain | fine-tuned web, seed 42 | 0.567 ± 0.030 | 0.597 ± 0.031 | 0.606 ± 0.019 | 0.610 ± 0.009 |
| GDD local edits | fine-tuned satellite, seed 42 | 0.519 ± 0.012 | 0.536 ± 0.007 | 0.538 ± 0.008 | – |
| GDD local edits | fine-tuned web, seed 42 | 0.506 ± 0.015 | 0.516 ± 0.016 | 0.520 ± 0.014 | – |
| fmow-fake-small | fine-tuned satellite, seed 42 | 0.512 ± 0.029 | 0.502 ± 0.024 | – | – |
| fmow-fake-small | fine-tuned web, seed 42 | 0.458 ± 0.045 | 0.437 ± 0.043 | – | – |

Accuracy when only the decision threshold of the fine-tuned model is re-set on the k shots, without a probe.


## 10. Label-free alternatives (seed 42)

| Features | Dataset | Accuracy | Accuracy, best naming (uses labels) | Det. | FA |
|---|---|---|---|---|---|
| fine-tuned satellite, seed 42 | RSFAKE-1M subset | 0.579 | 0.579 | 0.305 | 0.148 |
| fine-tuned satellite, seed 42 | UW fake satellite | 0.418 | 0.582 | 0.740 | 0.905 |
| fine-tuned satellite, seed 42 | DM-AER (size-matched) | 0.628 | 0.628 | 0.467 | 0.047 |
| fine-tuned satellite, seed 42 | GDD in-domain | 0.588 | 0.588 | 0.288 | 0.112 |
| fine-tuned satellite, seed 42 | GDD cross-generator | 0.574 | 0.574 | 0.249 | 0.102 |
| fine-tuned satellite, seed 42 | GDD cross-domain | 0.627 | 0.627 | 0.423 | 0.169 |
| fine-tuned satellite, seed 42 | GDD local edits | 0.534 | 0.534 | 0.216 | 0.148 |
| fine-tuned satellite, seed 42 | fmow-fake-small | 0.408 | 0.592 | 0.150 | 0.333 |
| fine-tuned web, seed 42 | RSFAKE-1M subset | 0.592 | 0.592 | 0.292 | 0.108 |
| fine-tuned web, seed 42 | UW fake satellite | 0.578 | 0.578 | 0.248 | 0.092 |
| fine-tuned web, seed 42 | DM-AER (size-matched) | 0.464 | 0.536 | 0.201 | 0.004 |
| fine-tuned web, seed 42 | GDD in-domain | 0.501 | 0.501 | 0.387 | 0.385 |
| fine-tuned web, seed 42 | GDD cross-generator | 0.505 | 0.505 | 0.420 | 0.410 |
| fine-tuned web, seed 42 | GDD cross-domain | 0.542 | 0.542 | 0.407 | 0.322 |
| fine-tuned web, seed 42 | GDD local edits | 0.476 | 0.524 | 0.368 | 0.416 |
| fine-tuned web, seed 42 | fmow-fake-small | 0.383 | 0.617 | 0.367 | 0.600 |

A. Two-cluster k-means on each outside dataset; clusters named by the model's own mean score.

| Features | Dataset | AUC | Accuracy | Det. | FA |
|---|---|---|---|---|---|
| fine-tuned satellite, seed 42 | Own test set | 0.964 | 0.908 | 0.892 | 0.075 |
| fine-tuned satellite, seed 42 | RSFAKE-1M subset | 0.587 | 0.550 | 0.151 | 0.051 |
| fine-tuned satellite, seed 42 | UW fake satellite | 0.559 | 0.499 | 0.042 | 0.043 |
| fine-tuned satellite, seed 42 | DM-AER (size-matched) | 0.783 | 0.424 | 0.146 | 0.012 |
| fine-tuned satellite, seed 42 | GDD in-domain | 0.505 | 0.535 | 0.099 | 0.028 |
| fine-tuned satellite, seed 42 | GDD cross-generator | 0.629 | 0.530 | 0.092 | 0.031 |
| fine-tuned satellite, seed 42 | GDD cross-domain | 0.751 | 0.541 | 0.092 | 0.009 |
| fine-tuned satellite, seed 42 | GDD local edits | 0.557 | 0.526 | 0.072 | 0.020 |
| fine-tuned satellite, seed 42 | fmow-fake-small | 0.362 | 0.500 | 0.000 | 0.000 |
| fine-tuned web, seed 42 | Own test set | 0.977 | 0.930 | 0.887 | 0.024 |
| fine-tuned web, seed 42 | RSFAKE-1M subset | 0.739 | 0.548 | 0.134 | 0.038 |
| fine-tuned web, seed 42 | UW fake satellite | 0.607 | 0.593 | 0.256 | 0.070 |
| fine-tuned web, seed 42 | DM-AER (size-matched) | 0.776 | 0.473 | 0.224 | 0.024 |
| fine-tuned web, seed 42 | GDD in-domain | 0.510 | 0.514 | 0.051 | 0.023 |
| fine-tuned web, seed 42 | GDD cross-generator | 0.607 | 0.532 | 0.086 | 0.021 |
| fine-tuned web, seed 42 | GDD cross-domain | 0.597 | 0.533 | 0.086 | 0.019 |
| fine-tuned web, seed 42 | GDD local edits | 0.543 | 0.524 | 0.068 | 0.020 |
| fine-tuned web, seed 42 | fmow-fake-small | 0.395 | 0.500 | 0.000 | 0.000 |
| original satellite | Own test set | 0.794 | 0.715 | 0.766 | 0.341 |
| original satellite | RSFAKE-1M subset | 0.620 | 0.591 | 0.386 | 0.204 |
| original satellite | UW fake satellite | 0.616 | 0.628 | 0.545 | 0.289 |
| original satellite | DM-AER (size-matched) | 0.813 | 0.679 | 0.584 | 0.130 |
| original satellite | GDD in-domain | 0.550 | 0.602 | 0.359 | 0.155 |
| original satellite | GDD cross-generator | 0.553 | 0.557 | 0.278 | 0.164 |
| original satellite | GDD cross-domain | 0.576 | 0.567 | 0.278 | 0.144 |
| original satellite | GDD local edits | 0.529 | 0.509 | 0.180 | 0.162 |
| original satellite | fmow-fake-small | 0.493 | 0.483 | 0.050 | 0.083 |
| original web | Own test set | 0.824 | 0.748 | 0.637 | 0.132 |
| original web | RSFAKE-1M subset | 0.549 | 0.558 | 0.352 | 0.235 |
| original web | UW fake satellite | 0.409 | 0.431 | 0.189 | 0.327 |
| original web | DM-AER (size-matched) | 0.825 | 0.701 | 0.599 | 0.093 |
| original web | GDD in-domain | 0.482 | 0.601 | 0.325 | 0.123 |
| original web | GDD cross-generator | 0.493 | 0.555 | 0.241 | 0.130 |
| original web | GDD cross-domain | 0.598 | 0.597 | 0.241 | 0.047 |
| original web | GDD local edits | 0.539 | 0.519 | 0.156 | 0.118 |
| original web | fmow-fake-small | 0.511 | 0.467 | 0.017 | 0.083 |

B. Linear probe fitted on our training set and applied unchanged, threshold 0.5.

| Features | Dataset | AUC | Accuracy | Det. | FA |
|---|---|---|---|---|---|
| fine-tuned satellite, seed 42 | Own test set | 0.948 | 0.800 | 0.974 | 0.388 |
| fine-tuned satellite, seed 42 | RSFAKE-1M subset | 0.571 | 0.527 | 0.897 | 0.843 |
| fine-tuned satellite, seed 42 | UW fake satellite | 0.597 | 0.572 | 0.904 | 0.760 |
| fine-tuned satellite, seed 42 | DM-AER (size-matched) | 0.735 | 0.742 | 0.978 | 0.737 |
| fine-tuned satellite, seed 42 | GDD in-domain | 0.494 | 0.453 | 0.604 | 0.699 |
| fine-tuned satellite, seed 42 | GDD cross-generator | 0.585 | 0.541 | 0.752 | 0.669 |
| fine-tuned satellite, seed 42 | GDD cross-domain | 0.726 | 0.673 | 0.752 | 0.405 |
| fine-tuned satellite, seed 42 | GDD local edits | 0.525 | 0.499 | 0.684 | 0.686 |
| fine-tuned satellite, seed 42 | fmow-fake-small | 0.355 | 0.425 | 0.517 | 0.667 |
| fine-tuned web, seed 42 | Own test set | 0.972 | 0.821 | 0.978 | 0.350 |
| fine-tuned web, seed 42 | RSFAKE-1M subset | 0.790 | 0.676 | 0.976 | 0.624 |
| fine-tuned web, seed 42 | UW fake satellite | 0.731 | 0.661 | 0.938 | 0.616 |
| fine-tuned web, seed 42 | DM-AER (size-matched) | 0.800 | 0.727 | 0.993 | 0.812 |
| fine-tuned web, seed 42 | GDD in-domain | 0.638 | 0.591 | 0.695 | 0.512 |
| fine-tuned web, seed 42 | GDD cross-generator | 0.688 | 0.625 | 0.751 | 0.502 |
| fine-tuned web, seed 42 | GDD cross-domain | 0.643 | 0.571 | 0.751 | 0.609 |
| fine-tuned web, seed 42 | GDD local edits | 0.566 | 0.547 | 0.602 | 0.508 |
| fine-tuned web, seed 42 | fmow-fake-small | 0.390 | 0.442 | 0.167 | 0.283 |
| original satellite | Own test set | 0.514 | 0.502 | 0.486 | 0.481 |
| original satellite | RSFAKE-1M subset | 0.468 | 0.505 | 0.956 | 0.945 |
| original satellite | UW fake satellite | 0.718 | 0.535 | 0.968 | 0.897 |
| original satellite | DM-AER (size-matched) | 0.428 | 0.572 | 0.843 | 0.976 |
| original satellite | GDD in-domain | 0.473 | 0.502 | 0.957 | 0.953 |
| original satellite | GDD cross-generator | 0.486 | 0.475 | 0.912 | 0.961 |
| original satellite | GDD cross-domain | 0.574 | 0.503 | 0.912 | 0.905 |
| original satellite | GDD local edits | 0.453 | 0.504 | 0.962 | 0.954 |
| original satellite | fmow-fake-small | 0.358 | 0.483 | 0.933 | 0.967 |
| original web | Own test set | 0.487 | 0.465 | 0.200 | 0.247 |
| original web | RSFAKE-1M subset | 0.696 | 0.645 | 0.736 | 0.447 |
| original web | UW fake satellite | 0.880 | 0.799 | 0.799 | 0.201 |
| original web | DM-AER (size-matched) | 0.589 | 0.630 | 0.774 | 0.662 |
| original web | GDD in-domain | 0.528 | 0.508 | 0.511 | 0.495 |
| original web | GDD cross-generator | 0.625 | 0.567 | 0.629 | 0.496 |
| original web | GDD cross-domain | 0.531 | 0.496 | 0.629 | 0.637 |
| original web | GDD local edits | 0.556 | 0.537 | 0.574 | 0.500 |
| original web | fmow-fake-small | 0.346 | 0.400 | 0.183 | 0.383 |

C. One-class score (mean cosine distance to the 5 nearest pristine training images), threshold at the 95th percentile of held-out pristine training images.

Source: `zeroshot_extras_metrics.json` (protocol `ZEROSHOT_EXTRAS_PROTOCOL.md`).


## Other training seeds: zero-shot, robustness, label-free (added 2026-11-27/28)

Protocol: `ZEROSHOT_EXTRAS_PROTOCOL.md`, addendum 2026-11-27. Unchanged scripts applied to the seed-43 and seed-44 checkpoints (`run_seed_evals.sh`). Files `cross_s43_*`, `cross_web_s43_*`, `robust_full_s43_*`, `zeroshot_extras_seeds_metrics.json` and the same for seed 44.


### Zero-shot AUC per seed (whole outside set)

| Dataset | Full s42 | Full s43 | Full s44 | Full mean ± sd | Web s42 | Web s43 | Web s44 | Web mean ± sd |
|---|---|---|---|---|---|---|---|---|
| RSFAKE-1M subset | 0.584 | 0.561 | 0.591 | 0.579 ± 0.016 | 0.668 | 0.653 | 0.680 | 0.667 ± 0.013 |
| UW fake satellite | 0.500 | 0.418 | 0.423 | 0.447 ± 0.046 | 0.527 | 0.527 | 0.527 | 0.527 ± 0.000 |
| DM-AER (size-matched) | 0.780 | 0.786 | 0.760 | 0.775 ± 0.013 | 0.738 | 0.768 | 0.740 | 0.749 ± 0.017 |
| GDD in-domain | 0.536 | 0.556 | 0.620 | 0.571 ± 0.044 | 0.481 | 0.489 | 0.481 | 0.484 ± 0.005 |
| GDD cross-generator | 0.648 | 0.663 | 0.718 | 0.677 ± 0.037 | 0.596 | 0.583 | 0.628 | 0.602 ± 0.023 |
| GDD cross-domain | 0.706 | 0.709 | 0.734 | 0.716 ± 0.015 | 0.634 | 0.614 | 0.625 | 0.625 ± 0.010 |
| GDD local edits | 0.568 | 0.555 | 0.533 | 0.552 ± 0.018 | 0.560 | 0.552 | 0.585 | 0.566 ± 0.017 |
| fmow-fake-small | 0.450 | 0.419 | 0.378 | 0.416 ± 0.036 | 0.432 | 0.451 | 0.453 | 0.446 ± 0.012 |
| DM-AER, all real images | 0.665 | 0.689 | 0.650 | 0.668 ± 0.020 | 0.657 | 0.691 | 0.656 | 0.668 ± 0.020 |


### Detection and false-alarm rates at threshold 0.5, mean over seeds

| Dataset | Full: detection | Full: false alarm | Web: detection | Web: false alarm |
|---|---|---|---|---|
| RSFAKE-1M subset | 0.093 ± 0.017 | 0.042 ± 0.007 | 0.112 ± 0.007 | 0.031 ± 0.007 |
| UW fake satellite | 0.024 ± 0.011 | 0.036 ± 0.004 | 0.026 ± 0.008 | 0.038 ± 0.014 |
| DM-AER (size-matched) | 0.118 ± 0.032 | 0.009 ± 0.003 | 0.181 ± 0.004 | 0.015 ± 0.005 |
| GDD in-domain | 0.087 ± 0.012 | 0.015 ± 0.002 | 0.051 ± 0.034 | 0.014 ± 0.007 |
| GDD cross-generator | 0.068 ± 0.011 | 0.016 ± 0.002 | 0.070 ± 0.007 | 0.012 ± 0.004 |
| GDD cross-domain | 0.068 ± 0.011 | 0.003 ± 0.003 | 0.070 ± 0.007 | 0.004 ± 0.001 |
| GDD local edits | 0.054 ± 0.010 | 0.012 ± 0.000 | 0.049 ± 0.012 | 0.012 ± 0.006 |
| fmow-fake-small | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |


### Robustness on the test set, mean ± sd over seeds 42-44

| Condition | Full: accuracy | Full: tamper IoU | Web: accuracy | Web: tamper IoU |
|---|---|---|---|---|
| clean | 0.903 ± 0.003 | 0.671 ± 0.003 | 0.935 ± 0.003 | 0.706 ± 0.005 |
| jpeg90 | 0.901 ± 0.004 | 0.670 ± 0.003 | 0.934 ± 0.003 | 0.703 ± 0.005 |
| jpeg70 | 0.910 ± 0.005 | 0.686 ± 0.005 | 0.935 ± 0.002 | 0.713 ± 0.006 |
| jpeg50 | 0.885 ± 0.005 | 0.643 ± 0.004 | 0.910 ± 0.002 | 0.665 ± 0.004 |
| jpeg30 | 0.867 ± 0.002 | 0.599 ± 0.005 | 0.894 ± 0.004 | 0.621 ± 0.004 |
| scale0.75 | 0.903 ± 0.003 | 0.671 ± 0.003 | 0.935 ± 0.003 | 0.706 ± 0.005 |
| scale0.5 | 0.730 ± 0.021 | 0.539 ± 0.011 | 0.868 ± 0.014 | 0.634 ± 0.006 |


### Label-free alternatives, mean ± sd over seeds 42-44

Features of the full model:

| Dataset | Clustering accuracy | Source probe AUC | One-class AUC | One-class false-alarm rate |
|---|---|---|---|---|
| RSFAKE-1M subset | 0.558 ± 0.018 | 0.548 ± 0.034 | 0.522 ± 0.043 | 0.876 ± 0.034 |
| UW fake satellite | 0.422 ± 0.005 | 0.490 ± 0.064 | 0.558 ± 0.035 | 0.770 ± 0.013 |
| DM-AER (size-matched) | 0.592 ± 0.031 | 0.770 ± 0.011 | 0.720 ± 0.017 | 0.737 ± 0.024 |
| GDD in-domain | 0.577 ± 0.010 | 0.528 ± 0.021 | 0.499 ± 0.006 | 0.744 ± 0.043 |
| GDD cross-generator | 0.612 ± 0.079 | 0.650 ± 0.021 | 0.589 ± 0.007 | 0.725 ± 0.051 |
| GDD cross-domain | 0.625 ± 0.044 | 0.748 ± 0.004 | 0.728 ± 0.009 | 0.457 ± 0.045 |
| GDD local edits | 0.529 ± 0.011 | 0.542 ± 0.014 | 0.517 ± 0.019 | 0.739 ± 0.049 |
| fmow-fake-small | 0.453 ± 0.039 | 0.362 ± 0.007 | 0.346 ± 0.010 | 0.767 ± 0.100 |

Features of the web model:

| Dataset | Clustering accuracy | Source probe AUC | One-class AUC | One-class false-alarm rate |
|---|---|---|---|---|
| RSFAKE-1M subset | 0.595 ± 0.003 | 0.737 ± 0.010 | 0.785 ± 0.008 | 0.604 ± 0.037 |
| UW fake satellite | 0.575 ± 0.003 | 0.584 ± 0.020 | 0.724 ± 0.018 | 0.581 ± 0.031 |
| DM-AER (size-matched) | 0.474 ± 0.012 | 0.772 ± 0.021 | 0.788 ± 0.011 | 0.781 ± 0.045 |
| GDD in-domain | 0.513 ± 0.047 | 0.523 ± 0.015 | 0.621 ± 0.016 | 0.504 ± 0.022 |
| GDD cross-generator | 0.542 ± 0.039 | 0.625 ± 0.016 | 0.694 ± 0.018 | 0.481 ± 0.022 |
| GDD cross-domain | 0.553 ± 0.011 | 0.607 ± 0.009 | 0.655 ± 0.021 | 0.563 ± 0.058 |
| GDD local edits | 0.497 ± 0.034 | 0.548 ± 0.005 | 0.564 ± 0.009 | 0.506 ± 0.025 |
| fmow-fake-small | 0.403 ± 0.027 | 0.381 ± 0.014 | 0.360 ± 0.027 | 0.267 ± 0.044 |


## Few-shot with the test generator left out of the labeled images (added 2026-11-27)

Protocol: `FEWSHOT_PROTOCOL.md`, addendum 2026-11-27 (evening). File `fewshot_logo_metrics.json`. Each generator or manipulation type in turn is tested alone against the real images of the held-out half. AUC, mean over the left-out groups; fine-tuned extractors: mean over seeds 42-44. 'seen' = shots of the original protocol, 'left out' = no labeled image of the tested group. The real images of shots and test share their sources.

| Dataset | Groups | Features | k=4: seen | k=4: left out | k=16: seen | k=16: left out | k=64: seen | k=64: left out | k=256: seen | k=256: left out | pool: seen | pool: left out |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RSFAKE-1M subset | 10 | fine-tuned satellite | 0.790 | 0.766 | 0.911 | 0.880 | 0.964 | 0.935 | 0.984 | 0.961 | 0.998 | 0.981 |
| RSFAKE-1M subset | 10 | fine-tuned web | 0.720 | 0.697 | 0.823 | 0.804 | 0.891 | 0.867 | 0.944 | 0.921 | 0.988 | 0.971 |
| RSFAKE-1M subset | 10 | original satellite | 0.885 | 0.859 | 0.936 | 0.915 | 0.958 | 0.939 | 0.979 | 0.960 | 0.996 | 0.979 |
| RSFAKE-1M subset | 10 | original web | 0.695 | 0.708 | 0.810 | 0.783 | 0.900 | 0.867 | 0.948 | 0.918 | 0.986 | 0.959 |
| UW fake satellite | 2 | fine-tuned satellite | 0.833 | 0.768 | 0.959 | 0.864 | 0.992 | 0.914 | 1.000 | 0.959 | 1.000 | 0.984 |
| UW fake satellite | 2 | fine-tuned web | 0.834 | 0.811 | 0.955 | 0.935 | 0.992 | 0.975 | 1.000 | 0.995 | 1.000 | 0.998 |
| UW fake satellite | 2 | original satellite | 0.933 | 0.918 | 0.982 | 0.962 | 0.996 | 0.978 | 0.999 | 0.992 | 1.000 | 0.996 |
| UW fake satellite | 2 | original web | 0.934 | 0.920 | 0.980 | 0.969 | 0.995 | 0.986 | 1.000 | 0.997 | 1.000 | 0.999 |
| fmow-fake-small | 3 | fine-tuned satellite | 0.558 | 0.546 | 0.601 | 0.510 | – | – | – | – | 0.577 | 0.517 |
| fmow-fake-small | 3 | fine-tuned web | 0.508 | 0.491 | 0.547 | 0.470 | – | – | – | – | 0.480 | 0.460 |
| fmow-fake-small | 3 | original satellite | 0.485 | 0.452 | 0.442 | 0.347 | – | – | – | – | 0.390 | 0.319 |
| fmow-fake-small | 3 | original web | 0.466 | 0.461 | 0.482 | 0.352 | – | – | – | – | 0.397 | 0.333 |
| GDD in-domain | 5 | fine-tuned satellite | 0.677 | 0.483 | 0.826 | 0.551 | 0.948 | 0.645 | 0.979 | 0.722 | 0.984 | 0.731 |
| GDD in-domain | 5 | fine-tuned web | 0.603 | 0.469 | 0.712 | 0.469 | 0.862 | 0.564 | 0.942 | 0.668 | 0.954 | 0.687 |
| GDD in-domain | 5 | original satellite | 0.724 | 0.629 | 0.825 | 0.670 | 0.927 | 0.735 | 0.977 | 0.768 | 0.983 | 0.772 |
| GDD in-domain | 5 | original web | 0.616 | 0.552 | 0.708 | 0.546 | 0.850 | 0.636 | 0.932 | 0.691 | 0.948 | 0.700 |
| GDD cross-generator | 3 | fine-tuned satellite | 0.771 | 0.648 | 0.929 | 0.771 | 0.963 | 0.819 | 0.978 | 0.846 | 0.983 | 0.856 |
| GDD cross-generator | 3 | fine-tuned web | 0.607 | 0.485 | 0.768 | 0.525 | 0.873 | 0.560 | 0.931 | 0.601 | 0.953 | 0.611 |
| GDD cross-generator | 3 | original satellite | 0.695 | 0.608 | 0.864 | 0.719 | 0.914 | 0.771 | 0.948 | 0.792 | 0.961 | 0.797 |
| GDD cross-generator | 3 | original web | 0.592 | 0.507 | 0.752 | 0.546 | 0.866 | 0.574 | 0.921 | 0.630 | 0.942 | 0.644 |
| GDD cross-domain | 3 | fine-tuned satellite | 0.845 | 0.702 | 0.939 | 0.810 | 0.979 | 0.871 | 0.990 | 0.907 | 0.992 | 0.917 |
| GDD cross-domain | 3 | fine-tuned web | 0.668 | 0.525 | 0.803 | 0.583 | 0.912 | 0.666 | 0.967 | 0.730 | 0.980 | 0.747 |
| GDD cross-domain | 3 | original satellite | 0.711 | 0.626 | 0.857 | 0.696 | 0.950 | 0.773 | 0.978 | 0.840 | 0.984 | 0.855 |
| GDD cross-domain | 3 | original web | 0.653 | 0.546 | 0.773 | 0.598 | 0.891 | 0.663 | 0.955 | 0.735 | 0.973 | 0.756 |
| GDD local edits | 6 | fine-tuned satellite | 0.570 | 0.571 | 0.673 | 0.636 | 0.727 | 0.692 | – | – | 0.761 | 0.726 |
| GDD local edits | 6 | fine-tuned web | 0.543 | 0.554 | 0.639 | 0.602 | 0.699 | 0.660 | – | – | 0.744 | 0.709 |
| GDD local edits | 6 | original satellite | 0.572 | 0.561 | 0.636 | 0.607 | 0.679 | 0.658 | – | – | 0.716 | 0.689 |
| GDD local edits | 6 | original web | 0.548 | 0.565 | 0.603 | 0.598 | 0.654 | 0.647 | – | – | 0.729 | 0.701 |


## TruFor on the test folder (added 2026-11-28)

Protocol: `TRUFOR_PROTOCOL.md`. Authors' code and weights, no training, 3,150 test images at 512 x 512, one run. Not a like-for-like comparison: TruFor was trained on edited photographs; masks are scored at 512 x 512 (our models at 384 x 384).

| Quantity | Value |
|---|---|
| AUC, pristine vs manipulated | 0.530 |
| AUC, pristine vs tampered | 0.594 |
| AUC, pristine vs fake | 0.455 |
| Accuracy at 0.5 | 0.516 |
| Detection rate at 0.5 (tampered / fake) | 0.229 / 0.085 |
| False-alarm rate at 0.5 | 0.101 |
| Tamper IoU / Dice / MCC (888 tampered images) | 0.207 / 0.255 / 0.244 |


## 11. Deviations, order of analyses, and open questions

Recorded in `RUNLOG.md` and the git history (`git log`). In summary:

- Checkpoints are selected on the validation split (10% of the training folder, seed 42), never on the test set.
- The few-shot, GDD and label-free experiments were added after the zero-shot results were known; each protocol file was committed before its numbers were computed (`FEWSHOT_PROTOCOL.md`, `ZEROSHOT_EXTRAS_PROTOCOL.md`).
- A resampling-augmentation run to improve zero-shot transfer was planned and not run (`ZEROSHOT_AUG_PROTOCOL.md`, marked NOT RUN).
- `full_s42` diverged once at epoch 19 with non-finite losses under the first mixed-precision setting; the log and folder are kept (`full_s42_DIVERGED_ep19_nan`), the fix is in `logs/fix_amp_resume.diff`, and all reported runs use bf16.
- `radio_s42` failed on the workstation (folder kept as `radio_s42_FAILED_workstation`) and was run on the laptop GPU, as were `fullnorm_s42`, `webnorm_s42` and `plain_s43`; `plain_s44` was moved from the laptop to the workstation.
- Inputs are scaled to [0, 1] without the backbones' own normalisation; `fullnorm_s42` and `webnorm_s42` measure the effect (section 3), one seed each.
- p-values are raw (not corrected for multiple comparisons).
- Single runs: frozen, ResNet-50 (both learning rates), C-RADIO, fullnorm, webnorm. Zero-shot, robustness, two-class and label-free numbers use the seed-42 checkpoints only.

Open questions:

- The exact MCC definition of May et al. (per image and averaged, or pooled over pixels) has not been confirmed against their code; our MCC is per image, averaged over tampered images.
- The weight of the loss on small tampered regions was not studied; small regions are the main failure (section 3, area buckets).
- SimLBR numbers from its paper are not reproduced here; only SimLBR trained on our data with the authors' code is reported.
