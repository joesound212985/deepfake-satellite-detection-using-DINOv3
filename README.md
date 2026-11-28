# Detecting and localizing AI-generated and tampered satellite images with DINOv3

Experiment code, run log, training logs and result files for the paper
*Detection and Localization of AI-Generated and Tampered Satellite Images With a Fine-Tuned DINOv3*
(Soundararajan, Chadha, Xu; in preparation, working title). The original script is on the `main` branch; this
branch (`ojcs-experiments`) holds the revised training script and everything needed to reproduce the tables.

**Status: the last training runs and the seed re-scoring are still in progress.** The tables below are
rebuilt from the result files whenever they change; rows marked as missing or pending belong to runs that
have not finished. `RUNLOG.md` is the authoritative record of every command and outcome.

## Model

A DINOv3 ViT-L/16 backbone pretrained on satellite imagery
(`facebook/dinov3-vitl16-pretrain-sat493m`) with an edge-gated SegFormer-style
decoder. One forward pass gives:

- a 3-class image label: pristine, tampered (partly manipulated) or fake (fully synthetic);
- a pixel mask of the manipulated region.

Input is 384×384. Training uses bfloat16 mixed precision with losses in float32,
50 epochs, batch size 3, backbone frozen for the first 3 epochs. Checkpoints are
selected on a validation split of the training set, never on the test set.
The full model has 309.4 M parameters.

## Results

<!-- results:start -->

*Generated 2026-11-27 by `update_readme_results.py` from the result files in this repository. Rows with one seed are single runs; the others are mean ± standard deviation over the listed seeds.*

### Own test set (3,150 images; three classes, tamper mask on the 888 tampered images)

| Row | Seeds | Accuracy | Macro-F1 | Balanced acc. | Tampered recall | Mask IoU | Mask Dice | Mask MCC | ms / image |
|---|---|---|---|---|---|---|---|---|---|
| Full (satellite pretraining) | 42, 43, 44 | 0.903 ± 0.003 | 0.902 ± 0.004 | 0.900 ± 0.004 | 0.774 ± 0.020 | 0.671 ± 0.003 | 0.762 ± 0.003 | 0.742 ± 0.004 | 88.6 |
| Plain decoder | 42, 43, 44 | 0.904 ± 0.003 | 0.902 ± 0.004 | 0.900 ± 0.004 | 0.771 ± 0.015 | 0.674 ± 0.001 | 0.766 ± 0.002 | 0.744 ± 0.002 | 98.1 |
| Web pretraining | 42, 43, 44 | 0.935 ± 0.002 | 0.935 ± 0.002 | 0.926 ± 0.002 | 0.805 ± 0.011 | 0.706 ± 0.005 | 0.795 ± 0.006 | 0.779 ± 0.005 | 88.8 |
| Frozen backbone | 42 | 0.643 | 0.512 | 0.593 | 0.015 | 0.462 | 0.572 | 0.536 | 88.2 |
| ResNet-50, backbone LR 2e-6 | 42 | 0.665 | 0.636 | 0.658 | 0.306 | 0.312 | 0.410 | 0.361 | 39.2 |
| ResNet-50, backbone LR 1e-4 | 42 | 0.762 | 0.752 | 0.753 | 0.514 | 0.357 | 0.442 | 0.413 | 39.2 |
| C-RADIOv4 | 42 | 0.835 | 0.819 | 0.826 | 0.572 | 0.587 | 0.679 | 0.660 | 147.3 |

### Zero-shot on outside datasets (no adaptation; seed-42 checkpoints)

AUC of real versus manipulated with a bootstrap 95% interval, and the detection / false-alarm rates at the fixed threshold 0.5.

| Dataset | Real / manip. | Full: AUC [95% CI] | Full: det. / FA @0.5 | Web: AUC [95% CI] | Web: det. / FA @0.5 |
|---|---|---|---|---|---|
| RSFAKE-1M subset | 5000 / 5000 | 0.584 [0.573, 0.595] | 0.108 / 0.040 | 0.668 [0.657, 0.678] | 0.105 / 0.025 |
| UW fake satellite | 4032 / 4032 | 0.500 [0.487, 0.513] | 0.033 / 0.034 | 0.527 [0.514, 0.539] | 0.036 / 0.040 |
| DM-AER (size-matched) | 494 / 1000 | 0.780 [0.753, 0.803] | 0.112 / 0.006 | 0.738 [0.713, 0.763] | 0.184 / 0.014 |
| GDD in-domain | 750 / 750 | 0.536 [0.507, 0.565] | 0.085 / 0.016 | 0.481 [0.450, 0.510] | 0.047 / 0.013 |
| GDD cross-generator | 1000 / 1000 | 0.648 [0.625, 0.673] | 0.076 / 0.016 | 0.596 [0.571, 0.621] | 0.073 / 0.014 |
| GDD cross-domain | 1000 / 1000 | 0.706 [0.684, 0.728] | 0.076 / 0.000 | 0.634 [0.610, 0.658] | 0.073 / 0.004 |
| GDD local edits | 500 / 500 | 0.568 [0.532, 0.603] | 0.060 / 0.012 | 0.560 [0.525, 0.596] | 0.054 / 0.012 |
| fmow-fake-small (120 images) | 60 / 60 | 0.450 [0.350, 0.556] | 0.000 / 0.000 | 0.432 [0.331, 0.536] | 0.000 / 0.000 |

### Few-shot adaptation (linear probe on the frozen fine-tuned satellite backbone, seed 42)

AUC on the held-out half of each outside dataset with k labelled images per class (10 random draws; protocol in `FEWSHOT_PROTOCOL.md`).

| Dataset | Zero-shot | k = 4 | k = 16 | k = 64 | k = 256 | Whole pool |
|---|---|---|---|---|---|---|
| RSFAKE-1M subset | 0.586 | 0.792 ± 0.106 | 0.910 ± 0.030 | 0.965 ± 0.013 | 0.985 ± 0.002 | 0.997 |
| UW fake satellite | 0.491 | 0.860 ± 0.073 | 0.965 ± 0.036 | 0.994 ± 0.009 | 1.000 ± 0.000 | 1.000 |
| DM-AER (size-matched) | 0.760 | 0.864 ± 0.051 | 0.947 ± 0.019 | 0.974 ± 0.005 | – | 0.994 |
| GDD in-domain | 0.529 | 0.677 ± 0.075 | 0.819 ± 0.078 | 0.949 ± 0.015 | 0.980 ± 0.003 | 0.985 |
| GDD cross-generator | 0.636 | 0.758 ± 0.106 | 0.926 ± 0.026 | 0.960 ± 0.006 | 0.977 ± 0.002 | 0.981 |
| GDD cross-domain | 0.702 | 0.847 ± 0.086 | 0.938 ± 0.023 | 0.977 ± 0.008 | 0.989 ± 0.001 | 0.991 |
| GDD local edits | 0.546 | 0.577 ± 0.092 | 0.677 ± 0.029 | 0.734 ± 0.017 | – | 0.766 |
| fmow-fake-small (120 images) | 0.518 | 0.562 ± 0.161 | 0.600 ± 0.106 | – | – | 0.584 |

<!-- results:end -->

## Table rows

| Row | What it is |
|---|---|
| `full` | The proposed model |
| `plain` | No edge gate, boundary loss or deep supervision |
| `frozen` | Backbone kept frozen |
| `web` | DINOv3 pretrained on web images instead of satellite images |
| `resnet` | ResNet-50 backbone |
| `resnetlr` | ResNet-50 with a conventional backbone learning rate (1e-4, fixed in advance); the `resnet` row uses the DINOv3 rate (2e-6) |
| `radio` | C-RADIOv4 backbone |
| SimLBR | Trained on our data with the authors' released code (`simlbr_official.py`, github.com/mvrl/SimLBR at commit ac90f86); binary real-vs-manipulated. `simlbr_baseline.py` is our earlier re-implementation, kept as a cross-check |

The `full`, `web` and `frozen` seed-42 checkpoints are also scored zero-shot on five external datasets:
fmow-fake-small, a 10,000-image subset of RSFAKE-1M, the UW Fake Satellite set, the
DM-AER-DeepFake-V1 official sample (StyleGAN2; real images size-matched to the fakes) and the four test
subsets of GDD (`run_gdd.sh`).
`setup_data.sh` downloads the first four; the main training set is gated by its authors' form
and has to be supplied separately (see the header of `setup_data.sh`).

## Additional analyses

Run automatically on the seed-42 checkpoints of the satellite- and web-pretrained models:

- the test set re-scored as real-vs-manipulated, for a like-for-like comparison with SimLBR;
- robustness to JPEG re-compression (quality 90/70/50/30) and down-scaling (0.75, 0.5);
- per-generator detection on RSFAKE-1M and bootstrap 95% intervals on all external sets;
- exact McNemar tests between the full, plain and web rows on the same test images, per seed;
- a qualitative figure (`qualitative.png`);
- an input-normalisation check (`fullnorm`, `webnorm`), reported separately from the main rows.

## Files

| File | Purpose |
|---|---|
| `dinov3_forensics_train_v2.py` | Training and evaluation script (variants via environment variables) |
| `dinov3_map_forgery_2025.original.py`, `v2_changes.diff` | Original script and the changes made to it |
| `go.sh`, `watch.py` | Start the whole queue (if not running) and show a live dashboard |
| `run_everything.sh`, `run_extra_seeds_v2.sh`, `run_web_seeds.sh` | The three queue stages: seed-42 rows; extra cross-dataset evaluations and seeds 43/44; web-backbone seeds |
| `eval_cross_dataset.py`, `prepare_rsfake_subset.py` | Zero-shot cross-dataset evaluation |
| `simlbr_official.py` | Runs the authors' SimLBR code (cloned separately, not redistributed here) on our data; see the file header for the three run-time adaptations |
| `simlbr_baseline.py`, `simlbr_metrics.json`, `simlbr_nolbr_metrics.json` | Our own SimLBR re-implementation and its results (cross-check) |
| `count_flops.py`, `flops.json` | FLOP count per variant |
| `analyze_extras.py` | Post-hoc analyses from saved per-image files: two-class test scores next to SimLBR, per-generator RSFAKE results, robustness table, bootstrap 95% intervals |
| `make_qualitative.py` | Qualitative figure; examples picked by a fixed rule, not by hand |
| `run_frozen_cross.sh` | Zero-shot scoring of the frozen-backbone checkpoint on the external sets |
| `FEWSHOT_PROTOCOL.md`, `fewshot_adapt.py`, `run_fewshot.sh` | Few-shot adaptation experiment: linear probes on frozen features with k labelled images per class from each external set (protocol fixed in advance) |
| `run_gdd.sh` | GDD (arXiv 2611.31487) as a fifth external test set: downloads its test splits and scores the full, web and frozen checkpoints zero-shot |
| `run_laptop.sh` | Share of the queue run on a second machine: `fullnorm`, `webnorm`, then `plain` seeds 43/44 |
| `train_norm.py`, `run_norm_check.sh` | Side experiment: same training with each backbone's own input normalisation (`fullnorm`, `webnorm` rows) |
| `aggregate_runs.py` | Builds the results table from `test_metrics.json` files |
| `update_readme_results.py` | Rewrites the results tables of this README from the result files |
| `setup_env.sh`, `setup_data.sh` | Environment and dataset setup |
| `RUNLOG.md`, `logs/` | Run log and per-run output |

## Reproducing

```
./setup_env.sh
./setup_data.sh
SEEDS="42 43 44" ROWS="full plain frozen resnet web radio" ./run_everything.sh
```

Datasets and checkpoints are not in this repository. The DINOv3 weights are
gated on Hugging Face and need an approved account. RSFAKE-1M is licensed for
non-commercial research only; follow each dataset's licence.
