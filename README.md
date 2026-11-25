# GRSL experiments: detecting AI-generated satellite imagery with DINOv3

Experiment code, run log and training logs for the letter
*Detecting AI-Generated Satellite Imagery Deepfakes Using DINOv3*
(Soundararajan, Chadha, Xu). The original script is on the `main` branch; this
branch (`grsl-experiments`) holds the revised training script and everything
needed to reproduce the tables.

**Status: runs in progress.** Numbers below are from a single seed and will be
replaced by mean ± std over three seeds. `RUNLOG.md` is the authoritative record
of every command and outcome.

## Model

A DINOv3 ViT-L/16 backbone pretrained on satellite imagery
(`facebook/dinov3-vitl16-pretrain-sat493m`) with an edge-gated SegFormer-style
decoder. One forward pass gives:

- a 3-class image label: pristine, tampered (partly manipulated) or fake (fully synthetic);
- a pixel mask of the manipulated region.

Input is 384×384. Training uses bfloat16 mixed precision with losses in float32,
50 epochs, batch size 3, backbone frozen for the first 3 epochs. Checkpoints are
selected on a validation split of the training set, never on the test set.

## Preliminary result (full model, seed 42, 3,150 test images)

| Metric | Value |
|---|---|
| Accuracy | 0.900 |
| Macro-F1 | 0.898 |
| Balanced accuracy | 0.896 |
| Mean IoU / Dice / MCC on tampered images | 0.675 / 0.766 / 0.747 |
| Parameters | 309.4 M |
| GFLOPs per 384×384 forward pass (1 per multiply-add) | 581.2 |
| Inference, 4-view test-time augmentation, RTX 4080 SUPER | 88.8 ms/image |

## Planned table rows

| Row | What it is |
|---|---|
| `full` | The proposed model |
| `plain` | No edge gate, boundary loss or deep supervision |
| `frozen` | Backbone kept frozen |
| `web` | DINOv3 pretrained on web images instead of satellite images |
| `resnet` | ResNet-50 backbone |
| `radio` | C-RADIOv4 backbone |
| SimLBR | Re-implementation of latent blending regularization following the official code (`simlbr_baseline.py`); binary real-vs-manipulated |

The `full` checkpoint is also scored zero-shot on three external sets:
fmow-fake-small, a 10,000-image subset of RSFAKE-1M, the UW Fake Satellite set and the
DM-AER-DeepFake-V1 official sample (StyleGAN2; real images size-matched to the fakes).
`setup_data.sh` downloads all four; the main training set is gated by its authors' form
and has to be supplied separately (see the header of `setup_data.sh`).

## Files

| File | Purpose |
|---|---|
| `dinov3_forensics_train_v2.py` | Training and evaluation script (variants via environment variables) |
| `dinov3_map_forgery_2025.original.py`, `v2_changes.diff` | Original script and the changes made to it |
| `go.sh`, `watch.py` | Start the whole queue (if not running) and show a live dashboard |
| `run_everything.sh`, `run_extra_seeds_v2.sh`, `run_web_seeds.sh` | The three queue stages: seed-42 rows; extra cross-dataset evaluations and seeds 43/44; web-backbone seeds |
| `eval_cross_dataset.py`, `prepare_rsfake_subset.py` | Zero-shot cross-dataset evaluation |
| `simlbr_baseline.py`, `simlbr_metrics.json`, `simlbr_nolbr_metrics.json` | SimLBR baseline and its results |
| `count_flops.py`, `flops.json` | FLOP count per variant |
| `aggregate_runs.py` | Builds the results table from `test_metrics.json` files |
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
