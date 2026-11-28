# Training on a second generator family (GDD): protocol

Written 2026-11-28 before any model was trained on GDD (Joe, 2026-11-28: "yes second family, train on GDD"; a
review pointed out that every trained model had seen one generator family only). No number from this experiment
existed when this file was committed. Until now GDD was used for testing only (`run_gdd.sh`, `run_gdd_loc.sh`).

## Question

Is the weak transfer of the benchmark-trained model a property of the model or of the training data? The same
model is trained on GDD instead and tested in both directions.

## Data

GDD (Arndt et al., arXiv 2611.31487, CC BY 4.0), its own split files. Training set = union of
`train_detection.csv` and `train_localization.csv`: 11,410 images (5,000 real from fMoW, Million-AID and xBD;
5,000 fully synthesized by DiffusionSat, R3GAN, Flux2-dev, Flux2-klein and NeDS; 1,410 locally manipulated, six
manipulation types, with masks).

Mapping to the three classes of the training script (`prepare_gdd_train.py` writes a `metadata.csv` in the
layout the script already reads; the images are not copied or converted):

| GDD | class | mask |
|---|---|---|
| label real | pristine | none |
| label fake, no manipulation type (fully synthesized) | fake | none |
| label fake with a manipulation type (including "fully-regenerated", which GDD lists with a mask) | tampered | GDD mask |

GDD's files are a mix of TIFF, PNG and JPEG and the format is not balanced over the classes. That is how the
dataset is published and it is used as it is; it is named as a caveat with the results.

## Training

The `full` row unchanged (no change to model, losses, augmentation, sampler or settings): existing switches only,
`TRAIN_ROOT` and `TEST_ROOT` pointed at the GDD layout.

- Checkpoint selection as in every other run: 10 % of the training set held out by the script (VAL_SEED=42), best
  epoch by macro F1 on it. GDD's own validation split is not used for training or selection.
- The script's built-in test is run on GDD's in-domain test images (union of `test_indomain_detection.csv` and
  `test_indomain_localization.csv`).
- Seeds 42, 43, 44; run tags `gdd_s42`, `gdd_s43`, `gdd_s44`
  (`~/results_dinov3/vit/dinov3-vitl16-pretrain-sat493m/gdd_s*`). Run on the third machine: accuracy numbers
  are used, timing is not.

Command: `SEED=$S TRAIN_ROOT=$L/train TEST_ROOT=$L/test EVAL_SPLITS=val,test RUN_TAG=gdd_s$S python dinov3_forensics_train_v2.py`

## Tests (fixed here)

All with `eval_cross_dataset.py`, the same decision rule and thresholds as for the benchmark-trained models, no
threshold or other choice made on any test set. Output files `cross_gddtrain_s{42,43,44}_*`.

1. GDD's own tests: in-domain, cross-generator, cross-domain, localization.
2. The other direction: the benchmark's test split (`--dataset overhead`), which the GDD-trained model has never
   seen, and the other external sets (fmow-fake-small, RSFAKE subset, UW, DM-AER).

## Reporting

Three-seed mean and sample standard deviation of the same columns as the existing zero-shot table, next to the
numbers of the benchmark-trained model on the same sets. Reported whichever way it comes out: if the GDD-trained
model transfers no better to the benchmark than the benchmark-trained model does to GDD, that is the finding.

## Not done

Training on both families together, or on RSFAKE-1M. Each would be a separate experiment.
