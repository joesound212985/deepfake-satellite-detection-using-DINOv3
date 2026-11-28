# TruFor on the test split: protocol

Written 2026-11-27 (late evening), before any TruFor number was computed (Joe passed on a review request for a
forensic localizer that needs no training). All results of our own models were known. One run, nothing tuned,
every number reported whatever it shows. Script: `trufor_eval.py` -> `trufor_overhead_metrics.json`
(+ `.per_image.json`); refuses to overwrite.

- Model: TruFor (Guillaro et al., CVPR 2023), authors' code (github.com/grip-unina/TruFor, commit ae54475,
  folder `TruFor_train_test`, config `trufor_ph3`) and authors' released weights `trufor.pth.tar`
  (TruFor_weights.zip, MD5 7bee48f3476c75616c3c5721ab256ff8 as published by the authors). No training, no
  fine-tuning, no change to the code. Licence: GRIP-UNINA, non-profit research use with reference to the authors;
  code and weights are kept outside this repository (`~/external/trufor_src`) and are not redistributed.
- Data: the 3,150 images of the test folder, the same files our models are scored on, read as stored
  (512 x 512 JPEG), full resolution, no resizing, no test-time augmentation. Preprocessing as in the authors'
  `test.py` (RGB / 256, batch size 1).
- Outputs used: the localization map (softmax probability of the forged class) and the image-level score
  (sigmoid of the detection output).
- Localization: map thresholded at 0.5; IoU, Dice and MCC per image with the same formulas as
  `eval_cross_dataset.pix_stats`, averaged over the 888 tampered test images. Ground-truth masks are binarised
  as in our evaluation but kept at 512 x 512 (our models are scored at 384 x 384); this difference is stated.
- Detection: AUC of pristine against manipulated (tampered + fake), and separately pristine against tampered
  and pristine against fake; accuracy, detection rate and false-alarm rate at threshold 0.5.
- To state with the result: TruFor was trained on edited photographs, not on overhead images or on fully
  generated images, and is applied here without adaptation, while our models are trained on the benchmark's
  training folder. It is a reference for an off-the-shelf forensic tool, not a like-for-like comparison.
