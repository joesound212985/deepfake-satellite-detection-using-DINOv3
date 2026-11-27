# Resampling-factor augmentation run (`fullaug_s42`): protocol

Written 2026-11-27, before any number under it was computed and before any code for it was written.
**NOT RUN. Joe decided on 2026-11-27 to skip this experiment.** No code was written and no number exists under this protocol. Kept as a record only.

This is an addition outside the handoff and a rule 2 change (training augmentation). It was chosen after the
zero-shot, few-shot and label-free results were known, and after label-using diagnostics on the reported
outside sets (section 7). That is stated wherever it is reported. One run, no tuning, every cell reported
whatever it shows. It replaces the C4 entry of `ZEROSHOT_IMPROVE_PLAN_DRAFT.md`; C0-C3 and C5 of that draft
are not part of this protocol and are not approved.

## 1. Question
Does widening the range of resampling factors seen in training change cross-dataset detection, with the
model, losses, learning rates, schedule, sampler and data unchanged?

Background, from the code: training images (512x512) are resized to 384 with `INTER_AREA`, and 60% of samples
then get a random-resized crop of the 384 image (area 0.7-1.0, bilinear back to 384). The net factors are
0.75-0.90; nothing at or above 1.0, nothing below 0.75.

## 2. The change (switch `AUG_SHIFT=1`, default off)
For the training split only, per sample:
- p = 0.40: whole image to 384x384 with `INTER_AREA` (same as the present no-crop path).
- p = 0.45: crop from the native image, side = short side x f, f log-uniform in [0.5, 1.0], aspect 0.9-1.1,
  random position, to 384x384 with `INTER_AREA` (the kernel the evaluation script uses). Net factor 0.75-1.5.
  Tampered images: the crop is redrawn up to 8 times until it keeps at least 50% of the mask pixels,
  otherwise the whole image is used.
- p = 0.15: whole image shrunk by s ~ U[0.5, 1.0] with `INTER_AREA`, then to 384x384 with `INTER_AREA`.
- Masks follow the image with `INTER_NEAREST`. `has_mask` is computed as now (whole-image mask at 384).
- Colour jitter, blur / noise, JPEG, flips and rotations follow, unchanged and in the present order.

What this really changes in `dinov3_forensics_train_v2.py` (all of it; nothing else is touched):
1. With the switch on, training images are read from disk at native size for every sample instead of from the
   384 RAM cache.
2. The present p = 0.6 crop (area 0.7-1.0 of the 384 image) is replaced by the three-way draw above.
3. The crop kernel changes from `INTER_LINEAR` to `INTER_AREA`.
4. The number of random draws per sample changes, so the run is not sample-paired with `full_s42`.
Validation, test, model, losses, learning rates, epochs, sampler, checkpoint rule (validation macro-F1): unchanged.

## 3. Checks before the run (source data only, CPU)
- Switch off: for 200 fixed training indices and fixed seeds, the edited script returns arrays identical to
  the committed script. If not, stop.
- Switch on: 2000 samples drawn; the observed branch shares, crop sides and mask-retention fallbacks are
  printed and saved. No model is involved.
- The diff is shown to Joe before the run. The script is not edited while any training that imports it is
  running on that machine.

## 4. Run
`SEED=42 AUG_SHIFT=1 EVAL_SPLITS=val,test RUN_TAG=fullaug_s42 python dinov3_forensics_train_v2.py`
Satellite backbone, all other settings as `full_s42`. Run folder `fullaug_s42` (new; `aggregate_runs.py`
keys rows by folder name, so it cannot merge into `full`). One GPU job at a time. A smoke run of a few
iterations with a separate tag may precede it; its numbers are not reported.

## 5. Scoring (unchanged scripts, new output prefix, nothing overwritten)
- Own test: `test_metrics.json` of the run.
- `eval_cross_dataset.py`, threshold 0.5, whole-image resize, 4 flips: RSFAKE subset, UW, DM-AER matched,
  fmow-fake-small, GDD indomain / crossgenerator / crossdomain / localization, and DM-AER all reals
  -> `cross_fullaug_*_metrics.json`.
- Own test with `--degrade scale:0.5`, `scale:0.75`, `jpeg:30/50/70/90` -> `robust_fullaug_*_metrics.json`.
- Comparison with `full_s42` on the same images: AUC, accuracy, balanced accuracy, detection rate, false-alarm
  rate; paired bootstrap of the differences (2000 resamples within class, seed 0) and exact McNemar, raw
  p-values with the same caveat as elsewhere.

## 6. Reporting
- Every set gets a row, better or worse, next to `full_s42`. Marked "single run".
- No acceptance rule. The own test set is report-only and gates nothing. No outside set selects anything.
- Whether the row goes in the letter or only the supplement is Joe's decision, made after the numbers are
  seen; the text says so.
- More seeds, or autoencoder reconstructions as extra fakes, need a new dated addendum and a new approval.

## 7. Disclosure: what was known when this was written
- All zero-shot results at 0.5 for three checkpoints, the label-free extras, the few-shot results.
- Scratch diagnostics of 2026-11-27 on the outside test sets (some use their labels): thresholds from own
  validation applied to the 8 sets, size-stratified scores, 512-px-only scores, per-generator rankings,
  native-size statistics. Copies: `~/grsl_zeroshot_diag_scratch_2026-11-27/` (outside the repo).
- The factor range 0.75-1.5 matches the band tabulated on the outside sets in those diagnostics. So the
  recipe is not blind to the reported sets: no outside image is used in training or selection, but the design
  was informed by their size statistics.
- Competing explanation on file (a smoothness cue rather than resampling); this run does not test it.

## 8. Cost (estimates)
Laptop epoch time for `plain_s43` is 813 s (its log), so 50 epochs is about 11.3 h; scoring about 1 h more.

## 9. Deviations this adds to `RESULTS.md`
A training row with changed augmentation, chosen after the zero-shot results were known; one seed; the code
changes of section 2; the machine it ran on.
