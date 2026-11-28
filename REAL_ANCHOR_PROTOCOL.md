# Real-anchored matched filter: protocol

Written 2026-11-28, before any number under this protocol was computed. Owner approval: Joe Soundararajan, 2026-11-28 ("approve it and time to run it").
Nothing below runs before that line is filled in and this file is committed.

This experiment is an addition outside the handoff. It was designed after the zero-shot, few-shot,
held-out-generator and label-free results were known, and that is stated wherever it is reported.
One run, no tuning, every cell reported whatever it shows. Script: `real_anchor.py` ->
`real_anchor_metrics.json` (CPU only, cached features, refuses to overwrite).

## Question
The few-shot probe needs labelled manipulated images of the new source and loses much of its gain when the
test generator is missing from them (Table VII). The label-free alternatives fail (Table VIII), and the
one-class score flags 46% to 88% of authentic images of a new source. Can a detector be moved to a new source
with authentic images of that source only, with no manipulated image and no knowledge of its generators?

## Idea
Treat detection as finding a known signal in source-specific background, as the adaptive matched filter does
in hyperspectral target detection.
- Signal: the manipulation signature `delta` = mean feature of the manipulated training images minus mean
  feature of the pristine training images. Training folder only.
- Background: the mean `mu_T` and covariance `C_T` of authentic images of the new source, estimated from
  k real images of that source.
- Score: `s(x) = delta' C_T^-1 (x - mu_T) / sqrt(delta' C_T^-1 delta)`.
  For images drawn like the reference reals, `s` is approximately standard normal, so a threshold of 1.645
  gives a nominal 5% false-alarm rate without any manipulated image (constant false-alarm rate).

Directions in which authentic images of the new source vary strongly (city, sensor, resampling) are divided
out by `C_T^-1`, which is the failure diagnosed for the zero-shot model (its score orders UW images by city).

## Data
The eight outside test sets of the zero-shot evaluation, with the pools and held-out halves of
`fewshot_splits.json` and `fewshot_gdd_splits.json`, unchanged. Reference real images come only from the pool.
Every reported number is on the held-out half. No manipulated image of an outside set is used anywhere.
Own data: training folder (`own_train`) for `delta` and the source statistics; test folder (`own_test`) as a
sanity row.

## Features
The cached features of the few-shot experiment: final-layer CLS token, 384x384, /255, one view, L2-normalised.
No further normalisation after centring, so every score is linear in the feature.
Extractors: `full_s42/43/44`, `web_s42/43/44`, `sat_orig`, `web_orig`.

## Scores (complete list; nothing else is tried under this protocol)
| name | mean | covariance | uses |
|---|---|---|---|
| `no_anchor` | pristine training images | pristine training images | no image of the new source (reference) |
| `anchor_mean` | k target reals | pristine training images | ranking identical to `no_anchor` by construction; only the operating point moves |
| `anchor_matched_filter` (primary) | k target reals | k target reals | the method |
| `anchor_rx` | k target reals | k target reals | squared Mahalanobis distance; no signature (one-class reference) |

Covariances: `sklearn.covariance.LedoitWolf` (no free parameter), also when k is below the feature dimension.
k in {4, 16, 64, 256} real images, 10 draws per k (`RandomState(seed).choice(pool_real, k)`, seeds 0-9, the
same real shots as the few-shot probe), plus the whole pool of real images (one fit). A k larger than the
pool's real images is skipped.

## Metrics
- AUC of real against manipulated on the held-out half.
- Detection rate and false-alarm rate at the fixed threshold 1.645 (not for `anchor_rx`, which has no
  calibrated threshold; AUC only).
- Per generator or manipulation type: that group alone against all held-out real images, and the mean over
  groups, as in Table VII. Since no manipulated image is used, every group is a left-out group.
- Mean and standard deviation over the 10 draws; for the fine-tuned extractors also over seeds 42/43/44.

## Comparisons to state (existing numbers, same held-out halves)
- Zero-shot fine-tuned model (Table V) and the probe trained on the training folder (Table VIII).
- Few-shot probe with k labelled images per class, seen and left-out generators (Tables VI and VII).
- One-class score with training reals as reference (Table VIII).

## Pre-declared headline
`anchor_matched_filter` at k = 64 on `sat_orig` and on the three-seed mean of `full`, AUC per set, next to the
zero-shot AUC and the left-out few-shot AUC at k = 64. Whether it is called a method in the paper depends on
that cell; the rule is fixed now: it is presented as a contribution only if it exceeds the zero-shot AUC on at
least six of the eight sets for one of the two extractors. Otherwise it is reported as one more negative
result in the label-free subsection, with the full table in the supplement.

## Limits to state in the paper
- Designed after the other results were known.
- The signature comes from one generator family; nothing guarantees that it points the same way for others.
- With k below the feature dimension (1024) the covariance is dominated by shrinkage, and the nominal 5%
  false-alarm rate will be exceeded; the measured rate is reported next to the nominal one.
- The reference real images and the held-out real images share their source, as in the few-shot experiment.
- fmow-fake-small: 30 real images in the pool, so only k = 4 and k = 16 exist.
- Prior work to check and cite before any novelty claim: adaptive matched filter and RX detectors in
  remote sensing, and target-calibrated detectors for generated images (logit-bias calibration with target
  data, prototype calibration).
