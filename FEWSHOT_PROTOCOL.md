# Few-shot adaptation experiment: protocol

Written 2026-11-25, before any few-shot number was computed (Joe approved the experiment the same day).
The zero-shot cross-dataset results of `full_s42` were already known and are the reason for the experiment.
Nothing below is tuned on results; every cell is reported whatever it shows.

## Question
Zero-shot, the fine-tuned model does not transfer to unseen generators. How much is recovered with a
small labelled sample from the new source, and are the fine-tuned features better or worse for this
than the original DINOv3 features?

## Data
The four external sets, exactly the subsets used for the zero-shot evaluation:
RSFAKE-1M subset (`subset_seed42_n500.csv`), UW fake satellite, DM-AER sample (size-matched), fmow-fake-small.

Each set is split once into an **adaptation pool** and a **held-out half** (50/50, stratified by group,
i.e. real / generator / manipulation type; `numpy.random.RandomState(0)`). The split is saved in
`fewshot_splits.json`. Labelled shots come only from the pool. Every reported number is on the held-out half.

## Feature extractors (all frozen during this experiment)
| name | weights |
|---|---|
| `full_s42` | backbone of the full model, seed 42 (satellite pretraining, fine-tuned on the training set) |
| `web_s42` | backbone of the web row, seed 42 (web pretraining, fine-tuned on the training set) |
| `sat_orig` | original DINOv3 ViT-L/16 sat493m, no forensic training |
| `web_orig` | original DINOv3 ViT-L/16 lvd1689m, no forensic training |

Feature = final-layer CLS token (`last_hidden_state[:, 0]`, the vector the model's own class head reads),
one view, 384x384, same preprocessing as every row of the letter (/255, no mean-std), then L2-normalised.

## Adaptation
- Linear probe: `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000)`, real vs manipulated.
  No hyper-parameter search, no validation split, no early stopping.
- k labelled images **per class**, k in {4, 16, 64, 256}; a k larger than the smaller class in the pool is skipped.
- 10 random draws per k (seeds 0-9), uniform over the pool's real / manipulated images.
- Upper reference: the same probe trained on the whole pool (one fit).

## References on the same held-out half
- Zero-shot: the fine-tuned model's own P(manipulated) = P(tampered) + P(fake), threshold 0.5
  (from the saved per-image files of the zero-shot evaluation; `full_s42` and `web_s42`).
- Threshold calibration: same score, threshold chosen on the k shots to maximise balanced accuracy on
  those shots (ties: midpoint of the widest best interval). AUC is unchanged by construction.

## Metrics
AUC (real vs manipulated), accuracy and balanced accuracy at the probe's 0.5 (or the calibrated threshold),
mean and standard deviation over the 10 draws. Per-group detection rates are saved as well.

## Limits to state in the letter
- One seed of each fine-tuned model. fmow-fake-small is 120 images (60 held out): very wide uncertainty.
- The split is within each external set, so "held-out" images share the source and generators of the shots.
  This measures adaptation to a known new source, not generalisation to yet another unseen one.
- `sat_orig` / `web_orig` receive /255 inputs without their own mean-std, like the frozen row of the main table.

## Addendum 2026-11-25 (late evening): GDD and a t-SNE figure
Written before any GDD few-shot number or any t-SNE plot was computed (Joe approved both the same evening).
The GDD zero-shot results were already known.

### GDD few-shot
Same protocol as above, unchanged in every setting, applied to the four GDD test subsets used for the
zero-shot evaluation, each treated as its own set with its own 50/50 split: `gdd_indomain`,
`gdd_crossgenerator`, `gdd_crossdomain`, `gdd_localization` (500 real + 500 locally manipulated).
Run with `python fewshot_adapt.py {extract,analyse} --sets gdd`; results in `fewshot_gdd_metrics.json`,
splits in `fewshot_gdd_splits.json`. The existing `fewshot_metrics.json` is not touched.
The GDD subsets may share real images with one another; each is split and evaluated on its own, so nothing
crosses between pool and held-out half within a reported number.
Feature extraction runs on the laptop (RTX 5090, torch 2.14.1+cu130) because the workstation GPU is full
with a training run; features of the first four sets were extracted on the workstation (cu126).

### t-SNE figure (`tsne_features.py` -> `tsne_features.png`)
Illustration only; no number in the letter is derived from it.
- Extractors: `sat_orig` (top row) and `full_s42` (bottom row). Sets (columns): RSFAKE-1M subset, UW,
  DM-AER (size-matched), GDD cross-generator. fmow-fake-small is left out (120 images).
- Per set: up to 500 real and 500 manipulated images, drawn with `numpy.random.RandomState(0)` from the
  whole set, the same images for both extractors.
- L2-normalised CLS features -> PCA to 50 dimensions -> `sklearn.manifold.TSNE(n_components=2,
  perplexity=30, init="pca", random_state=0)`, fitted separately per panel.
- One run, no parameter search; the figure is kept whatever it shows.

## Addendum 2026-11-26: other model seeds
Written before seeds 43/44 of the full model existed (Joe approved the same morning).
Same protocol, same splits, same shot draws, all eight sets (four external + four GDD subsets), with the
fine-tuned backbones of the other seeds as extractors: `full_s43`, `full_s44`, `web_s43`, `web_s44`.
Run with `python fewshot_adapt.py {extract,analyse} --sets seeds` after the training queue has finished;
results in `fewshot_seeds_metrics.json`. Purpose: spread of the few-shot numbers over model seeds
(report mean and std over seeds 42/43/44 of the per-seed means). Earlier result files are not touched.
The zero-shot reference rows in that file are those of seed 42 (the only seed scored zero-shot on every set).
