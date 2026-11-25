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
