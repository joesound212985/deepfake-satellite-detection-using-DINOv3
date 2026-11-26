# Label-free transfer experiments: protocol

Written 2026-11-26, before any of these numbers was computed (Joe approved the same day). They are additions
outside the handoff, chosen after the zero-shot and few-shot results were known; that is stated wherever they
are reported. One run each, no tuning, every cell reported whatever it shows. No labelled image of an outside
dataset is used anywhere below. Script: `zeroshot_extras.py` -> `zeroshot_extras_metrics.json`.

Common: L2-normalised final-layer CLS feature (384x384, /255, one view), the same features as the few-shot
experiment. Extractors: `full_s42`, `web_s42`, `sat_orig`, `web_orig`. Outside sets: the four external sets and
the four GDD subsets, exactly as in the zero-shot evaluation. Own data: the training folder (10,000 images:
the training and validation images of the main runs) and the test folder (3,150 images).
Features of the own data are extracted on the laptop with `python fewshot_adapt.py extract --sets own`.

## A. Clustering of the unlabelled target set (transductive)
For each outside set and each of `full_s42`, `web_s42`: `KMeans(n_clusters=2, n_init=10, random_state=0)` on
the features of all images of that set (labels not used). The cluster whose images have the higher mean
zero-shot score P(tampered)+P(fake) of the same model is called "manipulated".
Reported: accuracy, balanced accuracy, detection rate, false-alarm rate. Also reported, labelled as an oracle:
the accuracy with the better of the two possible cluster namings.
This needs a batch of unlabelled images from the target source, with both classes present.

## B. Linear probe on a frozen backbone, trained on our own training data
For each extractor: `LogisticRegression(C=1.0, max_iter=2000)`, pristine vs manipulated (tampered + fake), fitted
on all 10,000 images of the training folder. No model selection, so no validation split is needed.
Applied unchanged to our test folder and to every outside set. Reported: AUC, accuracy / detection rate /
false-alarm rate at 0.5.

## C. One-class score from real images only
For each extractor: the pristine images of the training folder are split 90/10 (`RandomState(0)`).
Score of an image = mean cosine distance to its 5 nearest neighbours among the 90% (reference set).
Threshold = 95th percentile of the scores of the other 10% (own pristine images, never an outside image).
Reported on our test folder and every outside set: AUC, detection rate and false-alarm rate at that threshold.
