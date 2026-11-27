DRAFT, not approved, nothing run under it.

JUDGE PLAN: improving GRSL zero-shot accuracy

Nothing was run for this plan except one `ls` to confirm input files exist. Every number is copied from the map reports, with the source file named. G/ = /home/adminuser/grsl/, Z/ = /tmp/claude-1000/-home-adminuser/2ee7f099-ffb7-49ce-9137-8086bf6cccca/scratchpad/zs/, F/ = /home/adminuser/data/features/fewshot_cls384/.

Verdict: a large zero-shot gain is not supported by any file. The failure is separability, not threshold, and the only cheap levers (a source-calibrated threshold and scale-consistent input) have diagnostic ceilings of a few points to roughly 0.6-0.78 balanced accuracy. Larger gains need retraining (owner approval, rule 2) or labelled outside data, which is no longer zero-shot.

## A. Diagnosis (full_s42, 4-view, threshold 0.5)

Format: AUC / acc@0.5 from G/cross_<set>_metrics.json `overall`; oracle best balanced accuracy from Z/diag_out.txt (uses test labels, diagnostic only).

1. fmow-fake-small: 0.4503 / 0.5000, oracle 0.5167. Separability; 60 of 60 fakes missed, local edits, n=120.
2. UW: 0.5001 / 0.4991, oracle 0.5706. Separability; the score orders by city (authentic Seattle vs Beijing AUC 0.949), not authenticity.
3. RSFAKE: 0.5837 / 0.5342, oracle 0.5703. Separability plus a size confound; per-generator AUC 0.268-0.816 (G/cross_detail.json).
4. GDD indomain 0.5357 / 0.5347 (oracle 0.5553); localization 0.5680 / 0.5240 (oracle 0.5630). Separability; 1024 px FLUX.2 ranked backwards (0.179-0.196), local edits 0.44-0.57.
5. GDD crossgenerator 0.6483 / 0.5300 (oracle 0.6095); crossdomain 0.7064 / 0.5380 (oracle 0.6460). Mixed: moderate ranking, threshold far too high.
6. DM-AER matched: 0.7798 / 0.4036, balanced 0.5530, oracle balanced 0.7161. Mainly calibration and prior shift (494 real / 1000 fake). All-reals variant: 0.6647 / 0.5325 (G/cross_allreals_dmaer_metrics.json).
7. Common pattern: 95-100% of sat/web errors are missed fakes; 66-100% of outside fakes have p_manip < 0.1 (Z/diag_out.txt). Training fakes come from one generator (`model_name` MB16, Z/tr_meta.out) at one resampling factor (512 to 384).
8. Resampling shortcut: RSFAKE reals flagged 0.1308 at 256 px vs 0.0204 at 512 px; 1024 px fakes 0 of 300 and 0 of 333 flagged (Z/analyse.out). Images already 512x512 have AUC 0.634-0.837 but balanced accuracy at 0.5 of only 0.547-0.583 (Z/nr_tiles.out, Z/rev_trainscale.out).

## B. Ranked candidates

| # | Candidate | When | Owner approval |
|---|---|---|---|
| C0 | Seed spread of the unchanged pipeline | GPU inference after queue | OK as evaluation outside handoff |
| C1 | Source-calibrated operating point (q95) | CPU today (single view); GPU ~88 s per checkpoint for 4-view | Protocol addendum + OK |
| C2 | Scale-consistent tiled input | GPU inference after queue | Yes (changes evaluation preprocessing) |
| C3 | C1 + C2, pre-declared headline | Same as C2 | Yes |
| C4 | Retrain with resampling-factor augmentation | New training | Yes, rule 2 |
| C5 | Retrain with autoencoder reconstructions as extra fakes | New training | Yes, rule 2 + package install + download |

**C0. Seed spread (precondition, not an improvement).**
- Recipe: score full_s43, full_s44, web_s43, web_s44 with unchanged eval_cross_dataset.py on the 8 primary sets.
- Cost: full_s42 took 3058.6 s for the 8 sets (sum of `wall_sec`); four checkpoints is about 3.4 h (proposal arithmetic).
- Selection: none.

**C1. Source-calibrated operating point.**
- Recipe: second threshold = 95th percentile of P(manip) on the 502 own-validation pristine images, applied unchanged to every set and the own test. The 0.5 row stays. q99 is a declared secondary.
- Reportable version uses the 4-view pipeline on the validation ids (about 88 s GPU per checkpoint, proposal arithmetic from 88.3 ms/image). The CPU version is single view on cached CLS.
- Selection: none; formula fixed, quantile by precedent (ZEROSHOT_EXTRAS_PROTOCOL.md:31).
- Not blind: Z/thr.out already shows it. Single view, full_s42, q95 = 0.01168: RSFAKE 0.580, UW 0.505, DM-AER 0.638 (balanced 0.692), GDD 0.555 / 0.579 / 0.599 / 0.543, fmow 0.492; own test 0.873 with false alarms 0.218.
- Weakness: AUC is unchanged, and the nominal 5% false-alarm rate becomes 0.2177 on the own test (Z/nr_srcval.out).

**C2. Scale-consistent input.**
- Recipe: resample both sides by 0.75 with INTER_AREA, no aspect change. Take 384x384 tiles, ceil(side/384) per axis, evenly spaced, at most 16 per image. Reflect-pad sides below 384. 4 flips per tile. Image score = mean softmax over tiles and flips. Threshold 0.5.
- Identical to baseline on 512x512 images (Z/misc.out: "scale:0.75 then 384 == direct 384 (INTER_AREA): True"), so own test and matched DM-AER are unchanged.
- Cost: 40216 tiles, 59.2 min per checkpoint (Z/nr_tiles.out). New script with a new output prefix.
- Selection: one rule fixed a priori. Source data cannot choose between input rules.
- Risk: it may lower pooled numbers by removing the shortcut (diffusion_sat_256 0.8042 becomes 0.5780 against same-size reals, Z/analyse.out). UW rests entirely on the padding branch.

**C3. C1 + C2 (headline).** Threshold = q95 of validation pristine scored through the C2 pipeline, which equals the C1 4-view threshold because the validation images are 512 px. No extra GPU cost.

**C4. Resampling-factor augmentation (`fullaug_s42`).**
- Recipe: `AUG_SHIFT=1`, default off. p=0.40 unchanged path; p=0.45 crop with side log-uniform in [256, 512] native px, aspect 0.9-1.1, to 384; p=0.15 whole image shrunk by U[0.5, 1] then the standard resize. Replaces the p=0.6 random-resized crop. Mask-retention redraw for tampered images. Everything else unchanged.
- Cost: about 9.9 h per run including scoring (8:48:32 wall for full_s42 plus 3058.6 s plus 0:10:30).
- Selection: recipe fixed in advance; checkpoint on validation macro-F1 as always. Gate and acceptance on shifted views of the source validation split only (standard, `--degrade scale:0.5`, centre 256 crop, `--degrade jpeg:75`).

**C5. Autoencoder reconstructions as extra fakes (`fullaugrec_s42`, replicate `_s43`).**
- Recipe: 2303 of the 4462 train-split pristine images (seed 42) passed once through `stabilityai/sd-vae-ft-mse` (encode, decode), saved as JPEG at the source `quality_factor`, 4:2:0, labelled fake without mask. Samples per epoch held at 9000. Run on top of C4.
- Cost: about 9.9 h per run plus reconstruction time (not found). `diffusers` is not installed.
- Selection: fixed in advance, one run plus one replicate.
- It is the only candidate that attacks the dominant error (single training generator), and it weakens the "trained on one dataset" claim.

**Excluded from the list.**
- Temperature scaling and validation-optimal threshold: no-op (Z/nr_srcval.out: own-test acc 0.9076 vs 0.9079).
- sat + web ensembling: 88-100% of missed fakes are shared (Z/agree_out.txt).
- Dropping TTA: post hoc.
- Best cell of extras A/B/C: selection on test data.
- Leave-one-set-out probe on outside labels: not zero-shot, and conflicts with RUNLOG.md:213 for GDD.
- Transductive methods.
- Weight interpolation between frozen and full: new weights, premise not met.

## C. Draft ZEROSHOT_IMPROVE_PROTOCOL.md

```markdown
# ZEROSHOT_IMPROVE_PROTOCOL.md

Written 2026-11-27, before any number under this protocol was computed.
Owner approval: [PENDING: name, date]. Nothing below runs before that line is filled in and this file is committed.

These experiments are additions outside the handoff. They were chosen after the zero-shot,
few-shot and label-free results were known, and after the post-hoc diagnostics listed in
section 9. That is stated wherever they are reported. One run each, no tuning, every cell
reported whatever it shows.

## 1. Question
Can the cross-dataset ("zero-shot") detection figures be improved without using any image,
labelled or unlabelled, of an outside dataset for any choice?

## 2. Definition kept
"Zero-shot, source-only": every weight, threshold, preprocessing rule and model choice is
fixed from the training folder and its validation split (VAL_SPLIT 0.1, VAL_SEED 42). The
rule is per image, label-free and identical for every set, including the own test set.
The existing rows (threshold 0.5, whole-image resize to 384, 4 flips) stay in the letter
unchanged and remain the reference.

## 3. Data roles
- Selection data: own validation split only (1000 images: 502 pristine, 262 tampered, 236 fake).
- Development set: none. No outside set is used to choose anything.
- Reported sets (all are test sets; none may inform a choice): RSFAKE subset (5000/5000),
  UW (4032/4032), DM-AER matched (494/1000), fmow-fake-small (60/60), GDD indomain (750/750),
  crossgenerator (1000/1000), crossdomain (1000/1000), localization (500/500).
  Sensitivity: DM-AER all reals (1000/1000). Own test set at the same operating point.
- Confirmation set [OWNER DECISION, section 11]: a second, disjoint RSFAKE draw from the
  official test split (seed 43, excluding ids in subset_seed42_n500.csv), built as in
  prepare_rsfake_subset.py. Scored once, after everything else, with the headline
  configuration and the baseline only.
- Licences: RSFAKE-1M non-commercial research only; DM-AER CC BY-NC-SA 4.0; GDD CC BY 4.0,
  test splits only, no training on it.

## 4. Candidates (complete list; nothing else is tried under this protocol)

### Inference only (weights unchanged)
- C0. Baseline seed spread. Unchanged eval_cross_dataset.py on full_s43, full_s44, web_s43,
  web_s44, 8 sets. Not an improvement; gives the noise floor.
- C1. Source-calibrated operating point. Threshold t95 = 95th percentile of P(manipulated)
  on the own-validation pristine images, computed with the same views as the scored
  pipeline, per checkpoint, applied unchanged everywhere. Secondary: t99.
  - C1a (CPU): single-view class head on cached CLS features, validation and targets both
    single view. Declared secondary because it is not the letter's 4-view pipeline.
  - C1b (GPU): validation ids scored with the 4-view pipeline; threshold applied to the
    stored 4-view per-image scores. This is the reportable C1 row.
- C2. Scale-consistent input at threshold 0.5. Resample both sides by 0.75 with INTER_AREA,
  no aspect change. 384x384 tiles, ceil(side/384) per axis, origins evenly spaced from 0 to
  side-384; if the count exceeds 16, a 4x4 evenly spaced grid. Sides below 384 after
  resampling are reflect-padded, image centred. 4 flips per tile. Image probabilities =
  mean of the softmax over tiles and flips; P(manipulated) = P(tampered) + P(fake).
  Declared secondary aggregator: maximum over tiles. No other variant.
- C3. C1b + C2 together. PRE-DECLARED HEADLINE.

### New training (rule 2; each needs its own explicit approval line)
- C4. `fullaug_s42`: resampling-factor augmentation behind `AUG_SHIFT=1` (default off).
  p=0.40 unchanged path; p=0.45 square-ish crop, side log-uniform in [256, 512] native px,
  aspect 0.9-1.1, INTER_AREA to 384; p=0.15 whole image shrunk by s ~ U[0.5, 1] with
  INTER_AREA, then the standard resize. Replaces the existing p=0.6 random-resized crop.
  Tampered images: crop redrawn up to 8 times until it keeps at least 50% of mask pixels,
  else full image. All other augmentation, losses, learning rates and schedule unchanged.
- C5. `fullaugrec_s42` and replicate `fullaugrec_s43`: C4 plus 2303 autoencoder
  reconstructions (stabilityai/sd-vae-ft-mse, encode-decode only) of train-split pristine
  images (random draw, seed 42), saved as JPEG at the source quality_factor, 4:2:0, class
  "fake", no mask. Samples per epoch held at 9000. Validation and test untouched.
- Checkpoint selection for C4/C5: validation macro-F1, as always. No hyper-parameter search.
- C4/C5 are scored with the unchanged pipeline (0.5, whole-image resize) and with C3.

## 5. Metrics
- Primary: balanced accuracy at the declared operating point.
- Always alongside: AUC, detection rate, false-alarm rate, plain accuracy; own-test
  accuracy and false alarms at the same operating point.
- Aggregate (fixed here): dataset-macro balanced accuracy over RSFAKE, UW, DM-AER matched
  and GDD (mean of its three detection splits). fmow-fake-small and GDD localization are
  reported but not in the aggregate.
- Secondary aggregate: dataset-macro AUC over the same four.
- Intervals: paired bootstrap of the differences against the baseline on the same images,
  2000 resamples within class, seed 0 (as analyze_cross_detail.py); exact McNemar on
  correctness (as analyze_extras.py); Holm correction over sets. GDD crossgenerator and
  crossdomain share their 1000 fakes and are resampled jointly.
- Seeds: full model seeds 42/43/44, mean and std; web as secondary.

## 6. Decision rule for adopting a candidate
The headline C3 is adopted as an additional row only if all hold, on the full model:
1. Aggregate balanced accuracy of C3 exceeds the baseline at 0.5 on each of the three
   seeds, by at least [OWNER: minimum relevant effect].
2. Aggregate AUC of C2 is not lower than the baseline's (mean over seeds).
3. No reported set is worse in balanced accuracy by more than [OWNER: margin].
4. Own-test accuracy and false alarms at the same operating point are printed in the same
   table row. [OWNER: maximum acceptable own-test false-alarm rate, or "report only"].
5. If the confirmation set is used: the sign of the balanced-accuracy difference agrees.
For C4/C5 additionally:
6. Source-only gate: on the shifted validation views (standard; `--degrade scale:0.5`;
   centre 256 crop to 384; `--degrade jpeg:75`) the minimum balanced accuracy is not below
   the baseline's.
7. Own-test 3-class accuracy not more than [OWNER: margin] below the full row.
8. The replicate seed agrees in sign with the first.
If C3 fails, C1b and C2 are still reported and nothing further is tried under this
protocol. A new attempt needs a new dated addendum listing every earlier attempt.
"The proposed model" is not redefined by any outside result; that is the owner's decision
on the criteria above.

## 7. Checks reported whatever they show
- Regression gate: with the new switches off, the new script reproduces the stored
  cross_*_metrics.json and overhead_full_metrics.json values exactly.
- AUC by native-size bin; per-generator table; DM-AER re-scored with `--degrade jpeg:75`.
- Padding stress on the source: own-validation images as centre 256x256 crops through C2
  (diagnostic only, not a selection step).

## 8. Reporting commitment
Every candidate in section 4 gets a supplement row with identical columns, improved or
not, including failures and the own-test cost. The letter states how many configurations
were tried and that they were "chosen after the zero-shot results were known".

## 9. Known before this protocol was written (disclosure)
None of the following may be used to select anything:
- Three checkpoints at 0.5 on all sets (cross_*_metrics.json).
- Label-free extras A/B/C over four extractors (zeroshot_extras_metrics.json).
- Few-shot results (fewshot_metrics.json, fewshot_gdd_metrics.json).
- Oracle thresholds and 5%-false-alarm points (cross_detail.json).
- Post-hoc scratch diagnostics of 2026-11-27 on the outside test sets, not results:
  validation q95/q99 thresholds applied to all 8 sets, single view vs 4 views, P(fake) or
  P(tampered) alone, sat+web mean/max/OR and 3-model mean, size-stratified scores,
  512-px-only scores, an 11-variant transfer check.
  C1, C2 and C3 are therefore not blind; C1's single-view outcome was already seen.

## 10. Mechanics
- New scripts with their own output prefixes (`zsi_*`), which refuse to overwrite.
- Training rows named so aggregate_runs.py cannot merge them into `full`.
- One GPU job at a time, after run_fewshot_seeds.sh has finished. The queue script's
  command line must not match "run_everything.sh|run_extra_seeds" (run_web_seeds.sh:14).
- A RUNLOG row naming the approval; commit this file before the first number.

## 11. Deviations this adds to the letter
- A second operating point (source-calibrated threshold) next to 0.5.
- A second input rule (scale-preserving tiles) next to the whole-image resize.
- Zero-shot scoring of seeds 43/44.
- If approved: new training rows with changed augmentation and added reconstructed fakes;
  the text must then say reconstructions of the model's own training reals were added.
- If approved: a second RSFAKE draw used as confirmation data.
```

## D. What can run today on CPU

Nothing below should produce a reportable number before the owner approves and the protocol is committed (rule 5). During this task ~/grsl is read-only, so outputs go to Z/ only.

Inputs confirmed present by `ls`:
- F/{full_s42,web_s42,sat_orig,web_orig}/<set>.npy and <set>.ids.json for sets dmaer, fmow, gdd_indomain, gdd_crossgenerator, gdd_crossdomain, gdd_localization, rsfake, uw, own_train, own_test (listing checked for full_s42; the other three extractor folders are as reported by the pipeline map).
- G/cross_<set>_metrics.per_image.json, G/cross_web_<set>_…, G/cross_frozen_<set>_…, G/overhead_full_metrics.per_image.json, G/overhead_web_metrics.per_image.json. Fields: id, group, label, p_pristine, p_tampered, p_fake, p_manip, pred_class.
- Templates already written by the map agents: Z/thr.py and Z/nr_srcval.py (head from checkpoint, validation split rebuild), Z/rev_endpoint.py (aggregate and bootstrap), Z/sizes.json (native sizes).

Not confirmed by me:
- The exact state-dict key path of `cls_head.weight` / `cls_head.bias` inside best_checkpoint.pt, and the logit class order. Take both from Z/thr.py, which reproduced the stored validation score (0.9941 full, 0.9981 web).
- The Python interpreter. The training proposal names ~/Desktop/.venv.

**D1. C1a, single-view source-calibrated threshold.** `OMP_NUM_THREADS=2 python Z/zsi_c1a.py`
```
for ext in full_s42, web_s42:
  W, b = cls_head.weight/bias from ~/results_dinov3/vit/<backbone>/<ext>/best_checkpoint.pt
         (torch.load(..., map_location="cpu", mmap=True); key path as in Z/thr.py)
  X = np.load(F/ext/own_train.npy); ids = json.load(F/ext/own_train.ids.json)
  val ids: rebuild as dinov3_forensics_train_v2.py:3304-3313
           (random.Random(42).shuffle(ids); first 10%); assert counts 502/262/236
  p = softmax(X @ W.T + b); assert val macro-F1 equals the checkpoint's stored score
  t95, t99 = quantiles of p_manip on val pristine
  for set in the 8 sets + own_test:
     p_manip from F/ext/<set>.npy; label from <set>.ids.json
     report AUC, bal.acc, acc, det, FA at 0.5, t95, t99; bootstrap 2000 within class, seed 0
  write Z/zsi_c1a_metrics.json; exit if it exists
```
This repeats what Z/thr.out already shows, adding web_s42, balanced accuracy and intervals. Expect no new information beyond that.

**D2. Baseline aggregates and paired-test machinery.** `OMP_NUM_THREADS=2 python Z/zsi_baseline.py`
```
read G/cross[_web|_frozen]_<set>_metrics.per_image.json
compute dataset-macro bal.acc and AUC (RSFAKE, UW, DM-AER matched, mean of 3 GDD detection splits)
paired bootstrap + McNemar functions, tested on sat vs web (ids identical per pair)
cross-check against Z/rev_endpoint.out section 2 (full 0.5301, web 0.5373, frozen 0.5765)
```

**D3. Confound table for the protocol appendix.** Join Z/sizes.json to the per-image files by id; AUC by native-size bin per set, as Z/size_out.txt.

**D4. Code-only preparation, no numbers.** Draft `zsi_tiles_eval.py` (C2) and the validation-id filter for C1b in scratch; unit-test the tile geometry on synthetic arrays on CPU. Tile counts to expect are in Z/nr_tiles.out.

Cannot run today: C1b, C2, C3, C0 (GPU inference); C4, C5 (training). Earliest GPU slot is after plain_s44 (about 16:05-16:16 on 2026-11-27 by the history map's arithmetic), then stage 3 and the few-shot seed extraction.

## E. Owner decisions

1. Approve this protocol as an outside-handoff experiment? Recommend yes for C0-C3; they change no weights.
2. Is a modest gain with a disclosed own-test cost worth a row? At q95 the single-view own-test false alarms were 0.218 (Z/thr.out). Recommend an added row, never a replacement for the 0.5 row.
3. Headline: C3, as pre-declared? Recommend yes, with balanced accuracy as primary metric, since DM-AER matched is 494/1000.
4. Fill the three margins in protocol section 6 (minimum relevant effect, per-set worsening, own-test false-alarm cap). Recommend fixing them before C1b runs; the training proposal suggested 0.02 for the own-test accuracy margin.
5. Approve retraining C4 (rule 2), about 9.9 h? Recommend deciding only after C0-C3 are in; it is the first candidate that changes the model.
6. Approve C5 (install `diffusers`, download the autoencoder, change the training data and the "one dataset" wording), about 2 x 9.9 h plus unmeasured reconstruction time? Recommend only if C4 passes its source-only gate.
7. Build the disjoint RSFAKE confirmation draw (seed 43; reals need fMoW parents re-fetched)? Recommend yes; it is the only evidence not designed on already-inspected images.
8. Use the GDD official validation split as a development set (download; two GDD rows stop being zero-shot; conflicts with RUNLOG.md:213)? Recommend no; keep source-only selection.
9. Run a leave-one-set-out probe on outside labels? Recommend no for this letter; it is a different claim.
10. Score web_s43/s44 as well as full_s43/s44 in C0 (about 3.4 h for four against about 1.7 h for two)? Recommend all four.
11. May C1a be run today on CPU into Z/ before sign-off? Recommend waiting; its outcome is already known from Z/thr.out and rule 5 says protocol first.

No protocol file was saved; the draft in section C exists only in this message.

---

# Adversarial review of this plan

ADVERSARIAL REVIEW OF "JUDGE PLAN: improving GRSL zero-shot accuracy"

Paths: G/ = /home/adminuser/grsl/, Z/ = /tmp/claude-1000/-home-adminuser/2ee7f099-ffb7-49ce-9137-8086bf6cccca/scratchpad/zs/, F/ = /home/adminuser/data/features/fewshot_cls384/. Nothing was written anywhere. No GPU was used. Every number below was read from the named file or printed by a short CPU command; hand arithmetic is marked.

## Blockers

1. **Own test split is used as an accept/reject criterion (rule 4).** Protocol 6.4 asks for a "maximum acceptable own-test false-alarm rate" and 6.7 says "Own-test 3-class accuracy not more than [margin] below the full row". Both gate adoption of a threshold or a retrained model on G/overhead_* test data. The own-test outcome at q95 is also already known (FA 0.2177, det 0.9573, Z/nr_srcval.out).
   - Fix: make own-test strictly report-only. Put any gate on the validation split (standard and shifted views) and delete the owner cap in 6.4.

2. **C4 silently changes more than the crop (rule 2), and "everything else unchanged" is false.** In G/dinov3_forensics_train_v2.py every image is resized to 384 with INTER_AREA before caching (`__getitem__` and the preload path, about lines 2255-2283). The existing crop at lines 2314-2318 then works on the 384 cache and resizes with INTER_LINEAR (line 2078).
   - "Crop side log-uniform in [256, 512] native px, INTER_AREA to 384" therefore needs a native-resolution cache: 6.59 GiB images + 2.2 GiB masks against 3.71 GiB now (Z/tr_counts.out).
   - It also changes the interpolation kernel (LINEAR to AREA) and the order of resize against blur, noise and JPEG augmentation.
   - It edits the training script itself, and protocol section 7 has no regression gate for training with `AUG_SHIFT=0` (only for the eval script).
   - Fix: list each change in the approval request, and add a source-only check that the switch-off path leaves the training input unchanged.

3. **Diagnosis A7 is wrong, which weakens the motive for C2 and C4.** A7 says training uses "one resampling factor (512 to 384)". The code already applies a p=0.6 random-resized crop with area scale (0.7, 1.0) (line 2314-2317). Z/tr_counts.out gives the net native-to-input factor as 0.750 to 0.896.
   - Fix: restate it as "factors 0.75-0.90, with an extra bilinear upsample on 60% of samples; nothing at or above 1.0 and nothing below 0.75".

4. **Protocol sections 1-2 claim no outside image, "labelled or unlabelled", informs any choice; that is untrue for C2 and C4.**
   - C4's range [256, 512] px equals the factor band 0.75-1.5 that Z/tr_cover.out tabulates on the outside test sets (column "inside [0.75,1.5] both sides"; UW median factor 1.500).
   - C2's 16-tile cap, padding branch and tile grid come from Z/nr_tiles.out and Z/sizes.json, which are size statistics of the test images.
   - The move to tiling came from label-using "512x512 only" AUC and oracle tables on the test sets (Z/nr_tiles.out, Z/rev_trainscale.out).
   - Fix: reword the definition to "no outside image is used at run time; the design was informed by size statistics and label-using diagnostics of the reported sets", and call C2-C5 exploratory rather than "pre-declared headline".

5. **The aggregate endpoint and decision rule 6.1 are not a test.** From Z/thr.out (single view), my arithmetic on the proposed 4-dataset macro balanced accuracy:

   | Checkpoint | at 0.5 | at q95 |
   |---|---|---|
   | full_s42 | 0.536 | 0.589 |
   | web_s42 | 0.537 | 0.584 |

   - The aggregate leaves out fmow, where q95 is known to hurt (full 0.492 against 0.500), and GDD localization (0.543).
   - The three owner margins (minimum effect, per-set worsening, FA cap) would be filled in after these outcomes are known. Web UW already goes from 0.497 to 0.472 at q95.
   - Fix: fix the margins without reference to Z/thr.out, or drop the rule and report C1 as a descriptive second operating point. Report the 8-set macro next to the 4-dataset one.

6. **C0 cannot use full_s44 yet, and the queue statement is wrong.** The training process now running has `RUN_TAG=full_s44 SEED=44` (/proc/1180624/environ; RUNLOG.md:294 start 2026-11-26T22:54:41). `full_s44/best_checkpoint.pt` exists (mtime Nov 27 02:06) with no test_metrics.json, so it is a mid-training checkpoint. The plan says the slot is "after plain_s44"; that is not what is running.
   - Fix: C0 and C1b must require `test_metrics.json` to exist before reading any checkpoint.

## Should-fix

7. **The q95 primary / q99 secondary ordering is contaminated.** Z/thr.out shows q95 above q99 in balanced accuracy on 7 of 8 outside sets for full_s42 (fmow is the exception: 0.492 against 0.500). Q99 is better on the own test (accuracy 0.914 against 0.873).
   - The precedent citation is real: ZEROSHOT_EXTRAS_PROTOCOL.md:31 reads "Threshold = 95th percentile...".
   - Fix: give both quantiles equal status in the table and cite the precedent as the only reason for the ordering.

8. **C1 is a move along the ROC curve, not a better detector; the wording should say so.** At q95 for full_s42 the false-alarm rates are: own test 0.218, RSFAKE 0.271, UW 0.440, DM-AER 0.150, GDD 0.149 / 0.141 / 0.101 / 0.148 (Z/thr.out).
   - The nominal 5% becomes 21.8% even in-domain. Validation images are all generator `MB16`; the test set also has `G17` and `MB18`, including for pristine rows (Z/tr_meta.out).
   - Q95 rests on about 25 of 502 images in a steep tail (q90 0.006083, q95 0.01168, q99 0.12, Z/nr_srcval.out), so it may move a lot between seeds.
   - The bootstrap in section 5 resamples target images only. Fix: also resample the 502 validation pristine images, or state that the interval is conditional on the threshold.

9. **Sections 6 and 8 contradict each other.** Section 6 says C3 is "adopted as an additional row only if" outside results pass; section 8 says every candidate gets a row regardless. Adoption conditioned on outside test results is selection on test sets.
   - Fix: define "adopt" as main-text placement only, and state that this placement was chosen on the reported sets.

10. **Owner decision E5 makes C4 adaptive.** "Decide C4 only after C0-C3 are in" conditions the retraining decision on outside test outcomes. Fix: decide C4 and C5 now, or disclose the dependence.

11. **The confirmation set is weak and optional.** A seed-43 RSFAKE draw has the same generators and the same size confound as the set already inspected, so it guards only against image-level overfitting.
    - `~/data/rsfake/REAL/fmow/real/test` holds exactly 963 + 3529 + 508 = 5000 files (Z/misc.out), all already used, so new reals need a download. Fake zips for all 10 generators are present.
    - Fix: say what it can and cannot confirm; make it mandatory, or drop decision rule 6.5.

12. **"Source data cannot choose between input rules" (C2) is false and inconsistent with the C4 gate.** G/robust_full_scale0.5_metrics.json already shows source data responding to resampling: own-test det 0.5656 against 0.8932 (Z/tr_endpoints.out).
    - UW rests entirely on the untested reflect-padding branch: 8064 of 8064 images need padding (Z/nr_tiles.out).
    - Fix: turn the section 7 "padding stress" into a source-only gate on validation that must pass before any outside number is produced.

13. **C2's secondary "maximum over tiles" aggregator adds a tile-count confound.** In GDD indomain the 1024 px bin has 300 fakes and 15 reals; in crossgenerator 333 and 24 (Z/size_out.txt:48, 69). Large images get more tiles, so a maximum rises with size, which correlates with label. Fix: drop it, or report it only size-stratified.

14. **C2 is under-specified for localization.** fmow and GDD localization report mask IoU (cross_fmow `overall.localization`), but the plan gives no mask stitching rule, and the 3-class argmax and edge gate are undefined for tiles. Fix: state that C2 is detection-only, or define them.

15. **C5 changes the sampler without saying so.** The sampler is class-balanced with `num_samples=len(sample_weights)` (lines 3357-3369).
    - Adding 2303 reconstructions to class "fake" halves the exposure of the real generator's fakes.
    - "Samples per epoch held at 9000" needs a code change, because the default would become 11303 (my arithmetic).
    - The recipe does not say whether reconstructions get the `sharpen=0.5` that every training fake has while pristine has 0.0 (Z/misc.out), and re-saving creates double JPEG.
    - Fix: list all of these as rule 2 items.

16. **The expected gain of C5 is not supported.** "The only candidate that attacks the dominant error (single training generator)" is a hypothesis.
    - The own test contains unseen generators G17 and MB18, yet fake recall is 741 of 751 (Z/tr_endpoints.out confusion matrix).
    - Z/tr_gen_rank.out points to a smoothness cue instead: Spearman of median hf_ratio against per-group AUC is -0.721 (p=0.0186) on RSFAKE.
    - Training fakes are smoother than pristine (hf_ratio AUC 0.3731, Z/tr_sharp.out). Autoencoder reconstructions are smooth too, so they may reinforce that cue.
    - The groups ranked backwards are the sharp ones: flux2dev 0.179, flux2kleinbase4b 0.196, RSSD_768 0.268.
    - Fix: delete the claim or mark it as a hypothesis.

17. **The executive verdict overstates the ceiling.** "Ceilings ... roughly 0.6-0.78 balanced accuracy" mixes whole-set oracles (0.5167-0.7161, Z/diag_out.txt T1) with oracle thresholds on 512x512-only subsets (0.6095-0.7776, Z/nr_tiles.out).
    - C2 leaves 512x512 images unchanged, so those subset oracles say nothing about C2.
    - No file measures any effect of C2.
    - What the files do show at a non-oracle source threshold (C1, single view) is 0.505-0.692 per set, fmow 0.492.

18. **Citation and number mismatches:**
    - A2 "authentic Seattle vs Beijing AUC 0.949" is not in any Z/ file or G/*.md. I recomputed it from G/cross_uw_metrics.per_image.json: 0.9487 (web: 0.9046). The value is right; the source is missing.
    - A8 "1024 px fakes 0 of 300 and 0 of 333" reads as RSFAKE. They are GDD indomain and GDD crossgenerator/crossdomain (Z/analyse.out:147, 156, 165).
    - D "Z/thr.py, which reproduced the stored validation score (0.9941 full, 0.9981 web)": thr.py computes no validation F1. That check is in Z/nr_srcval.py / nr_srcval.out (checkpoint scores 0.994117 and 0.998064).
    - Protocol section 9 "Oracle thresholds ... (cross_detail.json)": that file holds only auc, auc_ci95, tpr_at_5pct_fpr and per_group. Oracle thresholds are in Z/diag_out.txt.
    - "About 9.9 h per run": 8:48:32 (RUNLOG.md:119 to 123) + 3058.6 s + 0:10:30 (RUNLOG.md:167 to 174) = 35400.6 s = 9.83 h (my arithmetic). It also omits the 59.2 min tile pass that section 4 requires for C4/C5.
    - C2 cost of 40216 tiles / 59.2 min is correct for the 8 primary sets but leaves out DM-AER all-reals (3197 capped tiles), the own test and validation.
    - All other figures I checked match their files (see "Checked and correct" below).

19. **Section D mechanics that will trip:**
    - `<set>.ids.json` labels are binary (own_train: 5036 / 4964). The 502/262/236 assertion and macro-F1 need the `group` field (pristine/tampered/fake) or metadata.csv.
    - The key path is `st["model"]["cls_head.weight"]` with class 0 = pristine (Z/thr.py).
    - "Assert val macro-F1 equals the stored score" needs a tolerance: single view 0.9941 against stored 0.994117471.
    - Z/sizes.json has no key for matched DM-AER or the own sets (keys: fmow, rsfake, uw, allreals_dmaer, gdd_*).
    - There are no cached features for seeds 43/44 in F/, so C1a is seed 42 only.
    - G/eval_cross_dataset.py has no option to score validation ids and no centre-256-crop degrade (`--degrade` takes jpeg:Q or scale:S only). C1b and gate 6.6 need new code.
    - Every file named in section D exists; nothing is missing.

20. **The disclosure evidence is not durable.** Protocol section 9 cites Z/ outputs that live in a session scratchpad, which a reboot clears. Fix: with owner approval, archive Z/*.out and Z/*.py into the repo before the protocol is committed.

## Notes

21. **Licences.**
    - RSFAKE non-commercial is recorded (README.md:100-101), DM-AER CC BY-NC-SA 4.0 (RUNLOG.md:132), GDD CC BY 4.0 and test-only (RUNLOG.md:213).
    - Not found in any project file: the licence of `stabilityai/sd-vae-ft-mse`, the licence of the training dataset as it bears on derived reconstructions, and the fMoW licence for re-fetching parents from SpaceNet S3 (G/prepare_rsfake_subset.py).
    - Record all three before C5 or the confirmation draw. `~/data/gdd` also contains train_*.csv and val_*.csv (CSV only); they must stay unused.

22. **This is the second post-hoc round** on the same test sets, after ZEROSHOT_EXTRAS_PROTOCOL.md of 2026-11-26. The letter's count of "configurations tried" must include extras A/B/C and every scratch variant listed in section 9.

23. **Rule 5 is already strained.** Z/thr.out holds the C1a outcome. Running D1 before sign-off adds nothing and should wait, as the plan's E11 recommends.

24. **Checkpoint status for C0.** full_s43, web_s43 and web_s44 have test_metrics.json; full_s44 does not.

## Checked and correct

- All A1-A6 AUC and accuracy figures against G/cross_*_metrics.json `overall`, and oracles against Z/diag_out.txt T1.
- "95-100% of errors are missed fakes": I recomputed 0.9575-1.0 for sat and 0.9598-1.0 for web from the per-image files.
- "66-100% of fakes below 0.1": 0.658-1.0 in Z/diag_out.txt T2.
- 88-100% shared misses (Z/agree_out.txt).
- 0.1308 / 0.0204 and 0.8042 / 0.5780 (Z/analyse.out:97-98, 104).
- 512-only AUC 0.634-0.837 and balanced accuracy 0.547-0.583 (Z/nr_tiles.out).
- Every C1 q95 figure (Z/thr.out); 0.9076 against 0.9079 (Z/nr_srcval.out).
- 3058.6 s as the sum of `wall_sec` (my addition); 2303 and 4462 (Z/tr_counts.out).
- 0.5301 / 0.5373 / 0.5765 (Z/rev_endpoint.out).
- Line references train_v2.py:3304-3313, ZEROSHOT_EXTRAS_PROTOCOL.md:31, RUNLOG.md:213, run_web_seeds.sh:14.
- Own training images are 512x512 (400 of 400 sampled), so the C3 threshold equals the C1b threshold.

VERDICT: Not approvable as written. The conclusion that no large zero-shot gain is supported stands, but the protocol has to remove own-test gating, drop the "no outside image informed any choice" claim, enumerate C4/C5's real code changes, and present C1-C3 as exploratory and already partly seen before the owner is asked to sign.
