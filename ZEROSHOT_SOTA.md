# Improving zero-shot (cross-dataset / cross-generator) accuracy of a fine-tuned DINOv3 satellite forgery detector: state of the art to 2026-11-27

Scope: every number below was read on an opened source and re-checked by a second pass. No published work tests a fix on a fully fine-tuned DINOv3 with a 3-class head and mask decoder on satellite data, so every row in section 2 is a transfer from a neighbouring setting.

## 1. Where the state of the art is
- **Frozen or lightly adapted DINOv3 now leads natural-image in-the-wild benchmarks.** A frozen DINOv3 linear probe trained on GenImage SD v1.4 gets GenImage avg acc 0.964 and Chameleon 0.914 (Simplicity Prevails, ACM MM 2026, arXiv 2602.01738); ARA reaches Chameleon 95.4 balanced acc with ViT-7B (2026 preprint, arXiv 2608.15196).
- **Training data matters more than detector design.** With content-aligned real/fake pairs, DDA scores avg 90.7 over 11 benchmarks vs AlignedForensics 75.0 (NeurIPS 2025, arXiv 2505.14359), and B-Free scores avg bAcc 96.3 vs DMID 75.6 (Table 5; CVPR 2025, arXiv 2412.17671).
- **Whether full fine-tuning hurts depends on the data.**
  - On single-source data it trails low-rank adaptation: UniversalFakeDetect mAcc is 81.02 for a linear probe, 86.22 for full fine-tuning, 91.06-93.53 for LoRA and 95.19 for Effort (ICML 2025, arXiv 2411.15633).
  - For mask localisation it also trails: cross-dataset pixel F1 for ViT-L is 0.774 with LoRA r64 vs 0.681 with full fine-tuning (DINOv3-IML, 2026 preprint, arXiv 2604.16083).
- **With very diverse data and heavy degradation augmentation, full fine-tuning wins.** The NTIRE 2026 winner, an ensemble of fully fine-tuned DINOv3, reached robust ROC AUC 0.9723 (CVPRW 2026 report, arXiv 2604.11487). Community Forensics found freezing the backbone "consistently leads to worse results" (CVPR 2025, arXiv 2411.04125).
- **The best satellite cross-generator result is SimLBR.** On frozen DINOv3-Satellite it lifts RSFake-1M total acc from 73.90 to 88.88, with fake acc 48.28 -> 78.87 and real acc about 99 (CVPR 2026, arXiv 2602.20412). The test covers all ten generators, including the two it was trained on.
- **The best satellite cross-dataset result still loses 10-20 points.** Average balanced acc over the other datasets is 0.86 for GDD-trained SimLBR and 0.77 for RSFAKE-1M-trained SimLBR, against 0.94-1.00 in-dataset (GDD, preprint 23 Nov 2026, arXiv 2611.31487).
- **Part of the fixed-threshold drop is a score shift.** One scalar logit bias lifts Effort on GenImage from 91.41 to 96.64 (labelled target) or 96.33 (unlabelled target) (AAAI 2026, arXiv 2602.01973). This uses target data, so it is not zero-shot.
- **Training-free detectors are not competitive and mostly report AUC only.** WaRPAD scores AUROC 0.422 on DALL-E 3 (NeurIPS 2025, arXiv 2511.14030), and AEROBLADE scores AUC 0.389 on OpenFake (as measured in arXiv 2603.21619).

## 2. What to try on this detector (ranked by expected benefit for this setting)

| # | Technique | Source | What changes | Evidence (quoted) | Cost | Still zero-shot? |
|---|---|---|---|---|---|---|
| 1 | Second binary head with latent blending (SimLBR) on the original frozen DINOv3, run beside the current model; try both web (LVD) and satellite weights | SimLBR, CVPR 2026, 2602.20412; GDD 2611.31487 | Head-only training on the same single dataset; current detector untouched; fusing the two is untested | RSFake-1M total acc 73.90 -> 88.88; fake acc 48.28 -> 78.87. Cost on unseen real sources: GDD cross-domain acc 0.79 (LVD), 0.64 (SAT) | ~3 min on one H100 after 22-50 min of embedding precompute | yes |
| 2 | Rebuild the fake and tampered classes from your own real tiles: VAE reconstructions (several decoders) plus inpainted versions with masks, saved in the same format and size as the reals; then retrain | B-Free 2412.17671; DDA 2505.14359; Aligned Datasets (ICLR 2025) 2410.11835; DRCT (ICML 2024) | Retrain with new data; architecture unchanged | B-Free Table 2 avg bAcc: text-paired 61.9, reconstructed 80.7, with inpainting 96.4; NLL 1.91 -> 0.10. DDA recipe re-run on DINOv3: 89.16 -> 93.20 (PRL preprint) | One VAE pass or inpainting generation per real tile, plus a retrain | yes |
| 3 | Replace full fine-tuning with LoRA on QKV (r=32-64), optionally with ARA's anchor loss to a frozen-feature classifier; keep the decoder | DINOv3-IML 2604.16083; ARA 2608.15196 (both preprints) | Retrain with a switch | Cross-dataset pixel F1, ViT-L: LoRA 0.770 / 0.774 vs full 0.681. ARA ViT-L, five in-the-wild avg: linear probe 78.6, naive LoRA 88.3, ARA 91.4 | One retrain; 1.4M-12.2M trainable parameters vs ~309M | yes |
| 4 | Audit, then equalise, file format, JPEG quality, size and resampling history across the three classes and across target sets | Fake or JPEG? (preprint) 2403.17608; BIAS-ID (NeurIPS 2026 E&D) 2605.31153 | Audit is inference-only; the fix is a retrain | GenImage cross-generator acc: ResNet50 71.68 -> 82.74; Swin-T 74.09 -> 85.83. BIAS-ID: "none of them is completely unbiased" | Hours of CPU for the audit | yes |
| 5 | If full fine-tuning is kept: backbone lr around 1e-6 with EMA; the same JPEG and degradation augmentation on all classes, blur at low probability; more augmented views | B-Free; CNNDetection (CVPR 2020) 1912.11035; design-choices study 2511.21507 (preprint) | Retrain with a switch | B-Free end-to-end at lr 1e-6: bAcc 95.2 vs linear probe 68.5. ForenSynths mAP: no aug 90.1, JPEG only 93.0, blur only 84.4. Views 1 -> 4 -> 16: Next Period AUROC 81.73 -> 90.11 -> 93.74 (all-backbone avg, multi-generator data) | One retrain | yes |
| 6 | Train on native-resolution crops instead of resized images; centre crop or averaged tiles at test, matching training | SAFE (KDD 2025) 2408.06741; MAD '26 recipe study 2507.10236; B-Free; DDA | Retrain plus an inference change | Centre crop -> texture crop AUC: DMID 78.78 -> 89.46; RINE 91.26 -> 94.90. SAFE's resize-vs-crop numbers are in a figure only | One retrain | yes |
| 7 | Retrain only the last layer with non-negative weights on the existing backbone (Stay-Positive) | ICML 2025, 2502.07778 | Head-only; defined for a binary logit, so it needs adapting to 3 classes | AP at 50% inpainted pixels: Corvi 6.13 -> 97.48; Rajan 49.69 -> 99.66. UFD avg AP 86.41 -> 92.88 | Minutes | yes |
| 8 | Add remote-sensing generator and real-source diversity (GDD, RSFAKE-1M) | GDD 2611.31487; RSFAKE-1M 2505.23283 (preprints) | Retrain with new data; no longer single-source | Cross-dataset avg bAcc, RSFAKE-1M-trained -> GDD-trained: frozen DINOv3 probe 0.75 -> 0.81; SimLBR 0.77 -> 0.86. GDD has "approximately 2%" of RSFAKE-1M's image count | Gated dataset downloads plus a retrain | yes for sets not trained on |
| 9 | Pick the checkpoint on a third, out-of-distribution validation set that is never reported | FOX, WACV 2027, 2611.04111 | Training-run selection only | Chameleon AUC 83.6 -> 97.2, but the checkpoint was selected on Chameleon itself, so it is not a held-out number | None beyond scoring checkpoints | yes for other sets; no for the selection set (uses its labels) |
| 10 | Score the 3-class output as real vs any-manipulated | Remote Sensing 2026, 18, 2041 | Inference-only | Transfer BA: exact-family 0.777 -> collapsed 0.800; a dedicated binary detector got 0.828. These averages include adversarial-attack classes, not only real vs fake | None | yes |
| 11 | Logit-bias calibration on a separate target sample, never the test split | AAAI 2026, 2602.01973 | Inference-only | GenImage: Effort 91.41 -> 96.64 / 96.33; RINE 81.78 -> 97.94 / 97.59. Small where classes overlap: Chameleon Effort 63.15 -> 67.38 / 67.99. Balanced sets only | ~100 target images | no - uses unlabeled target (or target labels in the supervised variant) |
| 12 | Pseudo-label test-time adaptation (TTC) | preprint, 2608.00559, no code | Weight updates on the target pool | GenImage bAcc 80.1 -> 92.5; Chameleon 62.4 -> 91.1. Drops to 79.3 at 80% pseudo-label accuracy | Three adaptation rounds, multi-scale inference | no - uses unlabeled target |
| 13 | Few labelled target images | FSD (ICML 2025) 2501.08763; Remote Sensing 18, 2041 | Prototype or fine-tune | GenImage: zero-shot 77.1 -> 10-shot 84.1. Aerial transfer BA 0.615 -> 0.949 with 50 images per class (Swin-B) | Tens of labels | no - uses target labels |

Only rows 4 (audit part) and 10 work on the existing weights with no training and no target data, and neither has strong evidence. Rows 11-13 must be reported as target-calibrated or few-shot, not zero-shot.

## 3. What does not help or is unproven
- **More data from one generator:** fine-tuning DINOv3-L/16 on 360k real + 360k LDM images gave 74.49-75.09% vs 75.03% frozen on So-Fake-OOD (FGTS, preprint, 2511.22471).
- **Naive LoRA on single-source, misaligned data:** DINOv3 Chameleon fell 0.914 -> 0.803 -> 0.718 (Simplicity Prevails), and ReMoT's avg fell 91.2 -> 70.5 (preprint, 2611.34719).
- **Satellite-pretrained backbone as a cure:** on GDD the frozen probe scored 0.85 vs 0.85 on unseen generators and 0.77 (SAT) vs 0.82 (LVD) on unseen real sources; the authors favour LVD.
- **Frozen-probe heads as a replacement for the tampered class or mask:** DINOv3 scored 0.592 / 0.613 / 0.450 on BR-Gen localized edits (Simplicity Prevails).
- **Artifact and frequency CNNs:** zero-shot on RSFAKE-1M, SAFE scores ACC 55.10, AIDE (SD v1.4) 58.85, UniversalFakeDetect 61.16 and Geo-DefakeHop 52.80.
- **Training-free scores as a main detector:** see section 1; a perturbation-sensitivity score with DINOv3 ViT-S/16 gave AUC 0.5019 on OpenFake (2603.21619).
- **Plain entropy-minimisation TTA:** TENT and SAR both collapse to 50.0 on AIGCDetect (TTC paper).
- **Generic cutmix/mixup and heavy blur:** cutmix/mixup lowered B-Free from 81.4 to 78.6; blur-only augmentation scored 84.4 vs 90.1 with none. Naive per-patch labels on patch-mixed views also hurt (PRL, preprint, 2610.11876).
- **Contrast enhancement at inference:** false positives on OrbitalNet reals went 87 -> 63 for SimLBR but 40 -> 44 for the DINOv3 probe; the effect on fakes is not reported (GDD Table 11).
- **Dropping the 3-class head:** no evidence either way. The AI-GenBench "multiclass" result is about per-generator classes, not real/tampered/synthetic.
- **Wholesale adoption of newer preprint detectors:** ReMoT (two days old), REM, MIRROR, Semantic Prototype Calibration (2608.04935, abstract: 84.8 -> 92.9 for PE) and IAPL (CVPR 2026, 2508.01603, abstract: GenImage 96.7%) are binary natural-image detectors. The last four were added by the verifier from abstracts only.

## 4. Caveats
- **Natural images only:** rows 2-7, 9 and 11-13 have no satellite evidence. No paper applies reconstruction alignment, LoRA-vs-full comparison, calibration or TTA to overhead imagery.
- **Satellite evidence is thin:** it comes from SimLBR (one ablation column, no other baselines) and GDD, which is four days old, shares three authors with SimLBR, and has a split table summing to 19,410 against 18,410 in the text. RSFAKE-1M is a preprint.
- **Row 10's source** is mainly an adversarial-attack paper.
- **The mask branch has its own, larger drop:** TruFor trained on FLDCF scores pixel F1 .959 there and .350 on GDD (GDD Table 4). Only row 3 addresses it, and DINOv3-IML selected its checkpoints on test-set F1.
- **Backbone size:** many headline DINOv3 numbers use ViT-7B. At ViT-L the frozen probe is much weaker (ARA Table 11: 78.6 vs 93.4), and FGTS training-free at ViT-L/16 gets 61.83 acc on GenImage.
- **Errata and inconsistencies:**
  - SimLBR's GenImage 94.54 is corrected to 91.1 by the authors (ADM loading bug).
  - The 98.83% headline of arXiv 2601.20461 did not reproduce from its own tables (about 98.75).
  - The NPR crop comparison in the MAD '26 study is unresolved, so it is not used here.
- **Preprints with self-reported numbers:** ARA, ReMoT, DINOv3-IML, FGTS, PRL, TTC, GDD, RSFAKE-1M, Fake or JPEG?, and the design-choices study.
- **No code:** ARA, TTC, Simplicity Prevails, and the NTIRE entries. FOX's repo says "Code coming soon". PRL has a checkpoint but no training code. SimLBR has no satellite head released.
- **Licences:** B-Free is "only for informational and nonprofit purposes"; Effort is CC BY-NC 4.0; RSFAKE-1M is CC BY-NC 4.0 and gated; GDD is CC-BY-4.0 and gated; DINOv3 weights are under Meta's own licence.

## 5. References
- SimLBR: https://arxiv.org/abs/2602.20412 ; code https://github.com/mvrl/SimLBR ; heads https://huggingface.co/MVRL/SimLBR
- GDD: https://arxiv.org/abs/2611.31487 ; data https://huggingface.co/datasets/geodf/gdd
- RSFAKE-1M: https://arxiv.org/abs/2505.23283
- Remote Sensing 2026, 18, 2041: https://doi.org/10.3390/rs18122041
- Simplicity Prevails: https://arxiv.org/abs/2602.01738
- FGTS: https://arxiv.org/abs/2511.22471 ; https://github.com/hzlsaber/FGTS
- ARA: https://arxiv.org/abs/2608.15196
- ReMoT: https://arxiv.org/abs/2611.34719 ; https://github.com/yunncheng/ReMoT
- FOX: https://arxiv.org/abs/2611.04111
- Effort: https://arxiv.org/abs/2411.15633 ; OmniAID: https://arxiv.org/abs/2511.08423
- DINOv3-IML: https://arxiv.org/abs/2604.16083 ; https://github.com/Irennnne/DINOv3-IML
- B-Free: https://arxiv.org/abs/2412.17671 ; https://github.com/grip-unina/B-Free
- DDA: https://arxiv.org/abs/2505.14359 ; https://github.com/roy-ch/Dual-Data-Alignment
- Aligned Datasets: https://arxiv.org/abs/2410.11835 ; Stay-Positive: https://arxiv.org/abs/2502.07778 ; shared repo https://github.com/AniSundar18/AlignedForensics
- DRCT: https://proceedings.mlr.press/v235/chen24ay.html
- PRL: https://arxiv.org/abs/2610.11876
- Final-component reconstructions: https://arxiv.org/abs/2601.20461
- Fake or JPEG?: https://arxiv.org/abs/2403.17608
- BIAS-ID: https://arxiv.org/abs/2605.31153 ; https://github.com/jonasricker/biasid
- CNNDetection: https://arxiv.org/abs/1912.11035
- SAFE: https://arxiv.org/abs/2408.06741
- MAD '26 recipe study: https://arxiv.org/abs/2507.10236
- AI-GenBench: https://arxiv.org/abs/2504.20865 ; design choices: https://arxiv.org/abs/2511.21507
- Community Forensics: https://arxiv.org/abs/2411.04125
- NTIRE 2026 report: https://arxiv.org/abs/2604.11487
- AIGIBench: https://arxiv.org/abs/2505.12335
- Logit calibration: https://arxiv.org/abs/2602.01973 ; https://github.com/muliyangm/AIGI-Det-Calib
- TTC: https://arxiv.org/abs/2608.00559
- DEDCA (NeurIPS 2026 poster): https://arxiv.org/abs/2611.16307
- FSD: https://arxiv.org/abs/2501.08763
- WaRPAD: https://arxiv.org/abs/2511.14030
- Frequency-perturbation zero-shot: https://arxiv.org/abs/2603.21619
- AEROBLADE: https://arxiv.org/abs/2401.17879
- TRAIL (training-free edit localisation on frozen DINOv3, preprint): https://arxiv.org/abs/2608.18968
- Added by verifier, abstracts only: IAPL https://arxiv.org/abs/2508.01603 ; Semantic Prototype Calibration https://arxiv.org/abs/2608.04935 ; REM https://arxiv.org/abs/2512.20937 ; MIRROR https://arxiv.org/abs/2602.02222 ; GAPL https://arxiv.org/abs/2512.12982 ; RealStats https://arxiv.org/abs/2601.18900

---

# Completeness check (second pass, 2026-11-27)

COMPLETENESS REVIEW (searched and opened 2026-11-27; every number below was read on the opened page; where only the abstract page was read, that is stated)

## Additions / corrections

### (c) Critiques and negative results that affect the ranking

1. **Inpainting-exchange test (INP-X): detectors key on global VAE artifacts, not on the edited region** - https://arxiv.org/html/2602.00192v1 (preprint, 30 Jan 2026; full HTML read, first 100k chars)
   - Restoring the original pixels outside the mask drops accuracy: Hive 0.914 -> 0.548, Sightengine 0.926 -> 0.550, Corvi2023 0.942 -> 0.554, DNF 0.710 -> 0.604, SPAI 0.661 -> 0.542.
   - Training on exchanged images helps only modestly: ResNet-50 mIoU on the INP-X test set 0.463 (trained on standard inpainting) vs 0.472 (trained on INP-X); best INP-X accuracy 0.753 (CLIP ViT-B/32).
   - Why it matters: this bears directly on row 2 and on the tampered class and mask branch. If the inpainted tiles keep the full VAE round-trip outside the mask, the "tampered" class and the mask can be learned from a global shortcut. Row 2 should say to paste original pixels back outside the mask, and to test with an exchange-style set. It also qualifies row 7's Stay-Positive inpainting AP numbers.

2. **Real-class bias hidden by high accuracy; fix is head-only pruning (DEAR, "Dissect and Prune")** - https://arxiv.org/html/2606.10309 (ICML 2026 per the arXiv page; HTML read, first 100k chars)
   - Corvi on FLUX, original images: real acc 99.9%, fake acc 21.5%.
   - Post-processed average: Corvi real 97.3 / fake 47.7 / AUC 79.2 -> DEAR-c real 76.4 / fake 90.2 / AUC 92.3. Rajan AUC 92.9, fake 58.2 -> DEAR-r AUC 97.1, fake 89.5.
   - Backbone frozen; final features pruned using about 6,400 SD-1.5 inpainted diagnostic images; only the linear classifier is retrained.
   - Why it matters: it is the same symptom as the SimLBR baseline in the answer (fake acc 48.28 with real acc about 99), and a second head-only, zero-shot option beside rows 1 and 7. Caveats: ResNet-50 only, no DINOv3 or ViT evidence, and real accuracy falls by about 21 points in the post-processed average. Belongs near row 7, not above it.

3. **Pretraining-exposure critique of foundation-model probes; DINOv3 is the exception** - https://arxiv.org/html/2509.12995v3 (preprint, v3 15 Oct 2025; HTML read)
   - Chameleon avg, linear probes trained on GenImage: PE 96.1, DINOv3 92.4, MetaCLIP-2 91.8, SigLIP2 83.8, DINOv2 60.8, CLIP 55.6.
   - On WebAIG-25 (collected after the pretraining cut-offs): MetaCLIP-2 51.4 (real 27.4, fake 75.4), PE 68.1 (real 44.4, fake 91.7), SigLIP2 62.7, DINOv3 94.5 (real 97.5, fake 91.4).
   - The authors "speculate that this is due to data exposure".
   - Why it matters: section 1's "frozen DINOv3 leads" claim is missing this caveat for the class of models, and it is evidence for staying on DINOv3 and not moving to PE or MetaCLIP-2 (relevant to the Semantic Prototype Calibration entry in section 3, which is PE-based). It also weakens any expectation that web-pretraining exposure helps on satellite fakes. The Chameleon-vs-WebAIG comparison is across two tables, so treat the deltas as approximate.

4. **Semantic-shortcut critique of fine-tuned detectors (GSD, "semantic fallback")** - https://arxiv.org/abs/2603.09242 (preprint, v2 3 Jun 2026; abstract page only, no accuracy numbers on it)
   - Claims forensic fine-tuning leaves features organised around semantics; suppresses the dominant semantic subspace of a frozen CLIP encoder via SVD.
   - Why it matters: same family as Effort in section 1; mention only, no number to cite.

5. **Format-bias row (row 4) is adequately covered.** I found no newer benchmark-bias paper beyond Fake or JPEG? and BIAS-ID. One extra pointer surfaced in search snippets only (not opened, so no numbers): a June 2026 audit of training-free detectors under preprocessing and compression, https://arxiv.org/pdf/2606.20488 - consistent with the answer's "training-free is not competitive" line.

### (a) 2025-2026 methods

6. **HEDGE, NTIRE 2026 4th place of 193 teams** - https://arxiv.org/abs/2604.03555 (preprint; abstract page only)
   - Ensemble of DINOv3 routes built by "staged data expansion and augmentation escalation", a higher-resolution branch, and a MetaCLIP2 branch, fused in logit space.
   - Why it matters: a second, independent challenge entry supporting the answer's "diverse data plus heavy augmentation plus fine-tuned DINOv3" point (rows 5 and 6). No metric value on the page, so nothing new to quote.

7. **HiDA-Net, native-resolution tiles plus a downsampled global view** - https://arxiv.org/abs/2508.17346 (abstract page lists no venue; a search snippet said ICLR 2026, which I did not confirm)
   - Abstract: "increasing accuracy by over 13%" on Chameleon and 10% on HiRes-50K (50,568 images, up to 64 megapixels); includes token-wise forgery localisation and JPEG quality-factor estimation modules.
   - Why it matters: additional support for row 6 (no resize), and the most relevant architecture for large satellite tiles. Abstract-level numbers only.

8. **Inpainting-localisation work missing from the mask-branch discussion**
   - DinoLizer - https://arxiv.org/abs/2511.20722 (preprint, v2 30 Jul 2026; abstract page only): DINOv2 with LoRA on query and value plus one linear layer at 14 x 14 patch resolution, sliding window for large images; "a 20% higher Intersection over Union score" than the second-best model on average; "very robust to JPEG (double) compression"; code at https://github.com/anonyme610/dinolizer. Supports row 3 (LoRA beats full fine-tuning for masks) from an independent group.
   - DiffusionPrint - https://arxiv.org/abs/2604.12443 (CVPRW 2026; abstract page only): contrastive forensic feature map added to TruFor and MMFusion; "gains of up to +28% on mask types unseen during fine-tuning"; code at https://github.com/mever-team/diffusionprint. Relevant because the answer's caveat cites TruFor's .959 -> .350 drop and offers only row 3 for it.
   - "Detecting Localized Deepfakes" - https://arxiv.org/abs/2512.16688 (preprint, 18 Dec 2025; abstract page only): detectors trained on fully synthetic images "can reliably detect medium- and large-area manipulations or regeneration-style inpainting". A search snippet (not read by me on the paper) gave DINOv3 ViT-L/16 AUROC 0.86 on BR-Gen and weak results under 20% mask area; do not quote that number without opening the PDF. Complements the answer's BR-Gen 0.592 / 0.613 / 0.450 line.

### (b) Remote-sensing-specific work

9. **fMoW-based manipulation and localisation prototype** - https://arxiv.org/abs/2608.04840 (accepted at IEEE IGARSS 2026; abstract page only)
   - 60 images, 30 manipulated and 30 authentic, each with a ground-truth mask and acquisition metadata; copy-paste splicing and diffusion inpainting among three manipulation types; hosted at huggingface.co/datasets/geodf/fmow-fake-small; no detector baselines on the page.
   - Why it matters: a small extra held-out satellite set with masks for evaluating the tampered class. It sits under the same Hugging Face organisation as GDD, so it is not independent evidence, and it is too small to train on.

10. **"Deepfake Geography"** - https://arxiv.org/html/2511.17766v1 (preprint, single author; HTML read)
    - ViT-B/16 95.11% vs ResNet-50 87.02% test accuracy on pooled DM-AER + FSI, frozen backbones; no train-on-one, test-on-other experiment; fakes "primarily" StyleGAN2 and CycleGAN; image count inconsistent within the paper (over 130,000 / 138,064 / approximately 134,000).
    - Why it matters: list it only as a non-result. It adds no cross-generator evidence, and the answer loses nothing by omitting it.

11. **No further satellite cross-generator or cross-dataset method was found** beyond SimLBR, GDD, RSFAKE-1M, FLDCF and Remote Sensing 18, 2041. The answer's "satellite evidence is thin" caveat stands.

## Not verified by me
- I did not re-open the answer's own sources, so I make no statement on its quoted numbers.
- Items marked "abstract page only" carry no table-level numbers.
- Whether the answer's row 2 recipe (B-Free) already restores original pixels outside the mask was not checked; if it does, item 1 becomes a test-set recommendation and not a recipe change.

## Verdict
The ranking order should not change; amend row 2 to require original pixels outside the inpainting mask plus an exchange-style check (item 1), add DEAR as a head-only option beside row 7 (item 2), and add the pretraining-exposure caveat with DINOv3 as the exception to section 1 (item 3).
