# Plan: Maximize Proposed-Model Accuracy on `dataset_clean/`

## Single objective (locked this session)
**Push the proposed model's macro-F1 as high as possible on `dataset_clean/`.** The 12-baseline
comparison table is set aside — it is no longer a constraint on data, protocol, or architecture.
Reference points kept only as informal targets: proposed 0.5981 ± 0.0242, best baseline vgg16 0.6114.

**Consequence accepted:** once the dataset or protocol changes, the old baseline numbers are no
longer comparable. If a comparison table is needed later, all baselines must be re-run on the final
data/protocol (~4 h GPU, resumable, folds reusable). Noted once here; not revisited.

## Protocol change: stop measuring the leak
The current number selects the best epoch by macro-F1 **on the same fold it reports**
(`run_proposed_model.py:614` → `:685`). Optimizing against that metric partly optimizes the leak.

**New default: fixed-schedule training.** No early stopping, no best-checkpoint selection; train a
fixed number of epochs with warmup+cosine and evaluate the **final EMA weights** on the held-out
fold. This removes the leak by construction, keeps all 1,446 non-test images in training (no inner
val split needed), and is cheaper. Expect the headline number to *drop* on first measurement — that
drop is the leak being removed, and every subsequent gain is real. The old behaviour stays available
behind a flag for one comparison run.

## What the evidence says to work on
| Lever | Expected | Basis |
|---|---|---|
| **Label cleaning** (274 flagged; 154 are `sadness`) | +2 to +6 | 0 of 12 models match the label on 15.2% of images; 38% of the whole sadness class |
| **P0 correctness fixes** | +1 to +3 | three handicaps the baselines never had (see below) |
| **Class-balance surgery** | +1 to +3 | fear alpha 5.32× ⇒ precision 0.37; sadness recall 0.37 |
| **Keep VGG16 spatial detail** | +0.5 to +2.5 | VGG16 *alone* (0.6114) beats the hybrid (0.5981) — GAP is discarding what makes it strong |
| **Seed ensemble (3×)** | +0.5 to +1.5 | standard, reliable, 3× GPU |
| Post-hoc probability tricks | **0** — falsified | logit adjust / reverse prior / temperature / thresholds all tested on saved OOF probs |

## Handicaps to remove (measured, not theoretical)
1. **Irreproducibility** — one global `DL_GENERATOR` shared by every DataLoader means the random
   stream depends on how many loaders were built before it. Proof: `efficientformer_l1` fold-0 F1 =
   0.1337 vs 0.3727 across sessions, same code, same version tag. Fix: per-(model, fold) seed.
2. **Stage-2 double weighting** — `WeightedRandomSampler` *and* the alpha-weighted global `loss_fn`
   (`:641` comment vs `:545`).
3. **Stage-2 backbone not frozen** — `requires_grad=False` but `model.train()` keeps backbone BN
   running stats drifting (`:634` vs `:538`).
4. **EMA cold start** — deepcopy EMA begins at a random head, ~91% stale after epoch 1 (fold-1 val
   acc 8.6%) ⇒ ~15–20 epochs/fold wasted.
5. **Alpha computed from all 1,808 labels** (`:447`) instead of the fold's train split.
6. **`mark_done()` before `np.save()`** (`:743-749`) — a crash between them loses a fold silently.
7. **`fold_id_by_path.get(s, 0)`** (`:431`) silently routes unmatched paths to fold 0.
8. Eval precision fp16 vs the baselines' fp32; `PATIENCE == STAGE2_EPOCHS` (stage-2 early stop is a
   no-op); duplicate `val_ds_transform_loader` (`:580`, `:708`); `pipeline_version.json` overwritten
   immediately after warning (`:186`), destroying the evidence.

## Dataset track (now active — no longer gated)
- 1,808 images, all RGB, median 224×224, min side ≥131 px, single-subject, mostly frontal.
- Visual sampling found label noise in ~3 of 7 inspected images (a "surprise" grimace reading as
  anger, a "fear" toddler reading as neutral-curious, a near-neutral "sadness").
- Consensus audit exports the 274 zero-agreement candidates with per-model votes for manual review;
  class split: sadness 154, anger 30, joy 27, natural 25, surprise 24, fear 14.
- Rule: log every relabel/removal decision (thesis appendix + reproducibility).

## Sequence
1. P0 + P1 code fixes, fixed-schedule protocol → re-run 5 folds (~3 h GPU). Establishes the honest
   baseline for the proposed model.
2. Label review on the 274 candidates → re-run (~3 h). Biggest single jump.
3. P2 architecture: VGG16 spatial detail, fusion, drop-path parity, seed ensemble.
4. Optional novelty layers (geometry stream, conformal abstention) — accuracy-neutral but paper-relevant.

## Definition of Done
- Honest (no-peek) macro-F1 for the proposed model, reproducible: same fold re-run twice ⇒ identical
  score.
- Fold integrity asserted: 1,808/1,808 paths matched, sizes `[362, 362, 362, 361, 361]`.
- Per-fold results logged with mean ± std, and a documented list of every change that moved the number.

## Files
- `kaggle/run_proposed_model.py` + `final-autism-fer.ipynb` Cell 3 (kept byte-identical)
- `IMPROVE.md` — the execution checklist
- Storage: archive `final_results (1)/` as the record of the old run. Never resume from local
  `results/` — its `efficientformer_l1`/`maxvit` metrics and `paper_figures` come from a different run.
