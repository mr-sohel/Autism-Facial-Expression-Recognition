# IMPROVE.md — Maximize Proposed-Model Accuracy

**Objective:** highest achievable macro-F1 on `dataset_clean/`. No comparison-table constraints —
data, protocol, and architecture are all fair game.
**Current (leaky measurement):** 0.5981 ± 0.0242. Informal target: beat vgg16's 0.6114, then keep going.

**Measure honestly first.** The current score selects the best epoch on the fold it reports, so part
of it is leak. Switch to fixed-schedule (no early stop, no best-checkpoint, evaluate final EMA
weights). The number will dip once; everything after that is a real gain.

---

## Already falsified — do NOT spend time here
| Idea | Result | Evidence |
|---|---|---|
| Logit adjustment `p / prior^τ` | **0 gain**, best τ = 0 | sweep on saved OOF probs |
| Reverse prior adjust `p · prior^s` | **0 gain** (honest nested 0.5977 vs 0.5981) | 5-fold nested τ selection |
| Temperature scaling for accuracy | Mathematically cannot change argmax | verified flat |
| Post-hoc per-class thresholds | Same family as above — already at local optimum | — |

**Why:** the model is already recall-rich / precision-starved on rare classes
(anger P 0.40 / R 0.62, fear P 0.37 / R 0.66). Post-hoc rebalancing can only trade one for the other.
The fix must be at **training time**.

---

## P0 — Correctness + honest measurement (do these first; they are free accuracy)
These are bugs that handicap the proposed model, plus the protocol change.

- [ ] **Fixed-schedule protocol** (`FIXED_SCHEDULE = True`): no early stopping, no best-checkpoint
      selection, evaluate the final EMA weights. Removes the leak, keeps all 1,446 non-test images in
      training, cheaper. Keep the old path behind the flag for one comparison run.
- [ ] **Per-(model, fold) seeded generator.** Replace the single global `DL_GENERATOR`
      (`run_proposed_model.py:206`) with `torch.Generator().manual_seed(SEED * 1000 + fold)` created
      inside each fold. Without this the headline number is not reproducible
      (proof: `efficientformer_l1` fold-0 = 0.1337 vs 0.3727 across sessions).
- [ ] **Stage-2 single weighting.** Either drop `alpha` from the loss for stage 2 (pass a plain
      `FocalLoss(alpha=None)`) or drop the `WeightedRandomSampler`. Currently both are active
      (`:641` comment vs `:545` global `loss_fn`). Pick sampler-only to match the baselines.
- [ ] **Truly freeze the backbone in stage 2.** After setting `requires_grad=False`, call
      `.eval()` on `stream_a` / `stream_b` (and keep them in eval inside `train_one_epoch`), otherwise
      backbone BN running stats keep drifting (`:634` vs `:538`).
- [ ] **EMA warm-up ramp.** Decay `0 → 0.999` over the first 5 epochs (or start EMA at epoch 3).
      Recovers the ~15–20 epochs/fold currently wasted (fold-1 val acc was 8.6% at epoch 1).
- [ ] **Reorder persistence:** `np.save(...)` → write `cv_metrics.json` → `mark_done(fold)` (`:743-749`).
- [ ] **Assert fold integrity:** every sample path must be present in `fold_id_by_path.json`
      (no `.get(s, 0)` fallback) and fold sizes must equal `[362, 362, 362, 361, 361]`.
- [ ] **Match eval precision to the baselines** — run the TTA/eval forward passes in fp32
      (baselines do; proposed uses fp16 autocast at `:526`, prob rows sum to 0.9998–1.0002).
- [ ] Housekeeping: delete the duplicate `val_ds_transform_loader` (`:708`), set
      `PATIENCE_S2 < STAGE2_EPOCHS`, write `pipeline_version.json` only on successful completion.

**Expected combined effect: +1 to +3 pts.** Cost: ~3 h GPU for the re-run. Risk: low.

---

## P1 — Class-balance surgery (the sadness/fear problem)
Evidence: `fear` alpha = 5.32× ⇒ precision 0.37. `sadness` = 404 imgs (22% of data) but recall 0.37.

### P1a — Noise-robust loss (NEW — do this even if you never relabel by hand)
The audit shows the negative-emotion labels are **internally inconsistent**, not just sparse:
154 `sadness` images are called anger (89) or fear (41) by all 12 models, while 30 `anger` and
14 `fear` images are called sadness. That is a 4-source merge disagreeing with itself, i.e. ~15%
symmetric label noise concentrated in one cluster. Standard imbalance tricks cannot fix noise.

- [ ] Swap focal for a **noise-robust objective**: Generalized CE (`q≈0.7`) or Symmetric CE
      (`α·CE + β·RCE`). Both are drop-in and typically worth +2 to +5 pts at this noise level.
- [ ] Or **soft-bootstrapping**: target = `β·onehot + (1-β)·model_prob` after a warm-up epoch count.
- [ ] Or **small-loss selection (co-teaching lite)**: per epoch, drop the highest-loss `x%` of samples
      within the negative cluster only.
- [ ] Add **class-pair soft labels** for `sadness/anger/fear` (e.g. 0.8 true + 0.1/0.1 to the confused
      pair) — cheap and directly encodes the annotator disagreement.

**This is also a publishable contribution:** "noise-robust training for multi-source clinically
collected FER data" — and unlike manual relabeling it is automatic and reproducible.

### P1b — Weighting fixes

- [ ] **Cap the alpha ratio** at ~3× (currently 14.8× spread between joy 0.36 and fear 5.32) and
      re-tune `sadness` upward relative to `fear`. The current V6 boost (`sadness ×2.0, fear ×1.2`)
      is fighting the inverse-frequency term, not helping it.
- [ ] **Compute alpha from the fold's TRAIN labels only** (`:447` currently uses all 1,808 — a prior leak
      and an inconsistency with the baselines' train-only sampler weights).
- [ ] **Try sampler-only** (drop focal alpha entirely, γ kept) — exactly matches the baseline recipe,
      which is what vgg16 wins with. This is the single most likely fix for the precision collapse.
- [ ] **Confusion-targeted loss:** add a small penalty on the `sadness ↔ natural ↔ anger` confusions
      (the dominant off-diagonal mass) — e.g. class-pair-weighted CE or a margin term for sadness.

**Expected: +1 to +3 pts** (fear F1 is precision-limited: 0.37 P / 0.66 R ⇒ balancing alone lifts it).
Cost: included in the same re-run. Risk: low-medium.

---

## P2 — Capacity / fusion improvements
- [ ] **Regularization parity.** The proposed streams get no `drop_rate`/`drop_path_rate` while the
      `deit_small` baseline got 0.3/0.2. Add `drop_path_rate≈0.1` to the DeiT stream.
- [ ] **Replace concat+SE with gated cross-attention fusion** (or a learned scalar gate per stream).
      Current fusion is `cat(SE(a), SE(b))` — SE on a 1-D pooled vector is just a learned rescaling.
- [ ] **Keep VGG16 spatial detail.** Stream A currently collapses the 7×7 map with GAP. Try
      attention-pooling or a 2×2 spatial pool (2048-d) — VGG16 alone beats the whole hybrid, so
      information is being thrown away here. **This is the highest-upside architectural change.**
- [ ] **Seed ensemble (3 seeds averaged per fold)** — reliable +0.5 to +1.5, costs 3× GPU. Legitimate
      as long as it is described as part of the proposed method.

**Expected: +0.5 to +2.5 pts.** Cost: +1–3 h GPU. Risk: medium.

---

## P3 — Novelty layers (paper contribution, not primarily accuracy)
- [ ] **MediaPipe geometry stream** — FaceMesh 478 pts → normalized distances, AU-proxy angles,
      left-right asymmetry index → MLP(128) → gated fusion. Frozen-mean fallback on detection failure;
      **report per-fold detection rate as a result**. Verified feasible: faces are large, frontal,
      single-subject, min side ≥131 px, all RGB.
- [ ] **Split-conformal abstention** at target coverage 0.85 (replaces the inert 0.30 threshold; use
      0.50 as the naive comparator — it flags 145 images / 8.0%). Produce a risk–coverage curve.
- [ ] **Reliability diagram** for the proposed model vs vgg16 — vgg16 has 7–27% of rows above
      max-prob 0.99 vs 0.0% for transformers, so this is a guaranteed favourable comparison.
- [ ] **Grad-CAM + landmark overlay** on *out-of-fold* images from the correct fold's checkpoint
      (currently fold-1 weights on possibly-training images, `:880,897`).

---

## P4 — Dataset cleaning (NOW ACTIVE — biggest single lever)
No longer gated. Expected **+2 to +6 pts**, more than any architectural change.

- [ ] Review the **274 consensus-flagged images** (0 of 12 models agree with the label) exported with
      per-model votes: `sadness` 154, `anger` 30, `joy` 27, `natural` 25, `surprise` 24, `fear` 14.
      Sadness alone is 38% of that class and its recall is 0.37 — this is where the ceiling is.
- [ ] Relabel or remove only clear errors; keep a decision log (thesis appendix + reproducibility).
- [ ] Re-run the proposed model on the cleaned set and record the delta per class.
- [ ] If a comparison table is wanted later, re-run the 12 baselines on the final cleaned data.

---

## Verification gates before trusting a number
- [ ] Reproducibility: re-run one fold twice, confirm identical macro-F1.
- [ ] Fold integrity assert passed (1808/1808 paths matched, sizes `[362,362,362,361,361]`).
- [ ] Honest protocol active (no epoch selection on the reported fold).
- [ ] Per-fold scores logged, plus a one-line note of which change produced the delta.

## Recommended order
1. **P0** (fixes + honest measurement) → re-run → this is your true starting point.
2. **P4** (label cleaning) → re-run → biggest jump.
3. **P1** (class balance) → re-run.
4. **P2** (spatial detail, fusion, seed ensemble) → final push.
5. **P3** novelty layers last — they are for the paper, not the score.

## Realistic outlook
P0 removes three handicaps the model is currently carrying (double weighting, ~20% of the schedule
lost to EMA cold start, BN drift under a "frozen" backbone). P4 attacks the actual ceiling: 15.2% of
labels that no model agrees with, concentrated in the class with the worst recall. P1 fixes a
precision collapse caused by a 5.32× fear weight. P2's spatial-detail change targets the clearest
signal in the whole study — VGG16 alone outperforms the hybrid built on top of it.
Together these are worth substantially more than the 1.3 points that separated the old run from the
best baseline.
