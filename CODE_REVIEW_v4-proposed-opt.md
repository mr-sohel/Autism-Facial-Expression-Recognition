# Code Review: `kaggle/run_proposed_model.py` (v4-proposed-opt)

**Date:** 2026-08-26
**Scope:** `kaggle/run_proposed_model.py` (1,195 lines) audited against `PLAN.md` and the `IMPROVE.md`
P0/P1/P2 checklist.
**Verdict up front:** faithful, clean implementation of the planned v4 protocol — **fix Gap A and Gap B
below before the authoritative Kaggle run**, since the entire point of v4 is trustworthy numbers.

---

## Verification gates actually run

| Gate | Result |
|---|---|
| `python -m py_compile kaggle/run_proposed_model.py` | OK |
| Notebook cell 3 (`final-autism-fer.ipynb`) byte-identical to `.py` | OK |
| Fold-contract path matches `run_all_models.py:516` (`/kaggle/working/results/fold_id_by_path.json`) | OK |
| Legacy remnants (`DL_GENERATOR`, `CosineAnnealingWarmRestarts`, `.get(s, 0)` fallback) | None found |

## Verified correct — 9/9 IMPROVE.md P0 items

| Item | Status | Evidence (line refs) |
|---|---|---|
| Fixed schedule; final-EMA eval; no epoch selection on reported fold | ✅ | `FIXED_SCHEDULE=True` :101; val metrics monitoring-only :829–832, :888–890; final ckpt saved once per stage :844–847, :902–904 |
| Per-fold seeded generators (reproducibility) | ✅ | `torch.Generator().manual_seed(SEED*1000+fold)` :256; stage-2 loader uses `fold+100` :874; `seed_everything(SEED+fold)` :794 |
| Stage-2 single weighting (no sampler+alpha double-dip) | ✅ | sampler-only; `s2_alpha=None` :806–807; sampler built only when selected :868–874 |
| Backbone truly frozen in stage 2 (BN stats stop drifting) | ✅ | `model.train()` then `set_backbone_eval()` :739–742 |
| EMA warm-up ramp 0→0.999 (no cold-start waste) | ✅ | `_current_decay()` :531–535; warmup = 5 × steps/epoch :700–701; stage-2 EMA seeded from loaded weights (no cold start) :861 |
| Persistence order: data → metrics → mark_done | ✅ | npy files → `oof_paths.json` → `cv_metrics.json` → `mark_done(fold)` :973–980 |
| Fold integrity asserted; no silent fold-0 routing | ✅ | hard `RuntimeError` on unmatched paths :601–608; sizes checked sum + nonzero :626 (see Gap B for exactness) |
| fp32 eval/TTA (matches baselines) | ✅ | no autocast in `evaluate_tta` / `evaluate_model`; `softmax(logits.float())` |
| Housekeeping | ✅ | single `val_ds_transform_loader` :783; `PATIENCE_S2=8 < STAGE2_EPOCHS=20`; `pipeline_version.json` written only at end :1184 and version-mixed resume is **refused**, not warned :218–227 |

Additional cross-checks that passed:

- Grad-CAM uses fold-1 weights on fold-1 **out-of-fold** images (:1130, :1149–1155) — the v3
  train-image leak is gone.
- `UNCERTAINTY_THRESH=0.50` guardrail plus full risk–coverage sweep persisted to
  `risk_coverage.json` (:1024–1049).
- Class weights computed from the fold's train labels only, tempered `(1/freq)^0.5`, sadness boost,
  ratio cap (:552–585, :799–803). Full-set alpha at :632 is reference-print only, never used for training.
- Dataset resolution priority matches docs: auto-detect > Kaggle slug > local > MTCNN-warning fallback
  (:124–155); restore-from-prior-session runs before the version gate so a stale marker fails loudly.

---

## Genuine gaps

### Gap A — Resume-crash can silently duplicate OOF rows (low probability, cheap to close)

A crash between `np.save(...)` (:973) and `mark_done(fold)` (:980) leaves the fold's rows on disk but
the fold unmarked. On resume the loop re-runs that fold and **appends duplicate rows** to
`oof_preds/oof_labels/oof_probs.npy` and `oof_paths.json`. Nothing detects this: aggregate only checks
`all_preds.size == 0` (:985); there is no check of row count vs dataset size or path uniqueness.
(The reorder fixed the legacy silent-fold-loss failure mode but introduced this detectable-but-unchecked one.)

**Fix design (startup reconciliation):**
1. After loading OOF arrays + `oof_paths.json` + done-set, drop every trailing row whose image belongs
   to a fold not in `done["proposed_model"]`. Row→fold mapping is available via `oof_paths[i]` +
   `fold_ids` (path-keyed, order-independent).
2. Keep only a prefix (rows are appended per fold), then assert consistency:
   `len(all_preds) == len(all_labels) == len(all_probs) == len(oof_path_index)`.
3. After the fold loop completes, assert `len(all_preds) == len(samples)` (1,808 on the canonical set)
   and `len(set(oof_path_index)) == len(oof_path_index)` before aggregating.

### Gap B — PLAN.md DoD exact fold sizes are not asserted

`PLAN.md` Definition-of-Done requires fold sizes `[362, 362, 362, 361, 361]` asserted with 1,808/1,808
paths matched. The code checks only `sum(_fold_sizes) == len(samples)` and `min > 0` (:626).

**Fix design:** when `len(samples) == 1808`, assert `_fold_sizes == [362, 362, 362, 361, 361]`;
otherwise warn with the observed sizes (keeps the script usable if the dataset changes later).

---

## Nits (optional, non-blocking)

1. **Provenance gaps in `pipeline_version.json` config** — records key switches but omits
   `ALPHA_POWER` (materially changes class weights), `GCE_Q` / SCE params, `DEIT_DROP_PATH`,
   `LEARNING_RATE`, `BATCH_SIZE`, `WEIGHT_DECAY`, `SEED`, `N_FOLDS`. Add them so any results dir can be
   fully reconstructed from its own metadata.
2. **No intra-fold checkpoint under FIXED_SCHEDULE** — nothing is written until a stage completes, so
   a T4 session death mid-fold forfeits ~40 min of that fold's GPU time. Optional: periodic raw-state
   save every N epochs (kept out of the selection path so it cannot reintroduce leakage).
3. Style: missing blank line before `device = torch.device(...)` (:262).
4. `from torch.cuda.amp import GradScaler` is deprecated on torch ≥ 2.4 (works on Kaggle's 2.10;
   FutureWarning is suppressed by the blanket filter). Portable form: `torch.amp.GradScaler("cuda", ...)`.

---

## Deferred by plan (not defects)

- **Gated cross-attention fusion** replacing concat+SE, and **seed ensemble (3×)** — PLAN.md sequence
  step 3, intentionally postponed until the honest fixed-schedule baseline exists.
- **GCE/SCE noise-robust losses** implemented as switches (:105, :467–511) but default stays
  `"focal"` — per IMPROVE.md ordering (establish honest baseline first, then attack label noise).

---

## Recommended next actions (in order)

1. Implement Gap A reconciliation + Gap B size assertion (~20 lines total), regenerate notebook cell 3
   so it stays byte-identical to the `.py` (AGENTS.md requirement), re-run py_compile.
2. Wipe `/kaggle/working/results/proposed_model_proposed` (and local `results/proposed_model_proposed`)
   and launch the authoritative 5-fold v4 run on Kaggle.
3. Expect the headline macro-F1 to read lower than 0.5981 — that drop is the leak being removed, not a
   regression (PLAN.md).
