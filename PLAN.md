# Plan: Novel Superior Hybrid Model for Autism FER (Q1 Paper)

## Goal
Replace the current 13-model sweep (`final-autism-fer.ipynb` = `run_all_models.py` + `run_proposed_model.py`) with a **clean, bug-free, honest, and novel** hybrid architecture that beats the baselines under a **statistically defensible protocol**, and produces paper-ready figures — all runnable in ~9 GPU-hours on Kaggle T4/P100.

## Hard Truths from the Current Run (must fix)
1. **Best-epoch-on-test leakage.** `run_all_models.py:583` picks best checkpoint by val F1, then computes OOF metrics on that same fold. All 13 numbers are optimistically biased.
2. **Uncontrolled comparison (5 confounds).** Baselines: 80 epochs, WeightedRandomSampler, unweighted focal γ=2.0, drop_rate=0.3/drop_path=0.2. Proposed: 160+20 epochs, no sampler Stage 1, focal γ=1.5+alpha, no drop-path.
3. **Silent regularization drift.** `get_model` (`run_all_models.py:349-359`) falls back to zero dropout on TypeError. VGG16 trained with zero dropout; ConvNeXt got 0.3+0.2.
4. **ConvNeXt collapse = config bug.** `run_all_models.py:187` gives it lr=1e-3 with CosineAnnealingWarmRestarts; warmup /eonly applied for `ce_smooth` (`:553`). Guaranteed divergence.
5. **Stage-2 double-weighting.** `run_proposed_model.py:641` comment says "sampler only, no loss weights here" but `train_one_epoch` uses module-global `loss_fn` carrying alpha → precision/recall gap.
6. **EMA cripples early training.** Deepcopy EMA starts from pretrained-backbone + random-head; at ~90 steps/epoch it's ~91% stale after epoch 1 → val acc 8.5%, 40 wasted epochs.
7. **Vacuous guardrail.** 6 classes → max-prob ≥ 0.167 always; 0.30 threshold abstains on 4/1808 images. ROC/PR labeled "Macro" but computed micro (`run_proposed_model.py:784`).
8. **Silent leakage landmine.** `fold_id_by_path.get(s, 0)` (`run_proposed_model.py:431`) defaults unmatched paths to fold 0.
9. **Dataset is a 4-source merge** incl. "augmented-autism-facial-emotion-recognition" → likely near-dup images straddling folds. Every number is optimistically biased.

## Protocol Fixes (non-negotiable for Q1)
- **Nested + grouped CV.** Cluster near-duplicate images (perceptual hash) into groups; split groups, not images (GroupKFold). Hold out one fold as test; inner CV (4 folds) for model selection. Report **test-fold** metrics only. This eliminates both best-epoch-on-test leakage AND cross-fold dup leakage.
- **Fixed 40-epoch schedule** with warmup+cosine for ALL models (halves cost, loses nothing — your own log shows plateau well before epoch 100).
- **Controlled comparison.** Same transforms, same TTA, same optimizer family (AdamW), same regularization (drop_path where supported, explicit 0 otherwise). No per-model LR accidents.
- **Trim to 6 representative baselines** (fair, fit in ~9h): vgg16, resnet50, efficientnet_b0, deit_small, swin_tiny, convnext_tiny (fixed LR). Plus the proposed hybrid.
- **PIPELINE_VERSION bump** → `v4-grouped-nested`; warn-and-refuse on old results dirs.

## Novelty (the paper's contribution)
**Fusion of (a) FER-pretrained dual-stream CNN-ViT backbone, (b) a MediaPipe landmark/blendshape geometric stream, (c) a conformal-abstention clinical guardrail.** The geometric stream is the actual novelty: autistic children's expressions are less symmetric/less synchronized — geometry captures that, pixels alone don't. It also gives clinically readable explanations instead of a Grad-CAM blob. ImageNet→FER-pretraining on the image stream is the accuracy engine (feature gap is why all 13 plateau at 0.56–0.61 regardless of size).

## Deliverables (rewrite into `final-autism-fer.ipynb`)
1. **Cell 0 (setup)** — `!pip install mediapipe` (internet on, no extra datasets needed).
2. **Cell 1: `run_all_models.py` v2** — grouped+nested CV, 6 baselines, FER-pretrained vgg16 as the "baseline with the same head start" (for the ablation), fixed LR/regularization, honest OOF-on-test reporting, `fold_id_by_path.json` + `group_id_by_path.json`.
3. **Cell 2: `run_proposed_model.py` v2** — new hybrid: FER-pretrained VGG16-BN + DeiT-S dual-stream, MediaPipe landmark/blendshape stream (with fallback for detection failure), conformal abstention, honest figures.
4. **Cell 3: analysis** — the paper table, per-class metrics, conformal coverage curves, reliability diagrams, geometric-stream ablation, all with correct macro/micro labels.

## Sequence (ordered by cost/impact)
1. Duplicate + label-noise audit (perceptual hash clustering) → group IDs. Free, CPU.
2. FER-pretrain VGG16 backbone once on FER2013/RAF-DB (Kaggle, internet on) → reuse everywhere. ~45 min once.
3. Rewrite both scripts with the fixed protocol + new hybrid + landmark stream.
4. Conformal abstention + calibration on saved OOF probs. Free, CPU.
5. Paper figures (correct macro/micro, grouped CV, ablation table).

## Risks & Mitigations
- **MediaPipe failure on ASD faces** → fallback stream (frozen mean landmarks) + honest reporting of detection rate (itself a publishable finding).
- **GPU budget** → 40-epoch schedule, resumable per-fold, 6 baselines (not 12).
- **Landmark stream added only to proposed model** → baselines get FER-pretrained vgg16 too; report ImageNet-vs-FER as an ablation so the gain is attributable to architecture, not head start.

## Files to change
- `final-autism-fer.ipynb` (rewrite cells; keep in sync with `.py` scripts per AGENTS.md)
- `kaggle/run_all_models.py` (protocol + 6 baselines + grouped/nested CV)
- `kaggle/run_proposed_model.py` (new hybrid + landmark stream + conformal guardrail)
- `kaggle/autism-fer-model.ipynb` (mirror of the notebook)
- `AGENTS.md` (update pipeline version + workflow notes)
- `PLAN.md` (this file)

## Definition of Done
- Proposed hybrid **beats the best baseline** on honest test-fold macro-F1 (not best-epoch-on-test).
- All numbers reproducible, resumable, no leakage (grouped folds, single test fold).
- Figures correct (macro/micro labeled properly, conformal coverage, reliability).
- Paper-ready ablation: ImageNet vs FER-pretrained; pixel-only vs pixel+geometry; with/without conformal abstention.