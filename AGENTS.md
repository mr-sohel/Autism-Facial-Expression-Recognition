# AGENTS.md

## What This Is
Autism Facial Expression Recognition — 6-class classification (anger, fear, joy, natural, sadness, surprise) on `dataset_clean/` (1,808 images). All training code is self-contained scripts under `kaggle/`, run on Kaggle GPUs (T4/P100).

**Current objective (see `PLAN.md`):** maximize the proposed model's macro-F1. The 12-baseline comparison sweep is **DONE and frozen** — do not retrain baselines unless the dataset changes. Strategy and evidence live in `PLAN.md`; the execution checklist is `IMPROVE.md`.

## Headline numbers (v3 run, for reference only)
- Proposed-Model: macro-F1 **0.5981 ± 0.0242** (pooled OOF 0.5971) — measured with a leaky protocol (best epoch selected on the reported fold).
- Best baseline: vgg16 **0.6114 ± 0.0173**; proposed lost on all 5 folds.
- **All pre-v4 numbers are optimistic** (epoch-selection leakage). The first v4 fixed-schedule number will read lower — that is the honest baseline, not a regression.

## Workflow (Kaggle)
The notebook actually run on Kaggle is **`final-autism-fer.ipynb`** (root; 4 cells):
1. Markdown overview.
2. Restore-progress cell — copies a prior session's output (added as an Input dataset) into `/kaggle/working/results`. **Keep this cell in any rewrite.**
3. `run_all_models.py` copy — 12 baselines (FROZEN; only re-run if the dataset changes).
4. `run_proposed_model.py` copy — the proposed model.

The two code cells are regenerated copies of the `.py` scripts — after editing a script, regenerate its cell so they stay **byte-identical** (verify with a JSON round-trip diff). Cell 3 auto-detects the dataset under `/kaggle/input` and loads `fold_id_by_path.json` written by Cell 2. `kaggle/autism-fer-model.ipynb` is an older mirror — treat `final-autism-fer.ipynb` as canonical.

## Commands (local)
- `python kaggle/run_proposed_model.py` — falls back to local `dataset_clean/`, outputs `results/proposed_model_proposed/`. CPU-only locally (no XPU path); a 5-fold run is impractical locally — use Kaggle.
- `python kaggle/run_all_models.py` — Kaggle-only (no local dataset fallback; `StratifiedKFold` raises on empty).
- `python -m py_compile kaggle/run_proposed_model.py` — quick syntax gate after edits.
- `kaggle/preprocess_faces.py` / `kaggle/offline_augmentation.py` — **legacy, NOT in the pipeline** (MTCNN+CLAHE hurt accuracy; augmented set double-counted).

## Pipeline versions (they differ per script now)
- `run_all_models.py` = **`v3-emabn`** (unchanged, frozen). Known defects, documented not fixed: best-epoch-on-reported-fold selection (`:583`), `TypeError` fallback silently drops ALL regularization (`:353`), ConvNeXt collapse (focal + lr=1e-3 + CosineWarmRestarts, no warmup; pooled F1 0.1497), "Macro" ROC/PR figures computed micro (`:784,798`), `pipeline_version` marker written only at end-of-model (interrupted models falsely report `'v1'`).
- `run_proposed_model.py` = **`v4-proposed-opt`**. **Refuses** to resume a results dir with a different version (old code only warned, then overwrote the marker). The marker + config are written only after a successful run.
- Before an authoritative v4 run: wipe `/kaggle/working/results/proposed_model_proposed` (and local `results/proposed_model_proposed` if present).

## Results trees (important)
- **`final_results (1)/`** — downloaded Kaggle output of the completed v3 run. Authoritative record: OOF `.npy` for all 13 models (row-aligned, labels byte-identical), proposed-model checkpoints, `__huggingface_repos__.json` (pins exact timm weight commits — keep it). Archive; never resume from it into a v4 dir.
- **Local `results/`** — NOT a resume source. It has the 44 baseline `.pth` files but **no proposed model**, and its `efficientformer_l1`/`maxvit` metrics + `paper_figures` come from a *different* run than the downloaded tree.
- `final-autism-fer.log` — the v3 Kaggle session log.

## Dataset
- **`dataset_clean/`** — canonical 1,808 images in `train/valid/test/<class>/` layout (merged at runtime). Counts: anger 167, fear 68, joy 843, natural 201, sadness 404, surprise 125. dHash dedup was applied when it was built (no `cleaning_report.json` exists locally). All RGB, median 224×224, min side ≥131 px, single-subject, mostly frontal — MediaPipe-friendly.
- **Label noise is the measured ceiling:** a 12-model consensus audit (`label_audit_candidates.csv`, 274 rows) found **274 images (15.2%) where 0 of 12 models match the label — 154 are `sadness` (38% of that class)**. The negative-emotion labels are internally inconsistent (154 `sadness` read as anger/fear; 30 `anger` + 14 `fear` read as sadness). Manual relabeling OR noise-robust losses both target this.
- `dataset/` — original 1,988-image set (leaky). NOT used.
- Splits: StratifiedKFold(5, shuffle, seed 42) over the merged set; fold sizes `[362,362,362,361,361]`; `fold_id_by_path.json` is the shared fold contract. v4 asserts 100% path coverage (the old `.get(s, 0)` silent default is gone).

## Proposed-Model config (v4 switches at top of `run_proposed_model.py`)
- `FIXED_SCHEDULE=True` — no early stopping, no best-checkpoint selection; final-epoch EMA weights are evaluated. Val metrics during training are monitoring only.
- `STAGE1_LOSS` ∈ {`focal`, `gce`, `sce`} — GCE/SCE are noise-robust options motivated by the label audit.
- `STAGE1_WEIGHTING` / `STAGE2_WEIGHTING` ∈ {`alpha`, `sampler`, `both`} — explicit single-weighting selection. The v3 bug (stage 2 applying sampler AND alpha) is fixed.
- Class weights: computed from the **fold's train labels only**, tempered `(1/freq)^ALPHA_POWER` (0.5 → ~3.5× spread; v3's raw inverse-frequency gave fear 12–14× and collapsed its precision to 0.37), `SADNESS_BOOST=1.5`, `FEAR_BOOST=1.0`, `ALPHA_MAX_RATIO=6` safety net.
- `STREAM_A_POOL` ∈ {`gap`, `spatial2x2`, `attn`} — default **`attn`**. Lesson: VGG16 *alone* (0.6114) beat the GAP-pooled hybrid (0.5981); GAP on the 7×7 map discards the useful signal, and SE on a pooled vector is just a learned rescaling. `DEIT_DROP_PATH=0.1` gives the DeiT stream the regularization its baseline had.
- `NUM_EPOCHS=110` + `EMA_WARMUP_EPOCHS=5` (decay ramps 0→0.999; v3's cold-start EMA wasted ~15–20 epochs/fold, fold-1 val acc was 8.6% at epoch 1). Stage 2: 20 epochs, backbone truly frozen (`set_backbone_eval()` keeps BN in eval).
- Reproducibility: per-fold `torch.Generator` (`SEED*1000+fold`) + `seed_everything(SEED+fold)`. The old single shared generator made resumed sessions diverge (measured: same fold, same code → F1 0.1337 vs 0.3727).
- Eval/TTA run in fp32 (v3 used fp16 autocast; prob rows summed to 0.9998–1.0002).
- Persistence order: OOF `.npy` + `oof_paths.json` (row→image index) → `cv_metrics.json` → `mark_done`. v3 marked done first — a crash in between lost the fold silently.
- Guardrail: `UNCERTAINTY_THRESH=0.50` + full risk–coverage sweep → `risk_coverage.json`. (0.30 was inert: min observed max-prob 0.2491 → flagged 4/1808.)
- Grad-CAM uses fold-1 weights on fold-1 **out-of-fold** images (v3 likely drew training images).

## Training gotchas
- **No shared module:** the two scripts inline everything. Cross-cutting changes = edit twice + regenerate both notebook cells.
- **Falsified — do not retry:** post-hoc probability tricks give ZERO gain on this setup (logit adjustment `p/prior^τ`, reverse prior adjust `p·prior^s`, temperature scaling, per-class thresholds — all tested on saved v3 OOF probs, honest nested best = 0.5977 vs 0.5981 raw). The model is already recall-rich/precision-starved; fixes must be at training time.
- **Do not trust v3-era "macro" ROC/PR figures** (computed micro). Regenerate before any paper use.
- vgg16 is the best baseline AND the worst-calibrated (7–27% of rows with max-prob > 0.99 in every fold vs 0.0% for transformers) — never drive a clinical threshold from its confidence.
- Differential LR: backbone `lr*0.1`, head full `lr` (head = `classifier`/`se_a`/`se_b`/`pool_a`). Warmup+cosine LambdaLR only — CosineAnnealingWarmRestarts destabilizes transformers.
- `vit_base` uses timm tag `vit_base_patch16_224.augreg_in21k`. All baselines run at 224 input.
- Checkpoints are dicts (`model_state_dict` for proposed; `state_dict` + `ema.shadow` for baselines) at `results/<name>/fold{k}_best.pth`. Proposed EMA is a deepcopy `ModelEMA` — save/load EMA weights, and its `update()` decays params AND BN buffers.
- Baseline `EMA` (run_all_models.py) shadows parameters only — different semantics from the proposed `ModelEMA`. Known, disclosed, frozen.

## Environment
- Local: Python 3.14, PyTorch 2.12.0+xpu (Intel Arc), timm 1.0.28, numpy 2.4.6, scikit-learn 1.9.0.
- Kaggle: PyTorch 2.10.0+cu128, Tesla T4. Scripts are CUDA-or-CPU.
- Gitignored: `dataset/`, `dataset_clean/`, `results/`, `*.zip`, `*.log`. Root zips/logs are large artifacts (`final_results (1).zip` ~8.9 GB, `results.zip` ~17 GB).

## File Map
- `PLAN.md` — objective, strategy, evidence table (current)
- `IMPROVE.md` — P0→P4 execution checklist with expected gains
- `label_audit_candidates.csv` — 274 consensus-flagged images + per-model votes (review queue)
- `final-autism-fer.ipynb` — the Kaggle notebook (4 cells; cell 3 == run_proposed_model.py, byte-identical)
- `final-autism-fer.log` — v3 Kaggle run log
- `final_results (1)/` — authoritative v3 output (see Results trees)
- `kaggle/run_all_models.py` — frozen 12-baseline sweep (v3-emabn)
- `kaggle/run_proposed_model.py` — proposed model (v4-proposed-opt)
- `kaggle/autism-fer-model.ipynb` — older notebook mirror
- `kaggle/preprocess_faces.py`, `kaggle/offline_augmentation.py` — legacy, unused
- `dataset/` — original 1,988 images (leaky, unused) · `dataset_clean/` — canonical 1,808
