# Kaggle Run Guide — Autism FER Pipeline

Quick reference so you never lose the exact steps to produce the baseline comparison
and the Proposed-Model head-to-head.

## Run on Kaggle (GPU T4/P100)

Both scripts are regenerated verbatim inside `kaggle/autism-fer-model.ipynb`
(3 cells). Keep the `.py` files and the notebook cells in sync — edit one, update the other.

1. **Attach your dataset.** Either slug works (auto-detected by folder structure):
   - `/kaggle/input/datasets/mrsohel/dataset-clean`
   - `/kaggle/input/datasets/mrsohel/autism-dataset/dataset_clean`

2. **Start a FRESH session** so `/kaggle/working/results` is empty.
   If reusing an existing session, **delete `/kaggle/working/results` first** —
   old-pipeline outputs must not mix with new ones.

3. **Run cell 2 = `run_all_models.py`** (12-model baseline sweep + paper figures).
   - Trains the 12 `EXPERIMENTS` models under Stratified 5-fold CV (RAW images).
   - Writes per-model `cv_metrics.json`, OOF `.npy`, `fold_id_by_path.json`, `cv_done.json`, `paper_figures/`.
   - Resumable: timed-out sessions continue from the last completed fold.

4. **Run cell 3 = `run_proposed_model.py`** — **must run AFTER cell 2** (reads `fold_id_by_path.json`).
   - Proposed-Model dual-stream (VGG16-BN + DeiT-S + dual SE) on the SAME folds.
   - 5-view TTA + uncertainty guardrail + Grad-CAM.
   - Results land in `/kaggle/working/results/proposed_model_proposed/`.

## Confirm success

- After cell 2: `cv_done.json` lists all 12 models with folds `[0,1,2,3,4]`.
- After cell 3: `cv_metrics.json` exists under `results/proposed_model_proposed/`
  for the head-to-head table vs. the top baseline.

## Notes / gotchas

- **Do not delete** the local `results/results/` sweep on this machine — it is an
  OLD pipeline (no `pipeline_version`, has `inception_v3`, missing
  `convnext_tiny`/`maxvit_tiny`/`efficientformer_l1`). It is NOT comparable to the
  current 12-model run. Use only fresh Kaggle numbers.
- `run_all_models.py` is Kaggle-only (no local fallback). `run_proposed_model.py`
  falls back to local `dataset_clean/` if `/kaggle` is absent.
- Current `PIPELINE_VERSION` badge: `v3-emabn` (both scripts). If you change the
  eval/training methodology, bump it — stale creds must be deleted before re-running.