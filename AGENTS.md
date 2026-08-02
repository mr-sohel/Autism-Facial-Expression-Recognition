# AGENTS.md

## What This Is
Autism Facial Expression Recognition — 6-class classification (anger, fear, joy, natural, sadness, surprise). Comparative study: an 8-model baseline sweep vs. the proposed **Proposed-Model** dual-stream architecture. All real training code is self-contained scripts under `kaggle/`, designed to run on Kaggle GPUs (T4/P100).

## Workflow (Kaggle, canonical)
The full pipeline is `kaggle/autism-fer-model.ipynb` (3 cells, run in order):
1. Markdown overview (methodology + how to run).
2. `run_all_models.py` — train the 9 curated baselines (incl. `efficientnet_b0`) under Stratified 5-fold CV on **RAW** images.
3. `run_proposed_model.py` — train Proposed-Model on the **same** folds.

There is NO preprocessing, augmentation-balancing, or `!pip install` cell — the notebook reads the raw uploaded dataset (`/kaggle/input/datasets/mrsohel/autism-dataset/dataset`) directly. Both scripts are resumable: per-fold `.npy` OOF files + `cv_metrics.json` / resume markers mean a timed-out Kaggle session just continues where it left off. Cell 2 must run before Cell 3 (Proposed-Model loads `fold_id_by_path.json` produced by Cell 2).

## Commands (local)
- `python kaggle/run_all_models.py` — baselines + CV; Kaggle-only (hardcoded Kaggle input path)
- `python kaggle/run_proposed_model.py` — Proposed-Model; falls back to local `dataset/`, outputs `results/proposed_model_proposed/`. Local runs are CPU (no XPU autocast path).
- `python kaggle/preprocess_faces.py` / `python kaggle/offline_augmentation.py` — **legacy, NOT used by the notebook.** Kept for reference; MTCNN+CLAHE measurably hurt accuracy (vgg16 F1 0.548→0.528) and the offline-augmented set double-counted images.

**`src/`, `run_experiments.py`, and `kaggle/SETUP.md` no longer exist.** README.md / CLAUDE.md describe that deleted architecture — treat them as stale. `python src/train.py ...` will not work.

## Dataset
- **Canonical training set: `dataset_clean/`** — 1,808 unique images after dHash dedup (88 near-dup clusters, 86 label conflicts removed; see `cleaning_report.json`). Counts: anger 167, fear 68, joy 843, natural 201, sadness 404, surprise 125. Both scripts point here.
- `dataset/` — the ORIGINAL 1,988-image set (leaky: same faces labeled differently across source datasets, dup clusters straddled fold boundaries). Kept untouched; NOT used by the pipeline.
- Severe imbalance: joy=843 vs fear=68 (~12:1)
- The three splits are **merged and re-split with StratifiedKFold(5, shuffle, seed 42)** — every image is predicted exactly once (out-of-fold), so fear/surprise metrics (n~14/fold) are statistically defensible. Report mean ± std across folds.
- **Single weighting** (the old double-weighting bug is fixed): baselines use WeightedRandomSampler only; Proposed-Model uses FocalLoss `alpha` only. Proposed-Model additionally boosts sadness ×2.0 and fear ×1.2.

## Training Gotchas
- **No shared module:** `run_all_models.py` and `run_proposed_model.py` each inline their own datasets/losses/EMA/plots. Cross-script changes must be made twice.
- **`inception_v3` = 299 input** (all others 224). `MODEL_CONFIGS` carries per-model size; dataloaders are rebuilt per experiment.
- `vit_base` uses timm tag `vit_base_patch16_224.augreg_in21k`.
- **Differential LR:** backbone `lr*0.1`, head full `lr`. Head = params named `classifier`/`head`/`fc` (Proposed-Model adds `se_a`/`se_b`). CNNs train at `lr=1e-3` (focal loss); transformers at `lr=1e-4` (ce_smooth) — higher LR makes them collapse.
- Proposed-Model uses warmup + cosine LambdaLR. Do NOT switch to CosineAnnealingWarmRestarts — periodic LR spikes destabilized DeiT.
- Proposed-Model: VGG16-BN spatial features (forward_features + GAP, 512-d) + DeiT-S CLS token (384-d), dual SE blocks (r=16), head 896→512→256→6. Two stages: 160 ep (unbalanced shuffle loader), then 20 ep (frozen backbone + balanced sampler).
- Proposed-Model EMA is a `deepcopy` (`ModelEMA`) — save/load the EMA weights, not the raw model.
- Checkpoints are dicts (`state_dict`, `ema.shadow`, …) at `results/<name>/fold{k}_best.pth`; test eval applies EMA weights. Per-model OOF predictions/labels/probs are saved as incremental `.npy` files so a timed-out fold/session resumes without losing progress.
- `get_model()` in run_all_models.py catches `TypeError` (some timm models reject `drop_rate`/`drop_path_rate`) and retries `pretrained=False` on missing-weights `RuntimeError`.

## Preprocessing (preprocess_faces.py, LEGACY - not in the pipeline)
- MTCNN `keep_all`, min face 30 px, 20% padding, eye-alignment rotation, CLAHE (clip 2.0, tile 8) on LAB L-channel, resized 224, JPEG q95. 85% center-crop fallback. Skips existing outputs (rerun-safe).
- Needs `facenet-pytorch` + `opencv-python` — **not** in `requirements.txt` (preinstalled on Kaggle). Install manually for local runs.
- Measurably hurt accuracy (vgg16 F1 0.548→0.528) — the notebook trains on RAW images instead.

## Environment
- Local machine: Python 3.14, PyTorch 2.12.0+xpu (Intel Arc), timm 1.0.28.
- Kaggle scripts are **CUDA-only** (`cuda` else `cpu`); no XPU autocast path — they run on CPU on this machine.
- `dataset/`, `results/`, `*.zip`, `*.log` are gitignored; root `*.zip`/`*.log` files are large experiment artifacts.

## File Map
- `kaggle/autism-fer-model.ipynb` — canonical Kaggle notebook orchestrating the 2 scripts
- `kaggle/run_all_models.py` — 9-model baseline sweep + paper figures (`comparison.json`, `paper_figures/`); produces `fold_id_by_path.json`
- `kaggle/run_proposed_model.py` — Proposed-Model proposed model + 5-view TTA + uncertainty guardrail + Grad-CAM
- `kaggle/preprocess_faces.py` — MTCNN + CLAHE offline preprocessing (legacy, unused)
- `kaggle/offline_augmentation.py` — offline class balancing (legacy, unused)
- `dataset/` — original 1988 raw images (leaky; NOT used by pipeline)
- `dataset_clean/` — canonical 1,808-image cleaned set (dedup'd, conflicts removed; `_removed/` quarantine)
