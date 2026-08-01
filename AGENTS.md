# AGENTS.md

## What This Is
Autism Facial Expression Recognition — 6-class classification (anger, fear, joy, natural, sadness, surprise). Comparative study: an 8-model baseline sweep vs. the proposed **Care-FER** dual-stream architecture. All real training code is self-contained scripts under `kaggle/`, designed to run on Kaggle GPUs (T4/P100).

## Workflow (Kaggle, canonical)
The full pipeline is `autism-fer-model.ipynb` (5 cells, run in order):
1. `!pip install -q --no-deps facenet-pytorch`
2. `offline_augmentation.py` — balance train split to ~600 img/class → `/kaggle/working/dataset_augmented`
3. `preprocess_faces.py` — MTCNN face detect + eye-align + CLAHE → `/kaggle/working/dataset_mtcnn` (reads the augmented set)
4. `run_all_models.py` — train 8 curated baselines
5. `run_proposed_model.py` — train Care-FER

Training scripts auto-prefer `/kaggle/working/dataset_mtcnn`, else fall back to `/kaggle/input/datasets/mrsohel/autism-dataset/dataset`.

## Commands (local)
- `python kaggle/preprocess_faces.py` — MTCNN+CLAHE over `dataset/` → `dataset_mtcnn/` (CPU default)
- `python kaggle/offline_augmentation.py` — balance to 600/class → `dataset_augmented/`
- `python kaggle/run_proposed_model.py` — Care-FER training; falls back to local `dataset/`, outputs `results/care_fer_proposed/`

**`src/`, `run_experiments.py`, and `kaggle/SETUP.md` no longer exist.** README.md / CLAUDE.md describe that deleted architecture — treat them as stale. `python src/train.py ...` will not work.

## Dataset
- `dataset/{train,valid,test}/{anger,fear,joy,natural,sadness,surprise}/` — 1400/299/304
- Severe imbalance: joy=602 vs fear=60 (10:1)
- Imbalance is addressed with **both** WeightedRandomSampler AND inverse-frequency class weights in the loss (FocalLoss `alpha` / CE `weight`) — NOT an "unweighted loss" as older docs claim. Care-FER additionally boosts sadness ×2.0 and fear ×1.2.

## Training Gotchas
- **No shared module:** `run_all_models.py` and `run_proposed_model.py` each inline their own datasets/losses/EMA/plots. Cross-script changes must be made twice.
- **`inception_v3` = 299 input** (all others 224). `MODEL_CONFIGS` carries per-model size; dataloaders are rebuilt per experiment.
- `vit_base` uses timm tag `vit_base_patch16_224.augreg_in21k`.
- **Differential LR:** backbone `lr*0.1`, head full `lr`. Head = params named `classifier`/`head`/`fc` (Care-FER adds `se_a`/`se_b`). CNNs train at `lr=1e-3` (focal loss); transformers at `lr=1e-4` (ce_smooth) — higher LR makes them collapse.
- Care-FER uses warmup + cosine LambdaLR. Do NOT switch to CosineAnnealingWarmRestarts — periodic LR spikes destabilized DeiT.
- Care-FER: VGG16-BN spatial features (forward_features + GAP, 512-d) + DeiT-S CLS token (384-d), dual SE blocks (r=16), head 896→512→256→6. Two stages: 160 ep (unbalanced shuffle loader), then 20 ep (frozen backbone + balanced sampler).
- Care-FER EMA is a `deepcopy` (`ModelEMA`) — save/load the EMA weights, not the raw model.
- Checkpoints are dicts (`state_dict`, `ema.shadow`, …) at `results/<name>_best.pth`; test eval applies EMA weights.
- `get_model()` in run_all_models.py catches `TypeError` (some timm models reject `drop_rate`/`drop_path_rate`) and retries `pretrained=False` on missing-weights `RuntimeError`.

## Preprocessing (preprocess_faces.py)
- MTCNN `keep_all`, min face 30 px, 20% padding, eye-alignment rotation, CLAHE (clip 2.0, tile 8) on LAB L-channel, resized 224, JPEG q95. 85% center-crop fallback. Skips existing outputs (rerun-safe).
- Needs `facenet-pytorch` + `opencv-python` — **not** in `requirements.txt` (preinstalled on Kaggle). Install manually for local runs.

## Environment
- Local machine: Python 3.14, PyTorch 2.12.0+xpu (Intel Arc), timm 1.0.28.
- Kaggle scripts are **CUDA-only** (`cuda` else `cpu`); no XPU autocast path — they run on CPU on this machine.
- `dataset/`, `results/`, `*.zip`, `*.log` are gitignored; root `*.zip`/`*.log` files are large experiment artifacts.

## File Map
- `autism-fer-model.ipynb` — canonical Kaggle notebook orchestrating the 4 scripts
- `kaggle/run_all_models.py` — 8-model baseline sweep + paper figures (`comparison.json`, `paper_figures/`)
- `kaggle/run_proposed_model.py` — Care-FER proposed model + 5-view TTA + uncertainty guardrail + Grad-CAM
- `kaggle/preprocess_faces.py` — MTCNN + CLAHE offline preprocessing
- `kaggle/offline_augmentation.py` — offline class balancing (to ~600/class)
- `dataset/` — committed 1400/299/304 split
