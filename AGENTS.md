# AGENTS.md

## What This Is
Autism Facial Expression Recognition — 6-class image classification (anger, fear, joy, natural, sadness, surprise). Trains 14 deep learning models (CNNs + Transformers) for comparative evaluation.

## Run Commands
- **Local training (one model):** `python src/train.py --model <name> --loss ce_smooth --epochs 80 --batch-size 16 --mixup --ema`
- **Local training (all models):** `python run_experiments.py`
- **Kaggle training:** Upload `master_dataset_split/` as Kaggle dataset, run `kaggle/run_all_models.py` with GPU enabled (see `kaggle/SETUP.md`)

## Dataset
- Location: `master_dataset_split/{train,valid,test}/{anger,fear,joy,natural,sadness,surprise}/`
- Split: 1400 train / 299 valid / 304 test
- **Severe class imbalance:** joy=602 vs fear=60 (10:1 ratio). Handled via WeightedRandomSampler + class-weighted loss.
- 4 raw datasets aggregated into this split (see `Datasets/` folder).

## Architecture: Key Gotchas
- **`cvt_13` is not a real model name.** `run_experiments.py` uses `"cvt_13"` but `src/models.py` remaps it to `coatnet_1_224` via timm. Don't add `cvt_13` to MODEL_CONFIGS — it's already handled.
- **`inception_v3` expects 299x299 input** — all other models use 224 (except `crossvit_9_240` which uses 240). The pipeline handles this automatically.
- **`models.py` catches `TypeError`** when creating models — some timm models don't accept `drop_rate`/`drop_path_rate`. If adding new models, test with `pretrained=False` first.
- **Differential LR:** backbone gets `lr * 0.1`, head/classifier gets full `lr`. Pattern: params with `"classifier"`, `"head"`, or `"fc"` in name are head params.

## Device Handling
- `src/utils.py:get_device()` auto-detects CUDA > XPU (Intel Arc) > CPU
- `src/train.py` has separate autocast paths for XPU (bfloat16) and CUDA (float16)
- Kaggle notebook hardcodes CUDA only

## File Map
```
src/dataset.py    — Dataset class, augmentations, WeightedRandomSampler, class weights
src/models.py     — MODEL_CONFIGS dict, get_model()
src/losses.py     — FocalLoss, LabelSmoothingCrossEntropy, get_loss_fn()
src/utils.py      — EMA, MixUp, compute_metrics(), TrainingLogger, device detection
src/evaluate.py   — Confusion matrix plots, training curves, per-class F1 bars
src/train.py      — Single-model training entrypoint (CLI args)
run_experiments.py — Runs all 14 models sequentially via subprocess
kaggle/run_all_models.py — Self-contained Kaggle notebook (duplicates src/ code)
```

## Environment
- Python 3.14, PyTorch 2.12.0+xpu, timm 1.0.28
- No `pip install` needed locally — all deps in `requirements.txt`
- `src/train.py` uses `sys.path.insert(0, os.path.dirname(...))` to find siblings — run from repo root
