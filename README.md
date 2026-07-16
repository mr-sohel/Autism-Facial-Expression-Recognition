# Autism Facial Expression Recognition

A comparative study of 15 deep learning architectures for classifying facial expressions in individuals with Autism Spectrum Disorder (ASD). The study evaluates CNNs, Vision Transformers, and a custom hybrid model (EssWin6) across 6 emotion classes.

## Dataset

The unified corpus is aggregated from 4 sources, deduplicated, and split 70/15/15:

| Source | Description |
|--------|-------------|
| [FERAC Dataset](https://www.kaggle.com/datasets/) | 4-class ASD facial expressions |
| Nora Mahmoud's Mendeley Dataset | ASD/Non-ASD labeled faces |
| Dr. Fatma M. Talaat (Kaggle) | ASD facial emotion data |
| Hasibur Rahman's Kaggle Dataset | ASD facial expression samples |

**Final split** (`master_dataset_split/`):

| Split | anger | fear | joy | natural | sadness | surprise | Total |
|-------|------:|-----:|----:|--------:|--------:|---------:|------:|
| Train | 147 | 60 | 602 | 161 | 321 | 109 | **1,400** |
| Valid | 31 | 13 | 129 | 34 | 69 | 23 | **299** |
| Test | 32 | 14 | 130 | 35 | 69 | 24 | **304** |

> **Class imbalance:** joy (602) vs fear (60) = 10:1 ratio. Mitigated via `WeightedRandomSampler` + class-weighted loss.

## Architecture

### CNN Models
| Model | Params | Input Size |
|-------|-------:|:----------:|
| VGG-16 (BN) | 134.3M | 224 |
| VGG-19 (BN) | 139.6M | 224 |
| MobileNetV2 | 2.2M | 224 |
| MobileNetV3-Large | 4.2M | 224 |
| InceptionV3 | 21.8M | 299 |
| EfficientNetV2-S | 20.2M | 224 |
| EfficientNetV2-M | 52.9M | 224 |
| ResNet-50 | 23.5M | 224 |
| DenseNet-121 | 7.0M | 224 |
| ConvNeXt-Small | 49.5M | 224 |

### Transformer & Hybrid Models
| Model | Params | Input Size |
|-------|-------:|:----------:|
| ViT-B/16 | 85.8M | 224 |
| Swin-B | 86.7M | 224 |
| CoAtNet-1 (via `cvt_13`) | 41.5M | 224 |
| CrossViT-9 | 8.2M | 240 |

### Proposed Model: EssWin6

A custom dual-branch hybrid combining:
- **Branch 1:** EfficientNet-V2-S (CNN feature extraction)
- **Branch 2:** Swin Transformer-B (global context)
- **Fusion:** Gated attention mechanism with learned projection
- **Head:** LayerNorm → Dropout → FC → GELU → Dropout → FC

Uses Focal Loss; all baselines use Label Smoothing Cross-Entropy.

## Training

### Local (single model)
```bash
python src/train.py --model <name> --loss ce_smooth --epochs 80 --batch-size 16 --mixup --ema
```

### Local (all 15 models)
```bash
python run_experiments.py
```

### Kaggle (recommended)
Upload `master_dataset_split/` as a Kaggle dataset, then run `kaggle/run_all_models.py` with GPU enabled. See [`kaggle/SETUP.md`](kaggle/SETUP.md) for step-by-step instructions.

### Training Details
- **Optimizer:** AdamW with differential LR (backbone 0.1x, head 1x)
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- **Regularization:** MixUp (α=0.4), EMA (decay=0.999), dropout, weight decay
- **Augmentations:** Random flip, rotation (±15°), affine, color jitter, grayscale, Gaussian blur, random erasing
- **Class balancing:** WeightedRandomSampler + class-weighted loss
- **Early stopping:** Patience 15 epochs on validation F1-macro
- **Seed:** 42

## Evaluation

Metrics reported per model:
- Accuracy, Macro F1, Macro Precision, Macro Recall
- Per-class F1 scores
- Confusion matrices (normalized)
- Training/validation loss and accuracy curves

### Results Output
```
results/
├── <model_name>/
│   ├── best_model.pth
│   ├── <model>_test_metrics.json
│   ├── logs/<model>_history.json
│   └── plots/
│       ├── <model>_cm.png
│       ├── <model>_f1_per_class.png
│       └── <model>_curves.png
├── comparisons/
│   ├── all_results.json
│   ├── summary_table.txt
│   ├── comparison_accuracy.png
│   ├── comparison_f1_macro.png
│   └── ...
```

## Project Structure

```
├── src/
│   ├── dataset.py        # Dataset, augmentations, WeightedRandomSampler
│   ├── models.py         # MODEL_CONFIGS, EssWin6, get_model()
│   ├── losses.py         # FocalLoss, LabelSmoothingCrossEntropy
│   ├── utils.py          # EMA, MixUp, metrics, device detection
│   ├── evaluate.py       # Plots, confusion matrix, comparison charts
│   └── train.py          # Training entrypoint (CLI)
├── kaggle/
│   ├── run_all_models.py # Self-contained Kaggle notebook
│   └── SETUP.md          # Kaggle setup instructions
├── master_dataset_split/ # Train/Valid/Test splits
├── Datasets/             # Raw source datasets
├── run_experiments.py    # Run all 15 models locally
└── requirements.txt      # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.0+, and `timm`. GPU recommended; supports CUDA and Intel XPU (Arc).

## References

- Radočaj & Martinović (2025) — CNN vs Transformer comparison on FERAC
- Autoencoder-preprocessing + Xception/InceptionV3 (arXiv, 2025)
- Hybrid ResNet50V2 + InceptionV3 (ScienceDirect, 2025)
- Real-time FER + IoT system (Neural Computing and Applications, 2023)
