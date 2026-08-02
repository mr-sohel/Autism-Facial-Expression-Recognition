# Baseline Model Comparison Report
## Autism Facial Expression Recognition — `kaggle/run_all_models.py`

**Prepared for:** Supervisor review
**Date:** 2026-08-01
**Purpose:** Establish rigorous baseline performance of 8 curated pretrained models on our autism 6-class emotion dataset, as the empirical foundation for proposing a novel architecture (Care-FER) with explainable AI (XAI).

---

## 1. Objective & Research Question

We classify **6 facial emotions in autistic children** — *anger, fear, joy, natural, sadness, surprise* — using transfer learning from ImageNet-pretrained models. This report answers:

1. How does each architecture family perform on the *same* data, under the *same* protocol?
2. Which classes are systematically hard (low per-class F1)?
3. Where is the headroom for a novel architecture + XAI to beat the competitors?

---

## 2. Experimental Protocol (as implemented)

Everything below is enforced identically for all 8 models — this is what makes the comparison fair.

| Component | Setting |
|---|---|
| **Data** | RAW images (no face-crop / CLAHE — that preprocessing measurably hurt accuracy, vgg16 F1 0.548→0.528, so it was removed) |
| **Split** | All 3 splits (train/valid/test) **merged**, then **Stratified K-Fold CV, K=5, seed=42**. Every image is predicted exactly once (out-of-fold) |
| **Class imbalance** | Handled by **WeightedRandomSampler ONLY** (single weighting; the old double-weighting bug is fixed) |
| **Augmentation** | Light: horizontal flip, rotation, affine, color jitter, random grayscale. MixUp/RandomErasing **removed** (too destructive at ~2k images) |
| **Loss** | CNNs → **FocalLoss (γ=2)**; Transformers → **CrossEntropy (label smoothing 0.1)** |
| **Optimizer** | AdamW, **differential LR**: backbone `lr×0.1`, head `lr` (CNNs `1e-3`, Transformers `1e-4` — higher LR collapses ViTs) |
| **Scheduler** | CosineAnnealingWarmRestarts (T0=10, T2×) |
| **Epochs / Patience** | 80 max / early-stop patience 15 |
| **EMA** | Exponential Moving Average (0.999) — evaluated, not raw weights |
| **Input size** | 224×224 (inception_v3 = 299×299, per-model) |
| **Metrics** | Accuracy, macro F1, macro Precision, macro Recall, per-class F1, mean ± std across folds |

**Reproducibility:** per-fold `.npy` OOF files + `cv_metrics.json` + `cv_done.json` resume markers — a timed-out Kaggle session resumes where it left off. Fold assignment persisted to `fold_id_by_path.json` (shared with the Care-FER script so both use **identical** folds).

---

## 3. Results

### 3.1 Overall comparison (Run 1 — prior single-split pipeline; K-fold CV numbers pending re-run on Kaggle)

Sorted by macro F1 (from `new new log.log`, old train/valid/test split — **the 5-fold CV version is the current script and will produce mean ± std**):

| Model | Architecture family | Acc | **F1 (macro)** | Prec | Rec | Params |
|---|---|---:|---:|---:|---:|---:|
| **vgg16** | CNN (classic) | 0.674 | **0.548** | 0.538 | 0.587 | 134M |
| **deit_small_patch16_224** | Transformer (S) | 0.678 | **0.544** | 0.538 | 0.570 | 22M |
| vgg19 | CNN | 0.678 | 0.543 | 0.539 | 0.580 | 140M |
| vit_base_patch16_224 | Transformer (B) | 0.671 | 0.535 | 0.525 | 0.561 | 86M |
| inception_v3 | CNN | 0.658 | 0.523 | 0.519 | 0.544 | 22M |
| densenet121 | CNN | 0.658 | 0.523 | 0.521 | 0.542 | 7M |
| vit_tiny_patch16_224 | Transformer | 0.635 | 0.514 | 0.512 | 0.572 | 6M |
| ghostnet_100 | CNN (efficient) | 0.638 | 0.514 | 0.503 | 0.537 | 4M |
| mobilenetv2_100 | CNN (efficient) | 0.625 | 0.498 | 0.498 | 0.514 | 2M |
| **swin_base_patch4_window7_224** | Transformer (windowed) | 0.632 | **0.494** | 0.484 | 0.518 | 87M |
| mobilenetv3_large_100 | CNN | 0.622 | 0.491 | 0.488 | 0.507 | 4M |
| resnet50 | CNN | 0.602 | 0.464 | 0.459 | 0.480 | 24M |
| resnet18 | CNN | 0.566 | 0.426 | 0.422 | 0.455 | 11M |
| convnext_small | CNN (modern) | 0.250 | 0.188 | 0.224 | 0.212 | 49M |
| coat_lite_small | Transformer hybrid | 0.316 | 0.171 | 0.179 | 0.222 | 19M |

> ConvNeXt and CoAt collapse on this small dataset — modern architectures are *not* automatically better; the 2k-scale data rewards strong inductive biases (vgg16) and efficiency (22M-param Deit-S).

### 3.2 Per-class macro F1 (Run 1) — the hard classes

| Model | anger | **fear** | **joy** | natural | **sadness** | surprise |
|---:|---:|---:|---:|---:|---:|---:|
| vgg16 | 0.62 | **0.29** | 0.92 | 0.63 | 0.32 | 0.75 |
| deit_small | 0.56 | **0.21** | 0.95 | 0.63 | 0.32 | 0.75 |
| vit_base | 0.56 | **0.21** | 0.93 | 0.60 | 0.35 | 0.71 |
| densenet121 | 0.41 | **0.29** | 0.93 | 0.54 | 0.38 | 0.71 |
| resnet50 | 0.38 | **0.21** | 0.84 | 0.60 | 0.39 | 0.46 |
| swin_base | 0.41 | **0.29** | 0.92 | 0.63 | 0.29 | 0.58 |

**Diagnosis:**
- **joy ≈ 0.84–0.96** (majority class, n=858) — easy.
- **fear ≈ 0.21–0.29** (rarest, n=87) and **sadness ≈ 0.29–0.39** — consistently the failure classes across *all* architectures. This is the class-imbalance bottleneck.
- **anger** splits models (vgg 0.62 vs densenet 0.41) — a feature-extraction sensitivity difference, not pure imbalance.

### 3.3 Key takeaways for the proposed model

1. **Classic CNN + compact Transformer are the two best families** (vgg16 0.548, deit_small 0.544) — but *complementary*: vgg16 wins anger/fear, Deit-S wins joy/natural. A **dual-stream fusion of both** should combine their strengths (motivation for **Care-FER**).
2. **fear/sadness need explicit handling** — FocalLoss α-weights + minority boosting (Care-FER: sadness ×2.0, fear ×1.2).
3. **Headroom exists:** even the best model misses ~45% of minority-class cases; a fusion model + attention should raise macro F1 meaningfully.
4. **XAI requirement:** Grad-CAM on the best CNNs + attention maps on the transformer stream can explain *where* the model looks for each emotion — the published competitor baselines use plain CNNs without interpretability.

---

## 4. Next Step: Proposed Architecture (Care-FER) with XAI

Based on §3, we propose **Care-FER**, a **dual-stream hybrid**:

- **Stream A (spatial CNN):** VGG16-BN `forward_features` + global avg pool → 512-d (captures the anger/fear detail vgg16 excels at).
- **Stream B (token transformer):** DeiT-S CLS token → 384-d (captures global joy/natural patterns).
- **Dual SE attention blocks (r=16)** re-weight both streams; head **896→512→256→6**.
- **Focal loss** with per-class α (joy 0.39, sadness 1.45×, fear 4.57×) + two-stage training (160 ep unbalanced → 20 ep frozen-backbone balanced).
- **XAI:** Grad-CAM (spatial stream) + class-token attention (transformer stream) + 5-view TTA with uncertainty guardrail.

**Target:** beat the 0.548 macro-F1 baseline and the published competitor range (~73–79% emotion accuracy in ASD children) *and* publish the interpretability results.

---

## 5. Current Script Status

- `kaggle/run_all_models.py` (this script) — **ready to run**, produces `comparison.json`, `cv_metrics.json` per model, and paper figures (`paper_figures/1_cv_grouped_bar_metrics.png` … `7_model_correlation_heatmap.png`).
- Expected runtime on Kaggle T4: ~8 models × 5 folds × ~5.4 min/fold ≈ **~3.5 h**.
- **Important:** table in §3.1 is from the *previous* single-split run. The K-fold CV numbers (mean ± std) will be produced when the notebook's Cell 2 runs on Kaggle and must supersede it in the final report.
