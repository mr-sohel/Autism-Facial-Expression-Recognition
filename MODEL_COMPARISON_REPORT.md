---
title: "Autism Facial Emotion Recognition"
subtitle: "Project Progress & Baseline Model Comparison"
date: "August 2026"
---

<style>
@page { size: A4; margin: 2.5cm; }
body { font-family: "Segoe UI", Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22pt; color: #104e8b; border-bottom: 2px solid #104e8b; padding-bottom: 6px; }
h2 { font-size: 15pt; color: #104e8b; border-bottom: 1px solid #b0c4de; padding-bottom: 4px; }
h3 { font-size: 12pt; color: #2f4f4f; }
table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #888; padding: 6px 8px; text-align: left; }
th { background-color: #eef3fa; font-weight: 600; }
td { text-align: center; }
td:first-child { text-align: left; }
figure { text-align: center; margin: 16px 0; }
figure img { border: 1px solid #ccc; padding: 4px; }
figcaption { font-size: 10pt; font-style: italic; color: #555; margin-top: 6px; }
blockquote { border-left: 4px solid #104e8b; margin: 12px 0; padding: 8px 14px; background-color: #f7f9fc; color: #333; }
a { color: #104e8b; }
</style>

## Abstract

This report documents the current state of a research project to develop a deep-learning framework that accurately classifies six fundamental facial emotions (anger, fear, joy, natural, sadness, surprise) in children with Autism Spectrum Disorder (ASD). Because children with ASD exhibit atypical facial dynamics (e.g., reduced symmetry, asynchronous activation), standard emotion-recognition models often struggle. We aggregate four small ASD-specific datasets into a unified corpus, benchmark ten standard CNN and Transformer baselines under stratified 5-fold cross-validation, and motivate a hybrid architecture for the next phase.

---

## 1. Project Overview

The goal of this research is to develop a deep learning framework to accurately classify six fundamental facial emotions (**anger, fear, joy, natural, sadness, surprise**) in children with Autism Spectrum Disorder (ASD). Because children with ASD exhibit atypical facial dynamics (e.g., reduced symmetry, asynchronous activation), standard emotion recognition models often struggle to generalize to this population.

The current phase establishes a rigorous **baseline benchmark**. We evaluate ten standard CNN and Transformer architectures on a unified, cleaned dataset using stratified 5-fold cross-validation. This benchmark defines the target performance threshold that our proposed hybrid architecture must exceed in the next phase.

---

## 2. Dataset

Our **primary dataset** is [Nora Mahmoud's Mendeley Dataset](https://data.mendeley.com/datasets/b33pf78h62/1). Because ASD facial datasets are inherently small, we strategically merged the primary dataset with three other sources. This aggregation significantly increased the total number of images, allowing our models to learn more robust and generalized ASD-specific facial cues.

### 2.1 Data Sources

The unified corpus is aggregated from the following four sources:

| Source | Description |
|--------|-------------|
| [Nora Mahmoud's Mendeley Dataset](https://data.mendeley.com/datasets/b33pf78h62/1) | **Primary Dataset:** 6-class facial emotion categories |
| [FERAC Dataset](https://www.kaggle.com/datasets/) | 4-class ASD facial expressions |
| [Dr. Fatma M. Talaat (Kaggle)](https://www.kaggle.com/datasets/fatmamtalaat/autistic-children-emotions-dr-fatma-m-talaat) | ASD facial emotion data |
| [Hasibur Rahman's Kaggle Dataset](https://www.kaggle.com/datasets/mdhasiburrahman12/augmented-autism-facial-emotion-recognition) | ASD facial expression samples |

### 2.2 Cleaning & Splitting

- **Cleaning:** We took only the unique, raw images from all collected datasets and combined them to create a new, clean dataset (`dataset_clean/`).
- **Splitting:** The cleaned images were then unified and stratified into a **70/15/15** ratio (Train / Validation / Test).

**Final per-class split** (`dataset/`):

| Split | anger | fear | joy | natural | sadness | surprise | Total |
|-------|------:|-----:|----:|--------:|--------:|---------:|------:|
| Train | 147   | 60   | 602 | 161     | 321     | 109      | **1,400** |
| Valid | 31    | 13   | 129 | 34      | 69      | 23       | **299** |
| Test  | 32    | 14   | 130 | 35      | 69      | 24       | **304** |

> **Class imbalance:** joy (602) vs fear (60) is a 10:1 ratio. This is mitigated via a `WeightedRandomSampler` at the DataLoader level (loss is left unweighted to avoid double-weighting).

---

## 3. Baseline Model Comparison

We evaluated **10 standard baseline models** on the unaugmented raw dataset using a **Stratified 5-Fold Cross-Validation** approach to establish our benchmark. Every image is predicted exactly once out-of-fold (OOF), making metrics for rare classes (such as *fear*) statistically defensible.

### 3.1 Summary Table

| Model | Accuracy | F1-Macro | Precision | Recall |
|-------|---------:|---------:|----------:|-------:|
| **VGG-16** | **0.7201** | **0.6217** | **0.6098** | **0.6725** |
| **Swin Base** | 0.7201 | 0.6100 | 0.5961 | 0.6485 |
| **Inception V3** | 0.7196 | 0.6067 | 0.6030 | 0.6363 |
| **DeiT Small** | 0.7113 | 0.6026 | 0.5895 | 0.6417 |
| **DenseNet-121** | 0.7113 | 0.6001 | 0.5894 | 0.6298 |
| **ViT Base** | 0.7102 | 0.5984 | 0.5880 | 0.6339 |
| **Swin Tiny** | 0.7030 | 0.5972 | 0.5827 | 0.6381 |
| **EfficientNet-B0** | 0.6980 | 0.5871 | 0.5813 | 0.6167 |
| **MobileNetV2** | 0.6858 | 0.5790 | 0.5687 | 0.6174 |
| **ResNet-50** | 0.6825 | 0.5762 | 0.5652 | 0.6150 |

> **Result:** VGG-16 achieves the highest Macro-F1 (0.6217) and recall (0.6725), tied with Swin Base on accuracy (0.7201).

---

## 4. Key Charts & Takeaways

This section presents the comparative visualizations generated from the out-of-fold predictions of all baseline models.

### 4.1 Overall Performance

![Grouped bar chart comparing accuracy, precision, recall, and Macro-F1 across all ten baselines.](results/results/paper_figures/1_cv_grouped_bar_metrics.png){width=13.5cm}

**Takeaway:** VGG-16 and Swin Base achieved the highest overall performance across the evaluated metrics, making them the strongest standard architectures.

### 4.2 Balancing Rare Emotions

![Macro-F1 comparison across models, highlighting robustness to class imbalance.](results/results/paper_figures/2_cv_f1_comparison.png){width=13.5cm}

**Takeaway:** The Macro-F1 score indicates how effectively a model manages class imbalance rather than defaulting to majority classes. CNNs currently set the benchmark for maintaining precision on minority classes.

### 4.3 ROC Curve (Top Models)

![Receiver-operating-characteristic (ROC) curves for the top-performing models.](results/results/paper_figures/3_roc_curve.png){width=13.5cm}

**Takeaway:** The leading models achieve high Area Under the Curve (AUC) scores, demonstrating that the raw image data contains sufficient discriminatory features for accurate classification.

### 4.4 Precision vs. Recall

![Precision-recall curves illustrating the trade-off for minority classes.](results/results/paper_figures/4_pr_curve.png){width=13.5cm}

**Takeaway:** The tight clustering among the leading models indicates a strong capacity to maintain high precision without sacrificing recall, which is essential for handling minority classes like *fear* and *surprise*.

### 4.5 Radar Chart

![Multi-metric radar footprint of the baseline models.](results/results/paper_figures/5_radar_chart.png){width=13.5cm}

**Takeaway:** The multi-metric footprint visually confirms the well-rounded performance of the top baseline models, establishing the target performance threshold for our future hybrid architecture.

### 4.6 Where the Best Baseline Struggles

![Out-of-fold confusion matrix for the best baseline (VGG-16).](results/results/paper_figures/6_best_model_oof_cm.png){width=13.5cm}

**Takeaway:** The confusion matrix for VGG-16 reveals that the model occasionally misclassifies *fear* and *surprise*. Addressing these specific misclassifications will be a primary goal of our proposed hybrid model.

### 4.7 Feature Correlation Analysis

![Correlation heatmap between model predictions.](results/results/paper_figures/7_model_correlation_heatmap.png){width=13.5cm}

**Takeaway:** The correlation heatmap indicates that CNN- and Transformer-based models capture different aspects of facial features. This complementary observation directly motivates our plan to design a hybrid CNN-Transformer architecture in the next phase of the research.

---

## 5. Kaggle Training Implementation

The baseline benchmarking was executed entirely on Kaggle GPU instances using the standalone script `kaggle/run_all_models.py`. The training process was designed specifically to handle the small scale and imbalance of our ASD dataset while mitigating Kaggle session timeouts.

- **Stratified 5-Fold CV:** All models were evaluated using Stratified 5-Fold Cross-Validation on the raw images. This ensures every image is predicted exactly once out-of-fold (OOF), making metrics for rare classes (like *fear*) statistically defensible.
- **Differential Learning Rates:** To prevent destabilizing pretrained weights, we applied a differential learning rate (the backbone uses `lr * 0.1`, while the head uses full `lr`). Transformers were trained conservatively (`lr=1e-4` with Label-Smoothed Cross-Entropy), while standard CNNs were trained more aggressively (`lr=1e-3` with Focal Loss).
- **Handling Class Imbalance:** To avoid over-regularizing small models via loss weighting, the severe class imbalance was handled strictly via a `WeightedRandomSampler` at the DataLoader level.
- **Generalization (EMA & AMP):** Mixed-precision training (AMP) accelerated execution on the GPU, while an Exponential Moving Average (EMA) of model weights was maintained to boost final generalization.
- **Timeout-Resilient Execution:** Because Kaggle sessions often time out during long sweeps, the script is inherently resumable. It incrementally persists OOF predictions (`.npy`) and performance metrics (`cv_metrics.json`) to disk after every single fold. If a session is interrupted, re-running the script automatically skips completed folds and resumes exactly where it stopped.
- **Automated Figure Generation:** Upon completion, the script automatically parses the finalized OOF arrays to generate all of the comparative graphs (ROC, PR Curve, Radar Chart, Heatmaps) featured in this report.

---

## 6. Conclusion & Next Steps

The benchmark confirms that pretrained CNN and Transformer baselines perform competitively on the ASD emotion-classification task, with **VGG-16** and **Swin Base** leading on the majority of metrics. The main remaining weakness is misclassification between the rare and visually similar *fear* and *surprise* classes.

The correlation analysis reveals complementary behavior between CNN- and Transformer-based models. This motivates the **next phase**: designing and training a hybrid CNN-Transformer (dual-stream) architecture that combines the spatial detail of CNNs with the global context of Transformers, explicitly targeting the minority-class errors highlighted in this report.