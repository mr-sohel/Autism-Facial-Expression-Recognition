# Autism Facial Emotion Recognition:\
Project Progress & Baseline Comparison

## 1. Project Overview
The goal of this research is to develop a deep learning framework to accurately classify six fundamental facial emotions (anger, fear, joy, natural, sadness, surprise) in children with Autism Spectrum Disorder (ASD). Because children with ASD express emotions differently—often with less symmetrical or synchronized facial movements—standard emotion recognition models usually struggle to read their faces.

## 2. Dataset

<style>
@page { size: A4; margin: 2.54cm; }
table, th, td { border: 1px solid black; border-collapse: collapse; padding: 6px; }
th { background-color: #f2f2f2; }
</style>

Our **primary dataset** for this research is [Nora Mahmoud's Mendeley Dataset](https://data.mendeley.com/datasets/b33pf78h62/1). However, because ASD facial datasets are inherently small, we strategically merged our primary dataset with three other sources. This aggregation significantly increased the total number of images, allowing our models to learn more robust and generalized ASD-specific facial cues.

The unified corpus is aggregated from these 4 sources:

| Source | Description |
|--------|-------------|
| [Nora Mahmoud's Mendeley Dataset](https://data.mendeley.com/datasets/b33pf78h62/1) | **Primary Dataset:** 6-class facial emotion categories |
| [FERAC Dataset](https://www.kaggle.com/datasets/) | 4-class ASD facial expressions |
| [Dr. Fatma M. Talaat (Kaggle)](https://www.kaggle.com/datasets/fatmamtalaat/autistic-children-emotions-dr-fatma-m-talaat) | ASD facial emotion data |
| [Hasibur Rahman's Kaggle Dataset](https://www.kaggle.com/datasets/mdhasiburrahman12/augmented-autism-facial-emotion-recognition) | ASD facial expression samples |

**Data Cleaning & Unique Raw Images:**
We took only the unique, raw images from all the collected datasets and combined them to create a new, clean dataset (`dataset_clean/`). You can access our final merged dataset here: [Merged Dataset (Google Drive)](https://drive.google.com/file/d/1x7zQPdxqIzRn_UUlJplSoJ70XThYfAE2/view?usp=sharing).

**Data Splitting:**
The cleaned images were then unified and stratified into a **70/15/15** ratio (Train/Validation/Test).

<div style="page-break-inside: avoid;">

**Final split** (`dataset/`):

| Split | anger | fear | joy | natural | sadness | surprise | Total |
|-------|------:|-----:|----:|--------:|--------:|---------:|------:|
| Train | 147 | 60 | 602 | 161 | 321 | 109 | **1,400** |
| Valid | 31 | 13 | 129 | 34 | 69 | 23 | **299** |
| Test | 32 | 14 | 130 | 35 | 69 | 24 | **304** |

> **Class imbalance:** joy (602) vs fear (60) = 10:1 ratio. Mitigated via `WeightedRandomSampler` (loss is unweighted to avoid double-weighting).

</div>

## 3. Baseline Model Comparison

We evaluated 10 standard baseline models on the unaugmented raw dataset using a Stratified 5-Fold Cross-Validation approach to establish our benchmark.

### Summary Table

| Model | Accuracy | F1-Macro | Precision | Recall |
|---|---|---|---|---|
| **VGG-16** | **0.7201** | **0.6217** | **0.6098** | **0.6725** |
| **Swin Base** | 0.7201 | 0.6100 | 0.5961 | 0.6485 |
| **Inception V3** | 0.7196 | 0.6067 | 0.6030 | 0.6363 |
| **DeiT Small** | 0.7113 | 0.6026 | 0.5895 | 0.6417 |
| **DenseNet-121** | 0.7113 | 0.6001 | 0.5894 | 0.6298 |
| **ViT Base** | 0.7102 | 0.5984 | 0.5880 | 0.6339 |
| **Swin Tiny** | 0.7030 | 0.5972 | 0.5827 | 0.6381 |
| **EfficientNet-B0**| 0.6980 | 0.5871 | 0.5813 | 0.6167 |
| **MobileNetV2** | 0.6858 | 0.5790 | 0.5687 | 0.6174 |
| **ResNet-50** | 0.6825 | 0.5762 | 0.5652 | 0.6150 |

---

<div style="page-break-before: always;"></div>

<div style="page-break-inside: avoid;">

## 4. Key Charts and Takeaways

### 1. Overall Performance
![Grouped Bar Metrics](results/results/paper_figures/1_cv_grouped_bar_metrics.png){width=450px}

**Takeaway:** VGG-16 and Swin Base achieved the highest overall performance across the evaluated metrics, making them the strongest standard architectures.

</div>

### 2. Balancing Rare Emotions
![F1 Comparison](results/results/paper_figures/2_cv_f1_comparison.png){width=450px}

**Takeaway:** The Macro-F1 score indicates how effectively a model manages class imbalance rather than defaulting to majority classes. CNNs currently set the benchmark for maintaining precision on minority classes.

<div style="page-break-before: always;"></div>

<div style="page-break-inside: avoid;">

### 3. ROC Curve (Top Models)
![ROC Curve](results/results/paper_figures/3_roc_curve.png){width=450px}

**Takeaway:** The leading models achieve high Area Under the Curve (AUC) scores, demonstrating that the raw image data contains sufficient discriminatory features for accurate classification.

</div>

### 4. Precision vs Recall
![Precision Recall Curve](results/results/paper_figures/4_pr_curve.png){width=450px}

**Takeaway:** The tight clustering among the leading models indicates a strong capacity to maintain high precision without sacrificing recall, which is essential for handling minority classes like fear and surprise.

<div style="page-break-inside: avoid;">

### 5. Radar Chart
![Radar Chart](results/results/paper_figures/5_radar_chart.png){width=450px}

**Takeaway:** The multi-metric footprint visually confirms the well-rounded performance of the top baseline models, establishing the target performance threshold for our future hybrid architecture.

</div>

<div style="page-break-inside: avoid;">

### 6. Where the Best Baseline Struggles
![Best Model Confusion Matrix](results/results/paper_figures/6_best_model_oof_cm.png){width=450px}

**Takeaway:** The confusion matrix for VGG-16 reveals that the model occasionally misclassifies "Fear" and "Surprise." Addressing these specific misclassifications will be a primary goal of our proposed hybrid model.

</div>

<div style="page-break-inside: avoid;">

### 7. Feature Correlation Analysis
![Model Correlation Heatmap](results/results/paper_figures/7_model_correlation_heatmap.png){width=450px}

**Takeaway:** The correlation heatmap indicates that CNN- and Transformer-based models capture different aspects of facial features. This complementary observation directly motivates our plan to design a hybrid CNN-Transformer architecture in the next phase of the research.

</div>

## 5. Kaggle Training Implementation

We ran the baseline benchmarks on Kaggle GPUs using the standalone script `kaggle/run_all_models.py`. You can view and run the full experimental pipeline in our [Kaggle Notebook](https://www.kaggle.com/code/mrsohel/autism-fer-model). The training pipeline was customized for our dataset:

* **5-Fold Cross-Validation:** We used Stratified 5-Fold CV on the raw images so every image is tested exactly once. This gives us reliable metrics even for rare emotions like "Fear".
* **Careful Learning Rates:** We trained the pretrained backbone layers very slowly (10x slower than the classification head) to avoid losing their learned features. 
* **Handling Imbalance:** To fix the severe class imbalance, we oversampled rare classes during training using a `WeightedRandomSampler` instead of heavily weighting the loss function.
* **On-the-Fly Augmentation:** We applied lightweight, dynamic data augmentations (random flips, rotations, scaling, and color jitter) during training. Heavy augmentations (like MixUp) were intentionally excluded to preserve delicate facial features.
* **Stable Training:** We used mixed-precision (AMP) for faster GPU training and an Exponential Moving Average (EMA) to help the models generalize better.
* **Resumable:** Since Kaggle sessions often timeout, the script automatically saves its progress after every fold. If it crashes, restarting the script picks up exactly where it left off.
* **Auto-Generated Charts:** Once finished, the script automatically generates all the performance charts (ROC, Radar, etc.) shown in this report.