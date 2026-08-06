# Autism Facial Emotion Recognition:
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
We took only the unique, raw images from all the collected datasets, removed duplicates and label conflicts, and combined them to create a new, clean dataset (`dataset_clean/`). You can access our final merged dataset here: [Merged Dataset (Google Drive)](https://drive.google.com/file/d/1x7zQPdxqIzRn_UUlJplSoJ70XThYfAE2/view?usp=sharing).

<div style="page-break-inside: avoid;">

**Canonical Dataset Distribution** (`dataset_clean/`):

| Emotion | Count |
|---------|------:|
| Anger | 167 |
| Fear | 68 |
| Joy | 843 |
| Natural | 201 |
| Sadness | 404 |
| Surprise | 125 |
| **Total** | **1,808** |

> **Class imbalance:** Joy (843) vs Fear (68) = ~12:1 ratio. Mitigated via `WeightedRandomSampler` and Stratified 5-Fold Cross-Validation.

</div>

## 3. Training Methodology & Handling Imbalance

To ensure fair and accurate training, we implemented several best practices:

* **Stratified 5-Fold Cross-Validation (The Great Merger):** Even though the `dataset_clean/` directory contains physical `train`, `valid`, and `test` folders, the training script completely ignores these boundaries. First, it merges every single image into one giant pool of 1,808 images in RAM. Then, it mathematically chops this pool into 5 equal chunks (folds). For each fold, it uses 4 chunks (80%) to train, and 1 chunk (20%) to evaluate.
  * **Why no separate Test set?** Because we only have 68 "Fear" images, an 80/20 test split would leave us with just ~13 test images for Fear (which isn't statistically valid). Instead, the 5 evaluation folds combined become our ultimate Test set. By the time all 5 folds are finished, every single image was evaluated as an unseen "Out-of-Fold" (OOF) prediction exactly once. These 1,808 OOF predictions are glued together to calculate our final defensible metrics without needing a separate Test folder.
* **Weighted Random Sampler:** To combat the class imbalance, the training loader samples the rare classes (like Fear and Surprise) much more frequently than common classes (like Joy). This forces the model to pay equal attention to all emotions.
* **RAW Images:** We tested using MTCNN (face cropping) and CLAHE (contrast enhancement), but found they actually hurt accuracy because ASD faces sometimes have unusual poses that confuse crop algorithms. Therefore, we train directly on the RAW images.
* **Careful Learning Rates:** We trained the pretrained backbone layers very slowly (10x slower than the classification head) to avoid losing their learned features.
* **Stable Training:** We used mixed-precision (AMP) for faster GPU training and an Exponential Moving Average (EMA) to help the models generalize better.

## 4. Evaluation & Results

We do not judge these models on basic "Accuracy". In an imbalanced dataset where ~46% of the data is "Joy", a model could just guess "Joy" every time and get 46% accuracy without learning anything.

Instead, we use **Macro F1-Score**, which calculates the performance for each class individually and averages them, treating "Fear" as equally important as "Joy".

**The Key Finding so far:**
VGG-16 proved to be the strongest baseline model (achieving ~72% Accuracy and ~62% Macro F1). Because VGG-16 is heavily texture-focused, it is particularly good at picking up the subtle pixel-level muscle twitches present in ASD facial expressions compared to deeper models like ResNet-50 which can overfit on small datasets.

### Baseline Model Comparison (Summary Table)

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

## 5. Key Charts and Takeaways

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

## 6. Kaggle Training Implementation

We ran the baseline benchmarks on Kaggle GPUs using the standalone script `kaggle/run_all_models.py`. You can view and run the full experimental pipeline in our [Kaggle Notebook](https://www.kaggle.com/code/mrsohel/autism-fer-model). 

* **Resumable:** Since Kaggle sessions often timeout, the script automatically saves its progress after every fold. If it crashes, restarting the script picks up exactly where it left off.
* **Auto-Generated Charts:** Once finished, the script automatically generates all the performance charts (ROC, Radar, etc.) shown in this report.