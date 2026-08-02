# Baseline Model Comparison Report

This report summarizes the performance of 9 baseline models evaluated on the Autism Facial Expression Recognition dataset. The objective of this phase was to determine how well standard convolutional and transformer-based architectures can classify 6 distinct emotions (anger, fear, joy, natural, sadness, surprise) directly from raw images, establishing a benchmark for future architectural improvements.

## Summary Table

| Model | Accuracy | F1-Macro | Precision | Recall |
|---|---|---|---|---|
| **VGG-16** | **0.7201** | **0.6217** | **0.6098** | **0.6725** |
| **Swin Base** | 0.7201 | 0.6100 | 0.5961 | 0.6485 |
| **Inception V3** | 0.7196 | 0.6067 | 0.6030 | 0.6363 |
| **DeiT Small** | 0.7113 | 0.6026 | 0.5895 | 0.6417 |
| **DenseNet-121** | 0.7113 | 0.6001 | 0.5894 | 0.6298 |
| **ViT Base** | 0.7102 | 0.5984 | 0.5880 | 0.6339 |
| **EfficientNet-B0**| 0.6980 | 0.5871 | 0.5813 | 0.6167 |
| **MobileNetV2** | 0.6858 | 0.5790 | 0.5687 | 0.6174 |
| **ResNet-50** | 0.6825 | 0.5762 | 0.5652 | 0.6150 |

*Note: All models were evaluated using Stratified 5-Fold Cross-Validation. The metrics presented represent the average performance across all five folds, ensuring a robust and statistically sound evaluation.*

---

## Key Charts and Takeaways

### 1. Overall Performance
<img src="results/results/paper_figures/1_cv_grouped_bar_metrics.png" alt="Grouped Bar Metrics" width="450"/>

**Takeaway:** VGG-16 and Swin Base achieved the highest overall performance across the evaluated metrics, making them the strongest baseline models.

### 2. Balancing Rare Emotions
<img src="results/results/paper_figures/2_cv_f1_comparison.png" alt="F1 Comparison" width="450"/>

**Takeaway:** The Macro-F1 score indicates how effectively a model manages class imbalance rather than defaulting to majority classes. VGG-16 demonstrates the most consistent ability to balance performance across all emotion categories.

### 3. ROC Curve (Top 5 Models)
<img src="results/results/paper_figures/3_roc_curve.png" alt="ROC Curve" width="450"/>

**Takeaway:** The top 5 models achieve high Area Under the Curve (AUC) scores, demonstrating that the raw, unaugmented image data contains sufficient discriminatory features for accurate classification.

### 4. Precision vs Recall
<img src="results/results/paper_figures/4_pr_curve.png" alt="Precision Recall Curve" width="450"/>

**Takeaway:** The tight clustering among the leading models indicates a strong capacity to maintain high precision without sacrificing recall, which is essential for handling minority classes.

### 5. Radar Chart
<img src="results/results/paper_figures/5_radar_chart.png" alt="Radar Chart" width="450"/>

**Takeaway:** The multi-metric footprint visually confirms that VGG-16 exhibits well-rounded performance, without isolated failures in any single evaluation metric.

### 6. Where the Best Model Struggles
<img src="results/results/paper_figures/6_best_model_oof_cm.png" alt="Best Model Confusion Matrix" width="450"/>

**Takeaway:** The confusion matrix for VGG-16 reveals that the model still occasionally misclassifies "Fear" and "Surprise." This is an expected limitation due to the relative scarcity of training samples for these specific expressions.

### 7. Feature Correlation Analysis
<img src="results/results/paper_figures/7_model_correlation_heatmap.png" alt="Model Correlation Heatmap" width="450"/>

**Takeaway:** The correlation heatmap indicates that CNN- and Transformer-based models capture different aspects of facial features. This observation may help guide the design of an improved architecture in the next phase of the research.

---

## Methodology Overview

To ensure the evaluation was fair and representative of real-world generalization:
1. **Stratified 5-Fold Cross-Validation:** Instead of using a fixed test split, the 1,808 images were merged and divided into 5 folds. The models trained on 4 parts and tested on the remaining 1 part, repeating 5 times. This ensures every image is tested exactly once, producing highly reliable metrics.
2. **Handling Class Imbalance:** During training, minority classes were sampled more frequently so that the model learned all emotion classes more effectively.

---

## Conclusion
* **VGG-16** achieved the best overall performance among all baseline models.
* **Swin Base** produced comparable accuracy, demonstrating the effectiveness of Transformer-based models.
* All baseline models achieved approximately 68-72% accuracy, indicating that further improvements require architectural innovation rather than simply changing the backbone.
* Based on these findings, the next stage of this research is to develop a proposed dual-stream architecture by combining CNN and Transformer features.
