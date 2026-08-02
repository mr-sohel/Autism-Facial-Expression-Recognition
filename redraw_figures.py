import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize

OUTPUT_DIR = "results/results"
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "paper_figures")
os.makedirs(COMPARISON_DIR, exist_ok=True)
NUM_CLASSES = 6
CLASS_NAMES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]

MODEL_NAME_MAP = {
    "vgg16": "VGG-16",
    "inception_v3": "Inception V3",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenetv2_100": "MobileNetV2",
    "resnet50": "ResNet-50",
    "deit_small_patch16_224": "DeiT Small",
    "vit_base_patch16_224": "ViT Base",
    "swin_base_patch4_window7_224": "Swin Base",
    "swin_tiny_patch4_window7_224": "Swin Tiny"
}

def load_oof(name):
    model_dir = os.path.join(OUTPUT_DIR, name)
    return (np.load(f"{model_dir}/oof_preds.npy"),
            np.load(f"{model_dir}/oof_probs.npy"),
            np.load(f"{model_dir}/oof_labels.npy"))

all_results = {}
for name in MODEL_NAME_MAP.keys():
    metrics_path = f"{OUTPUT_DIR}/{name}/cv_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            all_results[name] = json.load(f)

models_sorted = sorted(all_results.keys(), key=lambda k: all_results[k]["mean"]["f1_macro"], reverse=True)
top_5 = models_sorted[:min(5, len(models_sorted))]

# 1. Grouped bar chart with error bars
rows = []
for m in models_sorted:
    r = all_results[m]["mean"]
    short_name = MODEL_NAME_MAP.get(m, m)
    for metric, key in [("Accuracy", "accuracy"), ("F1-Macro", "f1_macro"),
                        ("Precision", "precision_macro"), ("Recall", "recall_macro")]:
        rows.append({"Model": short_name, "Metric": metric, "Score": r[key],
                     "Std": r[key + "_std"]})
df_metrics = pd.DataFrame(rows)
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_metrics, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax1)
for i, metric in enumerate(["Accuracy", "F1-Macro", "Precision", "Recall"]):
    for j, m in enumerate(models_sorted):
        short_name = MODEL_NAME_MAP.get(m, m)
        row = df_metrics[(df_metrics["Metric"] == metric) & (df_metrics["Model"] == short_name)].iloc[0]
        ax1.errorbar(x=j + (i - 1.5) * 0.2, y=row["Score"], yerr=row["Std"],
                     fmt="none", c="black", capsize=2, linewidth=0.8)
ax1.set_ylim(0, 1.0)
ax1.set_title("Model Comparison (Stratified K-Fold CV) - Mean +/- Std")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/1_cv_grouped_bar_metrics.png", dpi=300)
plt.close(fig1)

# 2. Box plot of fold-level F1 across models
fold_rows = []
for m in models_sorted:
    r = all_results[m]["mean"]
    short_name = MODEL_NAME_MAP.get(m, m)
    fold_rows.append({"Model": short_name, "F1": r["f1_macro"],
                      "lower": r["f1_macro"] - r["f1_macro_std"],
                      "upper": r["f1_macro"] + r["f1_macro_std"]})
df_f1 = pd.DataFrame(fold_rows)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_f1, x="Model", y="F1", palette="coolwarm", ax=ax2)
for i, row in df_f1.iterrows():
    ax2.errorbar(x=i, y=row["F1"], yerr=[[row["F1"] - row["lower"]], [row["upper"] - row["F1"]]],
                 fmt="none", c="black", capsize=3)
ax2.set_ylabel("Macro F1 (mean +/- std across folds)")
ax2.set_ylim(0, 0.8)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/2_cv_f1_comparison.png", dpi=300)
plt.close(fig2)

# 3. OOF macro ROC (top 5)
fig3, ax3 = plt.subplots(figsize=(10, 8))
for m in top_5:
    preds, probs, labels = load_oof(m)
    short_name = MODEL_NAME_MAP.get(m, m)
    Y_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    fpr, tpr, _ = roc_curve(Y_bin.ravel(), probs.ravel())
    macro_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=2, label=f"{short_name} (AUC = {macro_auc:.3f})")
ax3.plot([0, 1], [0, 1], 'k--', lw=2)
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.set_title("Macro-Average OOF ROC Curve (Top 5 Models)")
ax3.legend(loc="lower right"); ax3.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/3_roc_curve.png", dpi=300); plt.close(fig3)

# 4. OOF Precision-Recall (top 5)
fig4, ax4 = plt.subplots(figsize=(10, 8))
for m in top_5:
    preds, probs, labels = load_oof(m)
    short_name = MODEL_NAME_MAP.get(m, m)
    Y_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    prec, rec, _ = precision_recall_curve(Y_bin.ravel(), probs.ravel())
    ap = average_precision_score(Y_bin, probs, average="macro")
    ax4.plot(rec, prec, lw=2, label=f"{short_name} (AP = {ap:.3f})")
ax4.set_xlabel("Recall"); ax4.set_ylabel("Precision")
ax4.set_title("Macro-Average OOF Precision-Recall Curve (Top 5 Models)")
ax4.legend(loc="lower left"); ax4.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/4_pr_curve.png", dpi=300); plt.close(fig4)

# 5. Radar chart (means)
categories = ["Accuracy", "F1-Macro", "Precision", "Recall"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
fig5, ax5 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax5.set_theta_offset(np.pi / 2); ax5.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
plt.ylim(0, 1)
for m in top_5:
    r = all_results[m]["mean"]
    short_name = MODEL_NAME_MAP.get(m, m)
    values = [r["accuracy"], r["f1_macro"], r["precision_macro"], r["recall_macro"]]
    values += values[:1]
    ax5.plot(angles, values, linewidth=2, label=short_name)
    ax5.fill(angles, values, alpha=0.1)
plt.title("Radar Chart - Top 5 Models (CV means)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/5_radar_chart.png", dpi=300, bbox_inches="tight"); plt.close(fig5)

# 6. OOF confusion matrix heatmap for the best model
best_model = models_sorted[0]
short_name = MODEL_NAME_MAP.get(best_model, best_model)
preds, probs, labels = load_oof(best_model)
cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
fig6, ax6 = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax6, vmin=0, vmax=1)
ax6.set_xlabel("Predicted"); ax6.set_ylabel("True")
ax6.set_title(f"{short_name} — Out-of-Fold Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/6_best_model_oof_cm.png", dpi=300); plt.close(fig6)

# 7. Model prediction correlation heatmap (OOF, ensemble diversity)
preds_dict = {}
for m in models_sorted:
    p, _, _ = load_oof(m)
    short_name = MODEL_NAME_MAP.get(m, m)
    preds_dict[short_name] = p
df_preds = pd.DataFrame(preds_dict)
corr = df_preds.corr(method="spearman").fillna(0)
fig7, ax7 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=0, vmax=1, ax=ax7)
ax7.set_title("Model Prediction Correlation (Spearman, OOF)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/7_model_correlation_heatmap.png", dpi=300); plt.close(fig7)
