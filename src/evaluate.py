import os
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import CLASS_NAMES


def plot_confusion_matrix(y_true, y_pred, save_path, model_name="Model", normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    else:
        cm_display = cm.astype("float")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display, annot=True, fmt=".2f" if normalize else "d",
        cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        vmin=0, vmax=1 if normalize else None,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"{model_name} - Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_curves(history, save_dir, model_name="Model"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = history["epochs"]
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, "b-", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, val_acc, "r-", label="Val Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_name} - Training & Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_per_class_f1(per_class_f1, save_dir, model_name="Model"):
    os.makedirs(save_dir, exist_ok=True)
    names = list(per_class_f1.keys())
    values = list(per_class_f1.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", len(names))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("F1-Score")
    ax.set_title(f"{model_name} - Per-Class F1-Score")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{model_name}_f1_per_class.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_model_comparison(results_dict, metric, save_dir, title=None):
    os.makedirs(save_dir, exist_ok=True)
    names = list(results_dict.keys())
    values = [results_dict[n].get(metric, 0) for n in names]

    sorted_pairs = sorted(zip(values, names), reverse=True)
    values_sorted = [v for v, n in sorted_pairs]
    names_sorted = [n for v, n in sorted_pairs]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = sns.color_palette("coolwarm", len(names_sorted))
    bars = ax.barh(names_sorted, values_sorted, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(title or f"Model Comparison - {metric.replace('_', ' ').title()}")
    for bar, val in zip(bars, values_sorted):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(save_dir, f"comparison_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def generate_all_comparison_plots(all_results, results_dir):
    comparison_dir = os.path.join(results_dir, "comparisons")
    os.makedirs(comparison_dir, exist_ok=True)

    metrics_to_plot = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    for metric in metrics_to_plot:
        plot_model_comparison(all_results, metric, comparison_dir,
                              title=f"Model Comparison - {metric.replace('_', ' ').title()}")

    # Summary table
    summary_path = os.path.join(comparison_dir, "summary_table.txt")
    with open(summary_path, "w") as f:
        header = f"{'Model':<35} {'Accuracy':>10} {'F1-Macro':>10} {'Precision':>10} {'Recall':>10}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for name, res in sorted(all_results.items(), key=lambda x: x[1].get("f1_macro", 0), reverse=True):
            line = f"{name:<35} {res.get('accuracy', 0):>10.4f} {res.get('f1_macro', 0):>10.4f} {res.get('precision_macro', 0):>10.4f} {res.get('recall_macro', 0):>10.4f}"
            f.write(line + "\n")

    # Save as JSON too
    json_path = os.path.join(comparison_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return comparison_dir, summary_path
