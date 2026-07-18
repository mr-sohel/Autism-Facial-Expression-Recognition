"""
=============================================================
  Autism Facial Expression Recognition — Full Kaggle Pipeline
  Train 14 models on free Kaggle GPU (T4/P100)
=============================================================
"""

# %% [markdown]
# # Setup & Installs

# %%
import os, sys, json, time, copy
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from tqdm.auto import tqdm

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
    print("WARNING: No GPU detected — training will be very slow")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# %% [markdown]
# # Configuration

# %%
# ---- Paths (Kaggle default) ----
DATA_DIR = "/kaggle/input/datasets/mrsohel/autism-fer-dataset/dataset"  # <-- CHANGE THIS
OUTPUT_DIR = "/kaggle/working/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Hyperparameters ----
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
MIXUP_ALPHA = 0.4
EMA_DECAY = 0.999
NUM_WORKERS = 2

CLASS_NAMES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---- Models to train ----
EXPERIMENTS = [
    # CNNs (Thrive with focal loss and higher LR)
    {"model": "vgg16",                        "loss": "focal",     "lr": 1e-3},
    {"model": "vgg19",                        "loss": "focal",     "lr": 1e-3},
    {"model": "mobilenetv2_100",              "loss": "focal",     "lr": 1e-3},
    {"model": "mobilenetv3_large_100",        "loss": "focal",     "lr": 1e-3},
    {"model": "inception_v3",                 "loss": "focal",     "lr": 1e-3},
    {"model": "tf_efficientnetv2_s",          "loss": "focal",     "lr": 1e-3},
    {"model": "tf_efficientnetv2_m",          "loss": "ce_smooth", "lr": 5e-4},  # Regressed on focal/1e-3
    {"model": "resnet18",                     "loss": "focal",     "lr": 1e-3},
    {"model": "resnet50",                     "loss": "focal",     "lr": 1e-3},
    {"model": "seresnet50",                   "loss": "focal",     "lr": 1e-3},
    {"model": "densenet121",                  "loss": "focal",     "lr": 1e-3},
    {"model": "ghostnet_100",                 "loss": "focal",     "lr": 1e-3},
    
    # Transformers & Hybrids (Need lower LR to prevent collapse)
    {"model": "convnext_small",               "loss": "ce_smooth", "lr": 3e-4},
    {"model": "vit_tiny_patch16_224",         "loss": "ce_smooth", "lr": 1e-4},
    {"model": "vit_base_patch16_224",         "loss": "ce_smooth", "lr": 1e-4},
    {"model": "deit_small_patch16_224",       "loss": "ce_smooth", "lr": 1e-4},
    {"model": "swin_base_patch4_window7_224", "loss": "ce_smooth", "lr": 1e-4},
    {"model": "mobilevit_s",                  "loss": "ce_smooth", "lr": 3e-4},
    {"model": "coat_lite_small",              "loss": "ce_smooth", "lr": 3e-4},
    {"model": "crossvit_9_240",               "loss": "ce_smooth", "lr": 1e-4},  # Worked well at 1e-4 in Run 1
]


# %% [markdown]
# # Dataset

# %%
class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            class_idx = CLASS_TO_IDX[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    self.samples.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_class_weights(dataset):
    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    return torch.FloatTensor([total / (NUM_CLASSES * counts.get(i, 1)) for i in range(NUM_CLASSES)])


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    train_ds = FacialExpressionDataset(os.path.join(data_dir, "train"), get_train_transforms(img_size))
    val_ds = FacialExpressionDataset(os.path.join(data_dir, "valid"), get_val_transforms(img_size))
    test_ds = FacialExpressionDataset(os.path.join(data_dir, "test"), get_val_transforms(img_size))

    # Weighted sampler for class imbalance
    counts = Counter(train_ds.labels)
    sample_weights = [1.0 / counts[label] for label in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    class_weights = compute_class_weights(train_ds)
    return train_loader, val_loader, test_loader, class_weights, train_ds


# %%
print("Loading datasets...")
train_loader, val_loader, test_loader, class_weights, train_ds = get_dataloaders(DATA_DIR)
class_weights = class_weights.to(DEVICE)
print(f"Train: {len(train_ds)} images")
print(f"Class weights: {dict(zip(CLASS_NAMES, [f'{w:.2f}' for w in class_weights.cpu()]))}")


# %% [markdown]
# # Losses

# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def get_loss_fn(loss_type, class_weights=None):
    if loss_type == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif loss_type == "ce_smooth":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)


# %% [markdown]
# # Models

# %%
MODEL_CONFIGS = {
    "vgg16": {"timm": "vgg16_bn", "size": 224},
    "vgg19": {"timm": "vgg19_bn", "size": 224},
    "mobilenetv2_100": {"timm": "mobilenetv2_100", "size": 224},
    "mobilenetv3_large_100": {"timm": "mobilenetv3_large_100", "size": 224},
    "inception_v3": {"timm": "inception_v3", "size": 299},
    "tf_efficientnetv2_s": {"timm": "tf_efficientnetv2_s", "size": 224},
    "tf_efficientnetv2_m": {"timm": "tf_efficientnetv2_m", "size": 224},
    "resnet18": {"timm": "resnet18", "size": 224},
    "resnet50": {"timm": "resnet50", "size": 224},
    "seresnet50": {"timm": "seresnet50", "size": 224},
    "densenet121": {"timm": "densenet121", "size": 224},
    "convnext_small": {"timm": "convnext_small.fb_in22k", "size": 224},
    "ghostnet_100": {"timm": "ghostnet_100", "size": 224},
    "vit_tiny_patch16_224": {"timm": "vit_tiny_patch16_224.augreg_in21k", "size": 224},
    "vit_base_patch16_224": {"timm": "vit_base_patch16_224.augreg_in21k", "size": 224},
    "deit_small_patch16_224": {"timm": "deit_small_patch16_224", "size": 224},
    "swin_base_patch4_window7_224": {"timm": "swin_base_patch4_window7_224", "size": 224},
    "mobilevit_s": {"timm": "mobilevit_s", "size": 256},
    "coat_lite_small": {"timm": "coat_lite_small", "size": 224},
    "crossvit_9_240": {"timm": "crossvit_9_240", "size": 240},
}


def get_model(name, pretrained=True):
    cfg = MODEL_CONFIGS[name]
    try:
        try:
            model = timm.create_model(cfg["timm"], pretrained=pretrained, num_classes=NUM_CLASSES,
                                      drop_rate=0.3, drop_path_rate=0.2)
        except TypeError:
            model = timm.create_model(cfg["timm"], pretrained=pretrained, num_classes=NUM_CLASSES)
    except RuntimeError as e:
        if pretrained and "pretrained" in str(e).lower():
            print(f"Warning: No pretrained weights for {name}. Using random init.")
            return get_model(name, pretrained=False)
        raise
    return model, cfg["size"]


# %% [markdown]
# # EMA & MixUp

# %%
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self):
        self.backup = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[n]

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# %% [markdown]
# # Metrics

# %%
def compute_metrics(y_true, y_pred):
    labels_list = list(range(NUM_CLASSES))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_list),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_list),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_list),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_list).tolist(),
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, labels=labels_list),
        "per_class_f1": dict(zip(CLASS_NAMES, [float(f) for f in f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels_list)])),
    }


# %% [markdown]
# # Training Loop

# %%
def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, use_mixup):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16):
            if use_mixup:
                mixed, ya, yb, lam = mixup_data(images, labels)
                outputs = model(mixed)
                loss = lam * criterion(outputs, ya) + (1 - lam) * criterion(outputs, yb)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update()

        if not use_mixup:
            correct += outputs.argmax(1).eq(labels).sum().item()
        else:
            correct += (lam * outputs.argmax(1).eq(ya).sum().item() + (1 - lam) * outputs.argmax(1).eq(yb).sum().item())
        total += labels.size(0)
        total_loss += loss.item() * images.size(0)

    return total_loss / total, correct / total if total else 0


@torch.no_grad()
def evaluate(model, loader, criterion, ema=None):
    if ema:
        ema.apply_shadow()
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    if ema:
        ema.restore()
    return total_loss / len(all_preds), [int(p) for p in all_preds], [int(l) for l in all_labels], np.array(all_probs)


# %% [markdown]
# # Plots

# %%
def plot_confusion_matrix(y_true, y_pred, path, name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion Matrix")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_curves(history, path, name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["epochs"], history["train_loss"], "b-", label="Train", lw=2)
    axes[0].plot(history["epochs"], history["val_loss"], "r-", label="Val", lw=2)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title=f"{name} — Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["epochs"], history["train_acc"], "b-", label="Train", lw=2)
    axes[1].plot(history["epochs"], history["val_acc"], "r-", label="Val", lw=2)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title=f"{name} — Accuracy")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_f1_bars(per_class_f1, path, name):
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(per_class_f1.keys(), per_class_f1.values(), color=sns.color_palette("viridis", len(per_class_f1)),
                  edgecolor="black", lw=0.5)
    ax.set_ylabel("F1-Score"); ax.set_title(f"{name} — Per-Class F1"); ax.set_ylim(0, 1.0)
    for b, v in zip(bars, per_class_f1.values()):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# %% [markdown]
# # Run All Experiments

# %%
all_results = {}
global_test_labels = None

for exp in EXPERIMENTS:
    name = exp["model"]
    loss_type = exp["loss"]

    print(f"\n{'='*60}")
    print(f"  TRAINING: {name} | Loss: {loss_type}")
    print(f"{'='*60}")

    model, input_size = get_model(name, pretrained=True)
    model = model.to(DEVICE)

    # Rebuild dataloaders (transforms are recreated each time for safety)
    train_loader, val_loader, test_loader, class_weights, _ = get_dataloaders(DATA_DIR, img_size=input_size)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    criterion = get_loss_fn(loss_type, None)

    # Differential learning rates
    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in n for k in ("classifier", "head", "fc")):
            head.append(p)
        else:
            backbone.append(p)

    exp_lr = exp.get("lr", LEARNING_RATE)
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": exp_lr * 0.1},
        {"params": head, "lr": exp_lr},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=(DEVICE.type == "cuda"))
    ema = EMA(model, EMA_DECAY)

    history = {"epochs": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_f1 = 0.0
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema, True)
        scheduler.step()
        val_loss, val_preds, val_labels, _ = evaluate(model, val_loader, criterion, ema)
        val_m = compute_metrics(val_labels, val_preds)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_m["accuracy"])

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{NUM_EPOCHS} | "
            f"TrL {train_loss:.4f} TrA {train_acc:.4f} | "
            f"VlL {val_loss:.4f} VaA {val_m['accuracy']:.4f} F1 {val_m['f1_macro']:.4f} | "
            f"{elapsed/60:.1f}min"
        )

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            patience_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "ema": ema.shadow, "args": exp}, f"{OUTPUT_DIR}/{name}_best.pth")
            print(f"    >> New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    >> Early stopping at epoch {epoch}")
                break

    # --- Test evaluation ---
    ckpt = torch.load(f"{OUTPUT_DIR}/{name}_best.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ema.shadow = ckpt["ema"]

    test_loss, test_preds, test_labels, test_probs = evaluate(model, test_loader, criterion, ema)
    test_m = compute_metrics(test_labels, test_preds)

    print(f"\n  TEST — Acc: {test_m['accuracy']:.4f} | F1: {test_m['f1_macro']:.4f} | Prec: {test_m['precision_macro']:.4f} | Rec: {test_m['recall_macro']:.4f}")
    print(test_m["report"])

    # Save everything
    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)
    with open(f"{model_dir}/test_metrics.json", "w") as f:
        json.dump(test_m, f, indent=2)
    with open(f"{model_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_confusion_matrix(test_labels, test_preds, f"{model_dir}/confusion_matrix.png", name)
    plot_curves(history, f"{model_dir}/training_curves.png", name)
    plot_f1_bars(test_m["per_class_f1"], f"{model_dir}/f1_per_class.png", name)

    all_results[name] = test_m
    all_results[name]["params"] = total_p
    all_results[name]["time_min"] = (time.time() - t0) / 60
    all_results[name]["test_probs"] = test_probs
    all_results[name]["test_preds"] = test_preds
    global_test_labels = test_labels

    print(f"  Saved to {model_dir}/")
    torch.cuda.empty_cache()

    # --- Explicit Cleanup ---
    # Prevents "Too many open files" and DataLoader worker process leaks across experiments on Kaggle
    del model, optimizer, scheduler, scaler, criterion, ema
    del train_loader, val_loader, test_loader
    import gc
    gc.collect()


# %% [markdown]
# # Final Comparison & Paper Figures

# %%
print(f"\n{'='*70}")
print("  FINAL MODEL COMPARISON")
print(f"{'='*70}")

models_sorted = sorted(all_results.keys(), key=lambda k: all_results[k]["f1_macro"], reverse=True)
top_5_models = models_sorted[:5]

header = f"{'Model':<35} {'Acc':>8} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Params':>12}"
print(header)
print("-" * len(header))
for name in models_sorted:
    r = all_results[name]
    print(f"{name:<35} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} {r['precision_macro']:>8.4f} {r['recall_macro']:>8.4f} {r.get('params',0):>12,}")

# Remove heavy arrays before saving JSON
json_results = copy.deepcopy(all_results)
for k in json_results:
    json_results[k].pop("test_probs", None)
    json_results[k].pop("test_preds", None)
with open(f"{OUTPUT_DIR}/comparison.json", "w") as f:
    json.dump(json_results, f, indent=2)

COMPARISON_DIR = os.path.join(OUTPUT_DIR, "paper_figures")
os.makedirs(COMPARISON_DIR, exist_ok=True)

# 1. Grouped Bar Chart (Acc, F1, Prec, Rec) for Top Models
df_metrics = []
for m in top_5_models:
    df_metrics.extend([
        {"Model": m, "Metric": "Accuracy", "Score": all_results[m]["accuracy"]},
        {"Model": m, "Metric": "F1-Macro", "Score": all_results[m]["f1_macro"]},
        {"Model": m, "Metric": "Precision", "Score": all_results[m]["precision_macro"]},
        {"Model": m, "Metric": "Recall", "Score": all_results[m]["recall_macro"]},
    ])
df_metrics = pd.DataFrame(df_metrics)
fig1 = plt.figure(figsize=(12, 6))
sns.barplot(data=df_metrics, x="Model", y="Score", hue="Metric", palette="Set2")
plt.ylim(0, 1.0)
plt.title("Top 5 Models - Performance Metrics")
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/1_grouped_bar_metrics.png", dpi=300)
plt.close(fig1)

# 2. Box Plot of Per-Class F1 Scores
f1_data = []
for m in models_sorted:
    for cls_name, f1 in all_results[m]["per_class_f1"].items():
        f1_data.append({"Model": m, "Class": cls_name, "F1": f1})
df_f1 = pd.DataFrame(f1_data)
fig2 = plt.figure(figsize=(14, 8))
sns.boxplot(data=df_f1, x="F1", y="Model", palette="coolwarm")
plt.title("Distribution of Per-Class F1 Scores across Models")
plt.xlabel("F1-Score")
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/2_boxplot_f1_distributions.png", dpi=300)
plt.close(fig2)

# 3. Macro ROC Curve (Top 5 Models)
fig3 = plt.figure(figsize=(10, 8))
Y_test_bin = label_binarize(global_test_labels, classes=list(range(NUM_CLASSES)))
for m in top_5_models:
    probs = all_results[m]["test_probs"]
    fpr, tpr, _ = roc_curve(Y_test_bin.ravel(), probs.ravel())
    macro_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f"{m} (AUC = {macro_auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Macro-Average ROC Curve (Top 5 Models)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(f"{COMPARISON_DIR}/3_roc_curve.png", dpi=300)
plt.close(fig3)

# 4. Precision-Recall Curve (Top 5 Models)
fig4 = plt.figure(figsize=(10, 8))
for m in top_5_models:
    probs = all_results[m]["test_probs"]
    prec, rec, _ = precision_recall_curve(Y_test_bin.ravel(), probs.ravel())
    ap = average_precision_score(Y_test_bin, probs, average="macro")
    plt.plot(rec, prec, lw=2, label=f"{m} (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Macro-Average Precision-Recall Curve (Top 5 Models)")
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.savefig(f"{COMPARISON_DIR}/4_pr_curve.png", dpi=300)
plt.close(fig4)

# 5. Radar Chart (Spider Chart)
categories = ['Accuracy', 'F1-Macro', 'Precision', 'Recall']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
fig5, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], categories)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.ylim(0, 1)

for m in top_5_models:
    values = [
        all_results[m]["accuracy"],
        all_results[m]["f1_macro"],
        all_results[m]["precision_macro"],
        all_results[m]["recall_macro"]
    ]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=m)
    ax.fill(angles, values, alpha=0.1)

plt.title("Radar Chart - Top 5 Models", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.savefig(f"{COMPARISON_DIR}/5_radar_chart.png", dpi=300, bbox_inches="tight")
plt.close(fig5)

# 6. Model Prediction Correlation Heatmap (Ensemble Diversity)
preds_dict = {m: all_results[m]["test_preds"] for m in models_sorted}
df_preds = pd.DataFrame(preds_dict)
corr = df_preds.corr(method="spearman").fillna(0)
fig6 = plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Model Prediction Correlation (Spearman)")
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/6_model_correlation_heatmap.png", dpi=300)
plt.close(fig6)

print(f"\nAll results and paper-ready figures saved to {COMPARISON_DIR}/")
print("Download the output dataset from the Kaggle 'Output' tab.")
