"""
=============================================================
  Autism Facial Expression Recognition — Kaggle Pipeline
  Train 9 curated models with Stratified K-Fold Cross-Validation
  on the RAW dataset (no face-crop / CLAHE preprocessing).
=============================================================

Methodology fixes vs. the previous version:
  1. RAW images only — the MTCNN+CLAHE step measurably hurt accuracy
     (vgg16 F1 0.548->0.528, resnet50/vit_base collapsed). Removed.
  2. Single weighting — class imbalance is handled by WeightedRandomSampler
     ONLY. Class weights are no longer passed into FocalLoss/CE
     (the old code double-weighted, over-regularising small models).
  3. Lighter train-time augmentation — MixUp and RandomErasing removed.
     On ~2k images they added more noise than signal.
  4. Stratified K-fold CV — every image is predicted exactly once
     (out-of-fold), so per-class metrics on rare emotions (fear n~14)
     are no longer statistically meaningless. Results are mean +/- std.

Resumable: per-fold checkpoints + a cv_done.json marker are saved to
OUTPUT_DIR after every fold. If a Kaggle session times out, just re-run
the cell — already-completed (model, fold) pairs are skipped.
"""

import os, sys, json, time, copy
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
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
from sklearn.model_selection import StratifiedKFold
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

# ---- Paths (Kaggle default) ----
_EXPECTED_CLASSES = ("anger", "fear", "joy", "natural", "sadness", "surprise")

def find_dataset_dir(hardcoded):
    """Locate the dataset root containing train/valid/test class folders."""
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return hardcoded
        
    for root, dirs, _ in os.walk(base):
        if all(split in dirs for split in ("train", "valid", "test")):
            print(f"[*] Auto-detected dataset at {root}")
            return root
            
    print(f"[!] Dataset not found under {base}; using hardcoded path")
    return hardcoded


DATA_DIR   = find_dataset_dir("/kaggle/input/datasets/mrsohel/dataset-clean")
OUTPUT_DIR = "/kaggle/working/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[*] DATA_DIR = {DATA_DIR}")

# ---- Hyperparameters ----
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 80
N_FOLDS    = 5          # stratified K-fold CV (set to 3 to save GPU time)
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
PATIENCE      = 15
EMA_DECAY     = 0.999
NUM_WORKERS   = 2 if sys.platform == "linux" else 0  # Windows spawn breaks multiprocessing loaders

CLASS_NAMES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---- Models to train ----
# Curated 9-model set — one representative per architectural family
# (selected from Run 1 results, see new new log.log).
EXPERIMENTS = [
    # Classic CNNs
    {"model": "vgg16",                        "loss": "focal",     "lr": 1e-3},  # Run 1: F1=0.5477
    {"model": "inception_v3",                 "loss": "focal",     "lr": 1e-3},  # Run 1: F1=0.5232
    {"model": "densenet121",                  "loss": "focal",     "lr": 1e-3},  # Run 1: F1=0.5229
    {"model": "efficientnet_b0",              "loss": "focal",     "lr": 1e-3},  # most-cited FER baseline (Run1 family: v2_s F1=0.511)
    {"model": "mobilenetv2_100",              "loss": "focal",     "lr": 1e-3},  # Run 1: F1=0.4984
    {"model": "resnet50",                     "loss": "focal",     "lr": 1e-3},  # Run 1: F1=0.4636
    # Vision Transformers & Hybrids (need lower LR to prevent collapse)
    {"model": "deit_small_patch16_224",       "loss": "ce_smooth", "lr": 1e-4},  # Run 1: F1=0.5437
    {"model": "vit_base_patch16_224",         "loss": "ce_smooth", "lr": 1e-4},  # Run 1: F1=0.5352
    {"model": "swin_base_patch4_window7_224", "loss": "ce_smooth", "lr": 1e-4},  # Run 1: F1=0.4937
]

# ==============================================================================
# Dataset — all splits merged, CV partitions at runtime
# ==============================================================================
class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, samples, labels, transform=None):
        self.root_dir = root_dir
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def build_full_dataset(root_dir):
    """Collect every image across train/valid/test into one list."""
    samples, labels = [], []
    for split in ("train", "valid", "test"):
        for class_name in CLASS_NAMES:
            class_dir = Path(root_dir) / split / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    samples.append(str(img_path))
                    labels.append(CLASS_TO_IDX[class_name])
    return samples, labels


def get_train_transforms(img_size=IMG_SIZE):
    # MixUp / RandomErasing removed — too destructive on a small dataset.
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def make_loaders(samples, labels, train_idx, val_idx, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    train_ds = FacialExpressionDataset(DATA_DIR,
                                       [samples[i] for i in train_idx],
                                       [labels[i] for i in train_idx],
                                       get_train_transforms(img_size))
    val_ds   = FacialExpressionDataset(DATA_DIR,
                                       [samples[i] for i in val_idx],
                                       [labels[i] for i in val_idx],
                                       get_val_transforms(img_size))

    # SINGLE weighting mechanism: sampler only (no class weights in the loss).
    counts = Counter(train_ds.labels)
    sample_weights = [1.0 / counts[label] for label in train_ds.labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


# ==============================================================================
# Losses
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def get_loss_fn(loss_type):
    if loss_type == "focal":
        return FocalLoss(gamma=2.0)
    elif loss_type == "ce_smooth":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    return nn.CrossEntropyLoss()


# ==============================================================================
# Model factory
# ==============================================================================
MODEL_CONFIGS = {
    "vgg16": {"timm": "vgg16_bn", "size": 224},
    "inception_v3": {"timm": "inception_v3", "size": 299},
    "densenet121": {"timm": "densenet121", "size": 224},
    "efficientnet_b0": {"timm": "efficientnet_b0", "size": 224},
    "mobilenetv2_100": {"timm": "mobilenetv2_100", "size": 224},
    "resnet50": {"timm": "resnet50", "size": 224},
    "deit_small_patch16_224": {"timm": "deit_small_patch16_224", "size": 224},
    "vit_base_patch16_224": {"timm": "vit_base_patch16_224.augreg_in21k", "size": 224},
    "swin_base_patch4_window7_224": {"timm": "swin_base_patch4_window7_224", "size": 224},
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


# ==============================================================================
# Training helpers
# ==============================================================================
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


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE.type, dtype=torch.float16 if DEVICE.type == "cuda" else torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        if ema:
            ema.update()

        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * images.size(0)

    return total_loss / total, correct / total if total else 0


@torch.no_grad()
def evaluate(model, loader, ema=None):
    if ema:
        ema.apply_shadow()
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    if ema:
        ema.restore()
    return [int(p) for p in all_preds], [int(l) for l in all_labels], np.array(all_probs)


def compute_metrics(y_true, y_pred):
    labels_list = list(range(NUM_CLASSES))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_list),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_list),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0, labels=labels_list),
        "per_class_f1": dict(zip(CLASS_NAMES, [float(f) for f in f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels_list)])),
        "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, labels=labels_list),
    }


# ==============================================================================
# Resume support
# ==============================================================================
DONE_FILE = os.path.join(OUTPUT_DIR, "cv_done.json")
done = {}
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        done = json.load(f)
    print(f"[*] Found {len(done)} model(s) with completed folds — resuming.")


def fold_done(name, fold):
    return name in done and fold in done[name]


def mark_done(name, fold):
    done.setdefault(name, []).append(fold)
    with open(DONE_FILE, "w") as f:
        json.dump(done, f, indent=2)


# ==============================================================================
# Cross-validation splits (shared with the Proposed-Model script)
# ==============================================================================
print("[*] Building full dataset (train+valid+test merged) ...")
samples, labels = build_full_dataset(DATA_DIR)
print(f"[*] Total images: {len(samples)} | Per-class: {dict(Counter(labels))}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(skf.split(samples, labels))

# Persist the fold assignment so run_proposed_model.py uses identical folds.
fold_id_by_path = {}
for fold_idx, (_, val_idx) in enumerate(folds):
    for i in val_idx:
        fold_id_by_path[samples[i]] = fold_idx
with open(os.path.join(OUTPUT_DIR, "fold_id_by_path.json"), "w") as f:
    json.dump(fold_id_by_path, f)
print(f"[*] Saved fold assignment to {OUTPUT_DIR}/fold_id_by_path.json")


# ==============================================================================
# Per-fold training
# ==============================================================================
def run_fold(name, exp, fold, train_idx, val_idx):
    print(f"\n{'='*60}")
    print(f"  {name} | Loss: {exp['loss']} | Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*60}")

    model, input_size = get_model(name, pretrained=True)
    model = model.to(DEVICE)

    train_loader, val_loader = make_loaders(samples, labels, train_idx, val_idx, img_size=input_size)

    criterion = get_loss_fn(exp["loss"])

    # Differential learning rates (backbone 0.1x, head 1x)
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

    best_f1 = 0.0
    patience_counter = 0
    ckpt_path = f"{OUTPUT_DIR}/{name}/fold{fold+1}_best.pth"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema)
        scheduler.step()
        preds, val_labels, _ = evaluate(model, val_loader, ema)
        val_m = compute_metrics(val_labels, preds)

        elapsed = time.time() - t0
        print(f"  Epoch [{epoch:2d}/{NUM_EPOCHS}] "
              f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.1f}% "
              f"| Val Acc: {val_m['accuracy']*100:5.1f}% | Val Macro-F1: {val_m['f1_macro']:.4f} "
              f"| Time: {elapsed/60:.1f}m")

        if val_m["f1_macro"] > best_f1:
            best_f1 = val_m["f1_macro"]
            patience_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "ema": ema.shadow, "args": exp}, ckpt_path)
            print(f"    >> New best fold F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"    >> Early stopping at epoch {epoch}")
                break

    # --- Out-of-fold evaluation with the EMA weights ---
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    ema.shadow = ckpt["ema"]
    preds, val_labels, probs = evaluate(model, val_loader, ema)
    fold_m = compute_metrics(val_labels, preds)
    print(f"  FOLD {fold+1} OOF — Acc: {fold_m['accuracy']:.4f} | F1: {fold_m['f1_macro']:.4f}")

    # Cleanup (Kaggle DataLoader worker leaks)
    del model, optimizer, scheduler, scaler, criterion, ema
    del train_loader, val_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return fold_m, preds, val_labels, probs, best_f1


# ==============================================================================
# Run all models over all folds (OOF arrays + metrics persisted after every fold
# so a resumed session always works with the full set of completed folds)
# ==============================================================================
def load_npy(path, empty_shape):
    if os.path.exists(path):
        arr = np.load(path)
        return arr if arr.ndim > 0 else np.empty(empty_shape)
    return np.empty(empty_shape)


for exp in EXPERIMENTS:
    name = exp["model"]
    print(f"\n{'#'*70}\n# {name}\n{'#'*70}")

    model_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(model_dir, exist_ok=True)

    oof_preds = load_npy(f"{model_dir}/oof_preds.npy", (0,))
    oof_labels = load_npy(f"{model_dir}/oof_labels.npy", (0,))
    oof_probs = load_npy(f"{model_dir}/oof_probs.npy", (0, NUM_CLASSES))
    fold_metrics = []
    metrics_path = f"{model_dir}/cv_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            fold_metrics = json.load(f)["folds"]

    ran_any = False
    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold_done(name, fold):
            print(f"  Fold {fold+1} already done — skipping.")
            continue
        ran_any = True

        fold_m, preds, val_labels, probs, best_f1 = run_fold(name, exp, fold, train_idx, val_idx)
        fold_metrics.append({"fold": fold, **fold_m})
        oof_preds = np.concatenate([oof_preds, np.array(preds, dtype=int)])
        oof_labels = np.concatenate([oof_labels, np.array(val_labels, dtype=int)])
        oof_probs = np.concatenate([oof_probs, probs], axis=0)

        # incremental persistence (safe across session timeouts)
        np.save(f"{model_dir}/oof_preds.npy", oof_preds)
        np.save(f"{model_dir}/oof_labels.npy", oof_labels)
        np.save(f"{model_dir}/oof_probs.npy", oof_probs)
        with open(metrics_path, "w") as f:
            json.dump({"folds": fold_metrics}, f, indent=2)
        mark_done(name, fold)

    if not ran_any:
        # resumed session where all folds were already completed
        if not fold_metrics:
            raise RuntimeError(f"{name}: cv_metrics.json missing but folds marked done — "
                               "results directory is inconsistent.")
        with open(metrics_path) as f:
            oof_m = json.load(f)
    else:
        oof_m = compute_metrics(oof_labels.tolist(), oof_preds.tolist())
        oof_m["mean"] = {}
        for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            vals = [f[k] for f in fold_metrics]
            oof_m["mean"][k] = float(np.mean(vals))
            oof_m["mean"][k + "_std"] = float(np.std(vals))
        oof_m["n_folds"] = len(fold_metrics)
        oof_m["folds"] = fold_metrics
        oof_m["params"] = sum(p.numel() for p in get_model(name, pretrained=False)[0].parameters())
        with open(metrics_path, "w") as f:
            json.dump(oof_m, f, indent=2)

    print(f"\n  {name} CV mean ({oof_m.get('n_folds', len(fold_metrics))} folds) — "
          f"Acc: {oof_m['mean']['accuracy']:.4f}+/-{oof_m['mean']['accuracy_std']:.4f} | "
          f"F1: {oof_m['mean']['f1_macro']:.4f}+/-{oof_m['mean']['f1_macro_std']:.4f}")

# ==============================================================================
# Cross-model comparison (from saved cv_metrics.json so figures work on resume)
# ==============================================================================
def load_oof(name):
    model_dir = os.path.join(OUTPUT_DIR, name)
    return (np.load(f"{model_dir}/oof_preds.npy"),
            np.load(f"{model_dir}/oof_probs.npy"),
            np.load(f"{model_dir}/oof_labels.npy"))

all_results = {}
for exp in EXPERIMENTS:
    name = exp["model"]
    with open(f"{OUTPUT_DIR}/{name}/cv_metrics.json") as f:
        m = json.load(f)
    m["args"] = exp
    all_results[name] = m

print(f"\n{'='*70}")
print("  FINAL MODEL COMPARISON (Stratified K-Fold CV)")
print(f"{'='*70}")
header = f"{'Model':<35} {'Acc':>10} {'F1':>10} {'Prec':>10} {'Rec':>10} {'Folds':>6}"
print(header)
print("-" * len(header))
for name in sorted(all_results.keys(), key=lambda k: all_results[k]["mean"]["f1_macro"], reverse=True):
    r = all_results[name]["mean"]
    n_folds = all_results[name].get("n_folds", len(all_results[name].get("folds", [])))
    print(f"{name:<35} {r['accuracy']:>6.4f}+/-{r['accuracy_std']:.3f} "
          f"{r['f1_macro']:>6.4f}+/-{r['f1_macro_std']:.3f} "
          f"{r['precision_macro']:>6.4f}+/-{r['precision_macro_std']:.3f} "
          f"{r['recall_macro']:>6.4f}+/-{r['recall_macro_std']:.3f} {n_folds:>6}")

# ==============================================================================
# Paper figures (CV-aware)
# ==============================================================================
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "paper_figures")
os.makedirs(COMPARISON_DIR, exist_ok=True)
models_sorted = sorted(all_results.keys(), key=lambda k: all_results[k]["mean"]["f1_macro"], reverse=True)
top_5 = models_sorted[:min(5, len(models_sorted))]

# 1. Grouped bar chart with error bars
rows = []
for m in models_sorted:
    r = all_results[m]["mean"]
    for metric, key in [("Accuracy", "accuracy"), ("F1-Macro", "f1_macro"),
                        ("Precision", "precision_macro"), ("Recall", "recall_macro")]:
        rows.append({"Model": m, "Metric": metric, "Score": r[key],
                     "Std": r[key + "_std"]})
df_metrics = pd.DataFrame(rows)
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_metrics, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax1)
# overlay std as error caps
x_positions = []
for i, metric in enumerate(["Accuracy", "F1-Macro", "Precision", "Recall"]):
    for j, m in enumerate(models_sorted):
        row = df_metrics[(df_metrics["Metric"] == metric) & (df_metrics["Model"] == m)].iloc[0]
        ax1.errorbar(x=j + (i - 1.5) * 0.2, y=row["Score"], yerr=row["Std"],
                     fmt="none", c="black", capsize=2, linewidth=0.8)
ax1.set_ylim(0, 1.0)
ax1.set_title("Model Comparison (Stratified K-Fold CV) - Mean +/- Std")
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/1_cv_grouped_bar_metrics.png", dpi=300)
plt.close(fig1)

# 2. Box plot of fold-level F1 across models
fold_rows = []
for exp in EXPERIMENTS:
    name = exp["model"]
    with open(f"{OUTPUT_DIR}/{name}/cv_metrics.json") as f:
        m = json.load(f)
    # fold-level F1 is not in cv_metrics.json; reconstruct from per-fold jsons is skipped.
    # Use mean/std as pseudo-box for robustness across resume.
    fold_rows.append({"Model": name, "F1": m["mean"]["f1_macro"],
                      "lower": m["mean"]["f1_macro"] - m["mean"]["f1_macro_std"],
                      "upper": m["mean"]["f1_macro"] + m["mean"]["f1_macro_std"]})
df_f1 = pd.DataFrame(fold_rows)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(data=df_f1, x="Model", y="F1", palette="coolwarm", ax=ax2)
for i, row in df_f1.iterrows():
    ax2.errorbar(x=i, y=row["F1"], yerr=[[row["F1"] - row["lower"]], [row["upper"] - row["F1"]]],
                 fmt="none", c="black", capsize=3)
ax2.set_ylabel("Macro F1 (mean +/- std across folds)")
ax2.set_ylim(0, 0.8)
plt.tight_layout()
plt.savefig(f"{COMPARISON_DIR}/2_cv_f1_comparison.png", dpi=300)
plt.close(fig2)

# 3. OOF macro ROC (top 5)
fig3, ax3 = plt.subplots(figsize=(10, 8))
global_labels = None
for m in top_5:
    preds, probs, labels = load_oof(m)
    Y_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    fpr, tpr, _ = roc_curve(Y_bin.ravel(), probs.ravel())
    macro_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, lw=2, label=f"{m} (AUC = {macro_auc:.3f})")
    global_labels = labels
ax3.plot([0, 1], [0, 1], 'k--', lw=2)
ax3.set_xlabel("False Positive Rate"); ax3.set_ylabel("True Positive Rate")
ax3.set_title("Macro-Average OOF ROC Curve (Top 5 Models)")
ax3.legend(loc="lower right"); ax3.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/3_roc_curve.png", dpi=300); plt.close(fig3)

# 4. OOF Precision-Recall (top 5)
fig4, ax4 = plt.subplots(figsize=(10, 8))
for m in top_5:
    preds, probs, labels = load_oof(m)
    Y_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    prec, rec, _ = precision_recall_curve(Y_bin.ravel(), probs.ravel())
    ap = average_precision_score(Y_bin, probs, average="macro")
    ax4.plot(rec, prec, lw=2, label=f"{m} (AP = {ap:.3f})")
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
    values = [r["accuracy"], r["f1_macro"], r["precision_macro"], r["recall_macro"]]
    values += values[:1]
    ax5.plot(angles, values, linewidth=2, label=m)
    ax5.fill(angles, values, alpha=0.1)
plt.title("Radar Chart - Top 5 Models (CV means)", y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/5_radar_chart.png", dpi=300, bbox_inches="tight"); plt.close(fig5)

# 6. OOF confusion matrix heatmap for the best model
best_model = models_sorted[0]
preds, probs, labels = load_oof(best_model)
cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
fig6, ax6 = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax6, vmin=0, vmax=1)
ax6.set_xlabel("Predicted"); ax6.set_ylabel("True")
ax6.set_title(f"{best_model} — Out-of-Fold Confusion Matrix")
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/6_best_model_oof_cm.png", dpi=300); plt.close(fig6)

# 7. Model prediction correlation heatmap (OOF, ensemble diversity)
preds_dict = {}
for m in models_sorted:
    p, _, _ = load_oof(m)
    preds_dict[m] = p
df_preds = pd.DataFrame(preds_dict)
corr = df_preds.corr(method="spearman").fillna(0)
fig7, ax7 = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=0, vmax=1, ax=ax7)
ax7.set_title("Model Prediction Correlation (Spearman, OOF)")
plt.tight_layout(); plt.savefig(f"{COMPARISON_DIR}/7_model_correlation_heatmap.png", dpi=300); plt.close(fig7)

print(f"\nAll CV results and paper-ready figures saved to {COMPARISON_DIR}/")
print("Download the output dataset from the Kaggle 'Output' tab.")
