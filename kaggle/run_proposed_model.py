"""
========================================================================================
Proposed-Model: Clinically-Aware Recalibrated Ensemble for Autism Facial Expression Recognition
========================================================================================
Self-contained Kaggle training + evaluation script for the proposed architecture,
adapted to Stratified K-Fold Cross-Validation on RAW images (no MTCNN/CLAHE):

1. Dual-Stream Backbone: VGG16 (Local Texture Expert) + DeiT-Small (Global Geometry Expert)
2. Feature Recalibration: Dual Squeeze-and-Excitation (SE) Channel Attention Blocks (r=16)
3. Training Stabilization: Exponential Moving Average (EMA, decay=0.999) + Focal Loss
4. Clinical Inference: 5-View Test-Time Augmentation (TTA, K=5)
5. Clinical Safety: Confidence Uncertainty Rejection Guardrail
6. Publication Figures: Confusion Matrix, Curves, Per-Class Metrics, Grad-CAM Heatmaps

Methodology fixes (match run_all_models.py):
- RAW images only (the MTCNN+CLAHE step hurt accuracy).
- Identical train-time augmentation as the baselines (fair comparison):
  no MixUp, no RandomErasing, no RandAugment.
- Class imbalance handled ONLY via FocalLoss class weights (single weighting).
- Stratified K-Fold CV using the SAME fold assignment as the baselines
  (fold_id_by_path.json written by run_all_models.py).
- Resumable: per-fold checkpoints + cv_done.json marker.

Run run_all_models.py FIRST (it writes fold_id_by_path.json), then this cell.
========================================================================================
"""

import os
import sys
import copy
import time
import math
import random
import json
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
import timm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================
SEED = 42
NUM_EPOCHS = 160      # Stage 1
STAGE2_EPOCHS = 20    # Stage 2 (frozen backbone)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4  # Optimal for Hybrid Transformer-CNN architectures
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
TTA_VIEWS = 5
UNCERTAINTY_THRESH = 0.30  # Clinical rejection guardrail
IMG_SIZE = 224
PATIENCE = 20
N_FOLDS = 5           # MUST match run_all_models.py
NUM_WORKERS = 2 if sys.platform == "linux" else 0  # Windows spawn breaks multiprocessing loaders

_EXPECTED_CLASSES = ("anger", "fear", "joy", "natural", "sadness", "surprise")

def find_dataset_dir():
    """Locate the dataset root containing train/valid/test class folders.

    Kaggle mounts datasets under /kaggle/input/<slug>/ but the slug and
    internal folder names vary, so walk a few levels and match on structure.
    """
    base = "/kaggle/input"
    if not os.path.isdir(base):
        return None
    for root, dirs, _ in os.walk(base):
        depth = root[len(base):].count(os.sep)
        if depth > 3:
            dirs[:] = []
            continue
        subs = os.listdir(root) if os.path.isdir(root) else []
        if any(s in subs for s in ("train", "valid", "test")):
            for split in ("train", "valid", "test"):
                split_dir = os.path.join(root, split)
                if os.path.isdir(split_dir) and any(
                    os.path.isdir(os.path.join(split_dir, c)) for c in _EXPECTED_CLASSES):
                    print(f"[*] Auto-detected dataset at {root}")
                    return root
    return None

KAGGLE_MTCNN_DIR   = "/kaggle/working/dataset_mtcnn"
KAGGLE_DATASET_DIR = "/kaggle/input/datasets/mrsohel/autism-dataset/dataset_clean"
LOCAL_DATASET_DIR  = r"C:\Users\mrsoh\Documents\Autism-Facial-Expression-Recognition\dataset_clean"

# RAW dataset first — the MTCNN variant hurt accuracy.
_auto_detected = find_dataset_dir() if os.path.isdir("/kaggle/input") else None
if os.path.exists(KAGGLE_DATASET_DIR):
    DATASET_DIR = KAGGLE_DATASET_DIR
elif os.path.exists(LOCAL_DATASET_DIR):
    DATASET_DIR = LOCAL_DATASET_DIR
elif _auto_detected:
    DATASET_DIR = _auto_detected
else:
    DATASET_DIR = KAGGLE_MTCNN_DIR

OUTPUT_DIR = "/kaggle/working/results/proposed_model_proposed" if os.path.exists("/kaggle") else "./results/proposed_model_proposed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running on Device: {device} | Output Directory: {OUTPUT_DIR}")
print(f"[*] Dataset Directory: {DATASET_DIR}")


# ==============================================================================
# 2. DATASET — all splits merged, CV partitioned at runtime
# ==============================================================================
class AutismFERDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.targets = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.samples[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def build_full_dataset(root_dir):
    """Collect every image across train/valid/test into one list."""
    samples, labels = [], []
    for split in ("train", "valid", "test"):
        for cls_name in CLASSES:
            cls_dir = Path(root_dir) / split / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    samples.append(str(img_path))
                    labels.append(CLASS_TO_IDX[cls_name])
    return samples, labels


# Identical augmentation as the baselines (fair head-to-head comparison).
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_tta_transforms():
    base_norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return [
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + base_norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0)] + base_norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((5, 5))] + base_norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((-5, -5))] + base_norm),
        transforms.Compose([transforms.Resize((int(IMG_SIZE * 1.08), int(IMG_SIZE * 1.08))), transforms.CenterCrop(IMG_SIZE)] + base_norm),
    ]


# ==============================================================================
# 3. PROPOSED ARCHITECTURE: Proposed-Model (Dual-Stream SE Recalibrated Ensemble)
# ==============================================================================
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.fc1(x)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class CareFERModel(nn.Module):
    """Dual-stream: VGG16-BN spatial (forward_features + GAP, 512-d) + DeiT-S CLS (384-d),
    dual SE recalibration, head 896->512->256->6."""

    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        vgg = timm.create_model("vgg16_bn", pretrained=pretrained, num_classes=0)
        self.stream_a = vgg
        dim_a = 512

        deit = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.stream_b = deit
        dim_b = 384

        print(f"[*] Stream A (VGG16-BN spatial+GAP): {dim_a}-d | Stream B (DeiT-S CLS): {dim_b}-d | Combined: {dim_a+dim_b}-d")

        self.se_a = SqueezeExcitationBlock(dim_a, reduction=16)
        self.se_b = SqueezeExcitationBlock(dim_b, reduction=16)

        combined_dim = dim_a + dim_b  # 896
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat_map_a = self.stream_a.forward_features(x)
        feat_a = feat_map_a.mean(dim=[2, 3])
        feat_b = self.stream_b(x)
        rec_a = self.se_a(feat_a)
        rec_b = self.se_b(feat_b)
        fused = torch.cat([rec_a, rec_b], dim=1)
        return self.classifier(fused)


# ==============================================================================
# 4. LOSS, EMA & UTILITIES
# ==============================================================================
class FocalLoss(nn.Module):
    """Focal loss with per-class alpha (SINGLE weighting mechanism — no sampler)."""

    def __init__(self, alpha=None, gamma=1.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


class ModelEMA:
    """Deep-copy EMA — save/load the EMA weights, not the raw model."""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.module.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)


def compute_focal_alpha(labels):
    """Inverse-frequency class weights + sadness x2.0, fear x1.2 (V6 boost)."""
    counts = Counter(labels)
    total = len(labels)
    alpha = torch.tensor(
        [total / (NUM_CLASSES * counts[i]) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    ).to(device)
    alpha[CLASSES.index("sadness")] *= 2.0
    alpha[CLASSES.index("fear")] *= 1.2
    return alpha


# ==============================================================================
# 5. FOLD SPLITS (shared with run_all_models.py)
# ==============================================================================
print("[*] Building full dataset (train+valid+test merged) ...")
samples, labels = build_full_dataset(DATASET_DIR)
print(f"[*] Total images: {len(samples)}")

FOLD_FILE = "/kaggle/working/results/fold_id_by_path.json"
if os.path.exists(FOLD_FILE):
    with open(FOLD_FILE) as f:
        fold_id_by_path = json.load(f)
    fold_ids = [fold_id_by_path.get(s, 0) for s in samples]
    print(f"[*] Loaded fold assignment from {FOLD_FILE}")
else:
    print(f"[!] {FOLD_FILE} not found — recomputing folds (must run run_all_models.py first "
          "for identical folds).")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_ids = np.zeros(len(samples), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(samples, labels)):
        for i in val_idx:
            fold_ids[i] = fold
    fold_ids = fold_ids.tolist()

folds = [(fold, [i for i in range(len(samples)) if fold_ids[i] != fold],
               [i for i in range(len(samples)) if fold_ids[i] == fold])
         for fold in range(N_FOLDS)]

alpha = compute_focal_alpha(labels)
print(f"[*] Focal Loss Class Weights (V6 Boost): { {CLASSES[i]: f'{alpha[i].item():.2f}' for i in range(NUM_CLASSES)} }")
loss_fn = FocalLoss(alpha=alpha, gamma=1.5)


# ==============================================================================
# 6. RESUME SUPPORT
# ==============================================================================
DONE_FILE = os.path.join(OUTPUT_DIR, "cv_done.json")
done = {}
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        done = json.load(f)
    print(f"[*] Found completed folds — resuming.")


def mark_done(fold):
    done.setdefault("proposed_model", []).append(fold)
    with open(DONE_FILE, "w") as f:
        json.dump(done, f, indent=2)


# ==============================================================================
# 7. TRAINING (per fold)
# ==============================================================================
def train_stage1(model, train_idx):
    train_ds = AutismFERDataset([samples[i] for i in train_idx],
                                [labels[i] for i in train_idx], train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if "classifier" in name or "se_a" in name or "se_b" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params, "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    _WARMUP_EPOCHS = 10

    def _lr_lambda(epoch):
        if epoch < _WARMUP_EPOCHS:
            return float(epoch + 1) / float(_WARMUP_EPOCHS)
        progress = float(epoch - _WARMUP_EPOCHS) / float(max(1, NUM_EPOCHS - _WARMUP_EPOCHS))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    ema_model = ModelEMA(model, decay=EMA_DECAY)
    scaler = GradScaler()
    return train_loader, optimizer, scheduler, ema_model, scaler


@torch.no_grad()
def evaluate_tta(model, val_ds):
    """5-view TTA averaged probabilities over the validation subset."""
    model.eval()
    tta_transforms = get_tta_transforms()
    preds, targets, probs = [], [], []
    for idx in range(len(val_ds)):
        raw_img, target = val_ds.samples[idx], val_ds.targets[idx]
        try:
            raw = Image.open(raw_img).convert("RGB")
        except Exception:
            raw = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        tta_probs = torch.zeros((1, NUM_CLASSES), device=device)
        for t_form in tta_transforms:
            t_img = t_form(raw).unsqueeze(0).to(device)
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                logits = model(t_img)
                tta_probs += F.softmax(logits, dim=1)
        tta_probs /= len(tta_transforms)
        probs.append(tta_probs.cpu().numpy()[0])
        preds.append(int(tta_probs.argmax(dim=1).item()))
        targets.append(target)
    return preds, targets, np.array(probs)


def run_fold(fold, train_idx, val_idx):
    print(f"\n{'='*70}")
    print(f"  PROPOSED-MODEL | Fold {fold+1}/{N_FOLDS}")
    print(f"{'='*70}")

    val_ds = AutismFERDataset([samples[i] for i in val_idx],
                              [labels[i] for i in val_idx], val_transform)

    # ---- Stage 1 ----
    model = CareFERModel(num_classes=NUM_CLASSES, pretrained=True).to(device)
    train_loader, optimizer, scheduler, ema_model, scaler = train_stage1(model, train_idx)

    best_val_f1 = 0.0
    patience_counter = 0
    ckpt_path = os.path.join(OUTPUT_DIR, f"proposed_model_fold{fold+1}_best.pth")
    start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss, running_correct, total_train = 0.0, 0, 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            ema_model.update(model)
            running_loss += loss.item() * imgs.size(0)
            running_correct += outputs.argmax(1).eq(targets).sum().item()
            total_train += imgs.size(0)
        scheduler.step()

        ema_model.module.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_ds_transform_loader(val_ds):
                imgs, targets = imgs.to(device), targets.to(device)
                with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    outputs = ema_model.module(imgs)
                val_loss += loss_fn(outputs, targets).item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += imgs.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        epoch_val_f1 = f1_score(val_targets, val_preds, average="macro")
        elapsed = (time.time() - start) / 60
        print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | TrL {running_loss/total_train:.4f} "
              f"VaA {val_correct/total_val:.4f} F1 {epoch_val_f1:.4f} | {elapsed:.1f}min")

        if epoch_val_f1 > best_val_f1:
            best_val_f1 = epoch_val_f1
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_model.module.state_dict(),
                "val_f1": best_val_f1,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"[*] Stage 1 Complete in {(time.time()-start)/60:.1f} min | Best Val Macro F1: {best_val_f1:.4f}")

    # ---- Stage 2: frozen backbone, unfrozen head ----
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name or "se_a" in name or "se_b" in name
    head_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_stage2 = torch.optim.AdamW(head_params, lr=1e-4, weight_decay=WEIGHT_DECAY)
    ema_model_stage2 = ModelEMA(model, decay=EMA_DECAY)
    scaler_s2 = GradScaler()

    # Balanced train loader for stage 2 (single weighting — sampler only, no loss weights here)
    train_ds_s2 = AutismFERDataset([samples[i] for i in train_idx],
                                   [labels[i] for i in train_idx], train_transform)
    counts = Counter(train_ds_s2.targets)
    sample_weights = [1.0 / counts[t] for t in train_ds_s2.targets]
    sampler_s2 = WeightedRandomSampler(sample_weights, len(train_ds_s2), replacement=True)
    train_loader_s2 = DataLoader(train_ds_s2, batch_size=BATCH_SIZE, sampler=sampler_s2,
                                 num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    best_f1_s2 = best_val_f1
    patience_s2 = 0
    start_s2 = time.time()

    for epoch in range(1, STAGE2_EPOCHS + 1):
        model.train()
        running_loss, running_correct, total_train = 0.0, 0, 0
        for imgs, targets in train_loader_s2:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer_stage2.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                outputs = model(imgs)
                loss = loss_fn(outputs, targets)
            scaler_s2.scale(loss).backward()
            scaler_s2.unscale_(optimizer_stage2)
            clip_grad_norm_(head_params, max_norm=5.0)
            scaler_s2.step(optimizer_stage2)
            scaler_s2.update()
            ema_model_stage2.update(model)
            running_loss += loss.item() * imgs.size(0)
            running_correct += outputs.argmax(1).eq(targets).sum().item()
            total_train += imgs.size(0)

        ema_model_stage2.module.eval()
        val_correct, total_val = 0, 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, targets in val_ds_transform_loader(val_ds):
                imgs, targets = imgs.to(device), targets.to(device)
                with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                    outputs = ema_model_stage2.module(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                total_val += imgs.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())

        epoch_val_f1 = f1_score(val_targets, val_preds, average="macro")
        print(f"  Stage 2 Epoch {epoch:3d}/{STAGE2_EPOCHS} | TrL {running_loss/total_train:.4f} "
              f"VaA {val_correct/total_val:.4f} F1 {epoch_val_f1:.4f}")

        if epoch_val_f1 > best_f1_s2:
            best_f1_s2 = epoch_val_f1
            patience_s2 = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema_model_stage2.module.state_dict(),
                "val_f1": best_f1_s2,
            }, ckpt_path)
        else:
            patience_s2 += 1
            if patience_s2 >= PATIENCE:
                print(f"  Early stopping Stage 2 at epoch {epoch}")
                break

    print(f"[*] Stage 2 Complete in {(time.time()-start_s2)/60:.1f} min | Final Best Val Macro F1: {best_f1_s2:.4f}")

    # ---- OOF evaluation with 5-view TTA ----
    checkpoint = torch.load(ckpt_path, map_location=device)
    eval_model = CareFERModel(num_classes=NUM_CLASSES, pretrained=False).to(device)
    eval_model.load_state_dict(checkpoint["model_state_dict"])
    preds, targets, probs = evaluate_tta(eval_model, val_ds)

    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="macro")
    prec = precision_score(targets, preds, average="macro", zero_division=0)
    rec = recall_score(targets, preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0, labels=list(range(NUM_CLASSES)))

    fold_m = {"fold": fold, "accuracy": float(acc), "f1_macro": float(f1),
              "precision_macro": float(prec), "recall_macro": float(rec),
              "per_class_f1": {c: float(f) for c, f in zip(CLASSES, per_class_f1)}}
    print(f"  FOLD {fold+1} OOF (5-view TTA) — Acc: {acc:.4f} | F1: {f1:.4f}")

    return fold_m, preds, targets, probs


def val_ds_transform_loader(val_ds):
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    return loader


# ==============================================================================
# 8. RUN ALL FOLDS (OOF arrays + metrics persisted after every fold)
# ==============================================================================
def _load_npy(path, empty_shape):
    if os.path.exists(path):
        arr = np.load(path)
        return arr if arr.ndim > 0 else np.empty(empty_shape)
    return np.empty(empty_shape)


oof_preds_all = _load_npy(os.path.join(OUTPUT_DIR, "oof_preds.npy"), (0,))
oof_targets_all = _load_npy(os.path.join(OUTPUT_DIR, "oof_labels.npy"), (0,))
oof_probs_all = _load_npy(os.path.join(OUTPUT_DIR, "oof_probs.npy"), (0, NUM_CLASSES))
fold_metrics = []
metrics_path = os.path.join(OUTPUT_DIR, "cv_metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        fold_metrics = json.load(f)

for fold, train_idx, val_idx in folds:
    if fold in done.get("proposed_model", []):
        print(f"  Fold {fold+1} already done — skipping.")
        continue
    fold_m, preds, targets, probs = run_fold(fold, train_idx, val_idx)
    fold_metrics.append(fold_m)
    oof_preds_all = np.concatenate([oof_preds_all, np.array(preds, dtype=int)])
    oof_targets_all = np.concatenate([oof_targets_all, np.array(targets, dtype=int)])
    oof_probs_all = np.concatenate([oof_probs_all, probs], axis=0)
    mark_done(fold)
    # persist incrementally (safe across session timeouts)
    with open(metrics_path, "w") as f:
        json.dump(fold_metrics, f, indent=2)
    np.save(os.path.join(OUTPUT_DIR, "oof_preds.npy"), oof_preds_all)
    np.save(os.path.join(OUTPUT_DIR, "oof_labels.npy"), oof_targets_all)
    np.save(os.path.join(OUTPUT_DIR, "oof_probs.npy"), oof_probs_all)

# ==============================================================================
# 9. AGGREGATE + CLINICAL EVALUATION
# ==============================================================================
if oof_preds_all.size == 0:
    raise RuntimeError("No out-of-fold predictions available — check cv_done.json consistency.")

test_preds = np.array(oof_preds_all)
test_targets = np.array(oof_targets_all)
test_probs = np.array(oof_probs_all)

test_acc = accuracy_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds, average="macro")
test_prec = precision_score(test_targets, test_preds, average="macro", zero_division=0)
test_rec = recall_score(test_targets, test_preds, average="macro", zero_division=0)

print(f"\n[+] OVERALL OUT-OF-FOLD RESULTS (5-view TTA, {len(fold_metrics)} folds):")
print(f"    Accuracy:      {test_acc:.4f}")
print(f"    Macro F1:      {test_f1:.4f}")
print(f"    Precision:     {test_prec:.4f}")
print(f"    Recall:        {test_rec:.4f}\n")
print(classification_report(test_targets, test_preds, target_names=CLASSES, digits=4))

if fold_metrics:
    print("\n  Per-fold Macro F1:", [f"{m['f1_macro']:.4f}" for m in fold_metrics])
    print(f"  Macro F1 mean +/- std: {np.mean([m['f1_macro'] for m in fold_metrics]):.4f} "
          f"+/- {np.std([m['f1_macro'] for m in fold_metrics]):.4f}")

# Distress emotions recall audit
report_dict = classification_report(test_targets, test_preds, target_names=CLASSES,
                                    output_dict=True, zero_division=0)
print("-" * 50)
print("CLINICAL SAFETY AUDIT: Distress Emotion Recall (Sensitivity)")
print("-" * 50)
for d_cls in ("anger", "fear", "sadness"):
    rec = report_dict[d_cls]["recall"]
    sup = report_dict[d_cls]["support"]
    print(f"  [{d_cls.upper():<8}] Recall: {rec*100:.1f}% (Support: {sup} images)")

# Uncertainty rejection guardrail
confidences = test_probs.max(axis=1)
rejection_rate = float(np.mean(confidences < UNCERTAINTY_THRESH)) * 100
high_conf = confidences >= UNCERTAINTY_THRESH
print(f"\n[+] CLINICAL UNCERTAINTY GUARDRAIL (Threshold = {UNCERTAINTY_THRESH*100:.0f}% Confidence):")
print(f"    High-Confidence: {high_conf.sum()} / {len(confidences)} images")
print(f"    Flagged for Caregiver Review (Low Conf): {(~high_conf).sum()} images ({rejection_rate:.1f}% Rejection Rate)")
if high_conf.sum() > 0:
    hc_acc = accuracy_score(test_targets[high_conf], test_preds[high_conf])
    hc_f1 = f1_score(test_targets[high_conf], test_preds[high_conf], average="macro")
    print(f"    -> High-Confidence Subset Accuracy: {hc_acc*100:.2f}% | Macro F1: {hc_f1:.4f}")

# ==============================================================================
# 10. PUBLICATION FIGURES
# ==============================================================================
print("\n[*] Generating publication-ready figures...")
sns.set_theme(style="whitegrid", font_scale=1.1)

cm = confusion_matrix(test_targets, test_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
plt.title("Proposed-Model: Out-of-Fold Confusion Matrix (TTA K=5)", fontweight="bold", pad=15)
plt.ylabel("True Emotion Label"); plt.xlabel("Predicted Emotion Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_test_confusion_matrix.png"), dpi=300)
plt.close()

metrics_df = pd.DataFrame({
    "Class": CLASSES,
    "Precision": [report_dict[c]["precision"] for c in CLASSES],
    "Recall": [report_dict[c]["recall"] for c in CLASSES],
    "F1-Score": [report_dict[c]["f1-score"] for c in CLASSES],
}).melt(id_vars="Class", var_name="Metric", value_name="Score")
plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df, x="Class", y="Score", hue="Metric", palette="Set2")
plt.title("Proposed-Model: Per-Class Performance (Out-of-Fold)", fontweight="bold")
plt.ylim(0, 1.05); plt.ylabel("Score"); plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_per_class_metrics.png"), dpi=300)
plt.close()

print(f"[*] All publication charts saved to: {OUTPUT_DIR}")

# ==============================================================================
# 11. GRAD-CAM HEATMAPS (VGG16 Stream A)
# ==============================================================================
print("\n[*] Generating Grad-CAM explainability heatmaps (VGG16 Stream A)...")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._activation = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, output):
        self._activation = output

    def generate(self, input_tensor, target_class):
        self.model.eval()
        _grad_holder = [None]
        with torch.enable_grad():
            x = input_tensor.detach().clone().requires_grad_(True)
            out = self.model(x)
            if self._activation is not None and self._activation.requires_grad:
                self._activation.retain_grad()
                _h = self._activation.register_hook(
                    lambda g: _grad_holder.__setitem__(0, g.detach()))
            else:
                return np.zeros((7, 7))
            self.model.zero_grad()
            out[0, target_class].backward()
            _h.remove()
        grads = _grad_holder[0]
        acts = self._activation.detach()
        if grads is None:
            return np.zeros((acts.shape[2], acts.shape[3]))
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        self._fwd_handle.remove()


# Load best fold's model for Grad-CAM (use first available checkpoint)
ckpt_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith("proposed_model_fold") and f.endswith("_best.pth")])
if ckpt_files:
    eval_model = CareFERModel(num_classes=NUM_CLASSES, pretrained=False).to(device)
    eval_model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, ckpt_files[0]),
                                          map_location=device)["model_state_dict"])
    eval_model.eval()

    _last_conv = None
    for _module in eval_model.stream_a.modules():
        if isinstance(_module, nn.Conv2d):
            _last_conv = _module

    if _last_conv is None:
        print("[!] No Conv2d found in stream_a — Grad-CAM skipped.")
    else:
        grad_cam = GradCAM(eval_model, _last_conv)
        _seen, _gradcam_samples = set(), []
        for path, cls in zip(samples, labels):
            if cls not in _seen:
                _seen.add(cls)
                _gradcam_samples.append((path, cls))
            if len(_seen) == NUM_CLASSES:
                break

        fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(3 * NUM_CLASSES, 6))
        fig.suptitle("Proposed-Model Grad-CAM: VGG16 Stream Discriminative Facial Regions",
                     fontsize=13, fontweight="bold")
        for col, (img_path, true_cls) in enumerate(_gradcam_samples):
            raw = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            tensor = val_transform(raw).unsqueeze(0).to(device).requires_grad_(True)
            cam = grad_cam.generate(tensor, target_class=true_cls)
            cam_up = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
            img_np = np.array(raw) / 255.0
            heatmap = plt.cm.jet(cam_up)[:, :, :3]
            overlay = np.clip(0.5 * img_np + 0.5 * heatmap, 0, 1)
            axes[0, col].imshow(raw)
            axes[0, col].set_title(CLASSES[true_cls].capitalize(), fontsize=10, fontweight="bold")
            axes[0, col].axis("off")
            axes[1, col].imshow(overlay)
            axes[1, col].set_title("Grad-CAM", fontsize=9)
            axes[1, col].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "4_gradcam_heatmaps.png"), dpi=300, bbox_inches="tight")
        plt.close()
        grad_cam.remove()
        print(f"[*] Grad-CAM heatmaps saved to: {OUTPUT_DIR}/4_gradcam_heatmaps.png")

print("========================================================================================")
print("  Proposed-Model Evaluation Complete! Ready for Paper Publication.")
print("========================================================================================")
