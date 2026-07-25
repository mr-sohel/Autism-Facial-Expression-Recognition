"""
========================================================================================
Care-FER: Clinically-Aware Recalibrated Ensemble for Autism Facial Expression Recognition
========================================================================================
Self-contained Kaggle training and evaluation script for the proposed architecture:
1. Dual-Stream Backbone: VGG16 (Local Texture Expert) + DeiT-Small (Global Geometry Expert)
2. Feature Recalibration: Dual Squeeze-and-Excitation (SE) Channel Attention Blocks (r=16)
3. Training Stabilization: Exponential Moving Average (EMA, decay=0.999) + Focal Loss
4. Clinical Inference: 5-View Test-Time Augmentation (TTA, K=5)
5. Clinical Safety: 70% Confidence Uncertainty Rejection Guardrail
6. Publication Figures: Confusion Matrix, Training Curves, Per-Class F1, and Grad-CAM Heatmaps
========================================================================================
To run on Kaggle:
1. Create a Kaggle Notebook with GPU accelerated (Dual T4 or P100).
2. Upload your deduplicated dataset to Kaggle as a dataset named 'autism-facial-expression-recognition'.
3. Copy-paste this entire script into a code cell and run!
========================================================================================
"""

import os
import sys
import copy
import time
import math
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.amp import autocast, GradScaler
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

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================
SEED = 42
NUM_EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 1e-4  # Optimal for Hybrid Transformer-CNN architectures
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
TTA_VIEWS = 5         # K=5 for Test-Time Augmentation
UNCERTAINTY_THRESH = 0.70  # Clinical rejection guardrail (70% confidence)
IMG_SIZE = 224
PATIENCE = 15         # Early stopping patience (epochs without F1 improvement)
MIXUP_ALPHA = 0.4     # MixUp regularization strength
NUM_WORKERS = 2

# Kaggle dataset paths (fallback to local if running locally for testing)
KAGGLE_DATASET_DIR = "/kaggle/input/datasets/mrsohel/autism-dataset/dataset"  # Confirmed slug from Kaggle notebook logs
LOCAL_DATASET_DIR = r"C:\Users\mrsoh\Documents\Autism-Facial-Expression-Recognition\dataset"
DATASET_DIR = KAGGLE_DATASET_DIR if os.path.exists(KAGGLE_DATASET_DIR) else LOCAL_DATASET_DIR

OUTPUT_DIR = "/kaggle/working/results/care_fer_proposed" if os.path.exists("/kaggle") else "./results/care_fer_proposed"
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
# 2. DATASET & CLINICAL AUGMENTATIONS
# ==============================================================================
class AutismFERDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        self.targets = []
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.root_dir}")
            
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists(): continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self.samples.append(str(img_path))
                    self.targets.append(cls_idx)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        if self.transform:
            img = self.transform(img)
        return img, target

# Standard training transforms (matches baseline run_all_models.py augmentation strength)
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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

# Standard validation transform
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 5-View Test-Time Augmentation (TTA) transforms generator
def get_tta_transforms():
    base_norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return [
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + base_norm), # 1. Center original
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(p=1.0)] + base_norm), # 2. Flip H
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((5, 5))] + base_norm), # 3. Rotate +5 deg
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomRotation((-5, -5))] + base_norm), # 4. Rotate -5 deg
        transforms.Compose([transforms.Resize((int(IMG_SIZE*1.08), int(IMG_SIZE*1.08))), transforms.CenterCrop(IMG_SIZE)] + base_norm), # 5. Slight zoom
    ]

# Load Datasets
train_dataset = AutismFERDataset(DATASET_DIR, split="train", transform=train_transform)
val_dataset = AutismFERDataset(DATASET_DIR, split="valid", transform=val_transform)
test_dataset = AutismFERDataset(DATASET_DIR, split="test", transform=val_transform)

# Weighted Random Sampler to overcome 10:1 severe class imbalance on fear
class_counts = Counter(train_dataset.targets)
total_samples = len(train_dataset)
class_weights = {cls_idx: total_samples / (len(class_counts) * count) for cls_idx, count in class_counts.items()}
sample_weights = [class_weights[target] for target in train_dataset.targets]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total_samples, replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"[*] Dataset Loaded: Train={len(train_dataset)} | Valid={len(val_dataset)} | Test={len(test_dataset)}")


# ==============================================================================
# 3. PROPOSED ARCHITECTURE: Care-FER (Dual-Stream SE Recalibrated Ensemble)
# ==============================================================================
class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block (r=16).
    Dynamically recalibrates feature channels to suppress background noise and amplify emotion cues.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (B, C)
        w = self.fc1(x)
        w = self.relu(w)
        w = self.fc2(w)
        w = self.sigmoid(w)
        return x * w


class CareFERModel(nn.Module):
    """
    Proposed Dual-Stream Architecture:
    - Stream A: VGG16 (Local Texture Expert - eyes, lips micro-features)
    - Stream B: DeiT-Small (Global Geometry Expert - holistic attention)
    - Dual SE Blocks: Recalibrates both streams independently before fusion
    - Classification Head: Blends features with dropout and linear projection
    """
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        # Stream A: VGG16 Backbone
        vgg = timm.create_model("vgg16_bn", pretrained=pretrained, num_classes=0)
        self.stream_a = vgg

        # Stream B: DeiT-Small Backbone
        deit = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.stream_b = deit

        # Probe ACTUAL output dimensions with a dummy forward pass.
        # timm's num_features can be wrong for some models (e.g. vgg16_bn reports 512
        # but actually outputs 4096 after its FC pre_logits layers).
        with torch.no_grad():
            _dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
            dim_a = int(vgg(_dummy).shape[1])
            dim_b = int(deit(_dummy).shape[1])
        print(f"[*] Stream A (VGG16-BN) feature dim: {dim_a} | Stream B (DeiT-Small) feature dim: {dim_b}")

        self.se_a = SqueezeExcitationBlock(dim_a, reduction=16)
        self.se_b = SqueezeExcitationBlock(dim_b, reduction=16)

        # Combined Classification Head
        combined_dim = dim_a + dim_b
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features
        feat_a = self.stream_a(x)  # (B, dim_a)
        feat_b = self.stream_b(x)  # (B, dim_b)

        # Recalibrate with Dual SE Blocks
        rec_a = self.se_a(feat_a)
        rec_b = self.se_b(feat_b)

        # Fuse feature streams
        fused = torch.cat([rec_a, rec_b], dim=1)  # (B, dim_a + dim_b)
        out = self.classifier(fused)
        return out

model = CareFERModel(num_classes=NUM_CLASSES, pretrained=True).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"[*] Proposed Model Created: Care-FER | Total Parameters: {total_params:,} ({total_params/1e6:.1f}M)")


# ==============================================================================
# 4. LOSS FUNCTION, EMA & TRAINING UTILITIES
# ==============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for severe clinical class imbalance (prioritizing distress emotions).
    alpha: per-class weight tensor (inverse-frequency) or None for uniform weighting.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # Tensor of shape (num_classes,) or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Pass per-class weights into CE so rare classes (fear, anger) get upweighted
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        return focal_loss.sum()


class ModelEMA:
    """Exponential Moving Average (EMA) for training weight stabilization."""
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

# Compute inverse-frequency class weights (matches run_all_models.py baseline formula)
# This upweights rare classes (fear: weight ~3.89) to boost distress emotion recall
_class_counts_list = [class_counts[i] for i in range(NUM_CLASSES)]
_class_weights = torch.tensor(
    [total_samples / (NUM_CLASSES * c) for c in _class_counts_list], dtype=torch.float32
).to(device)
print(f"[*] Focal Loss Class Weights: { {CLASSES[i]: f'{_class_weights[i].item():.2f}' for i in range(NUM_CLASSES)} }")
loss_fn = FocalLoss(alpha=_class_weights, gamma=2.0)

# Differential learning rate: backbones get 0.1x LR, SE blocks & head get full LR
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "classifier" in name or "se_a" in name or "se_b" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LEARNING_RATE * 0.1},
    {"params": head_params, "lr": LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
ema_model = ModelEMA(model, decay=EMA_DECAY)
scaler = GradScaler()


# MixUp augmentation (matches baseline pipeline)
def mixup_data(x, y, alpha=MIXUP_ALPHA):
    """MixUp: blends pairs of training images and labels for regularization."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


# ==============================================================================
# 5. TRAINING & VALIDATION LOOP
# ==============================================================================
print("\n" + "="*70)
print("  STARTING CLINICAL TRAINING: Care-FER Architecture")
print("="*70)

best_val_f1 = 0.0
best_model_path = os.path.join(OUTPUT_DIR, "care_fer_best.pth")
train_losses, val_losses, train_accs, val_accs, val_f1s = [], [], [], [], []

patience_counter = 0
start_time = time.time()
for epoch in range(1, NUM_EPOCHS + 1):
    # --- TRAINING (with MixUp) ---
    model.train()
    running_loss, running_correct, total_train = 0.0, 0, 0
    
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
            mixed_imgs, ya, yb, lam = mixup_data(imgs, targets)
            outputs = model(mixed_imgs)
            loss = lam * loss_fn(outputs, ya) + (1 - lam) * loss_fn(outputs, yb)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        ema_model.update(model)
        
        running_loss += loss.item() * imgs.size(0)
        # MixUp-aware accuracy: weighted sum of matches against both mixed labels
        running_correct += (lam * outputs.argmax(1).eq(ya).sum().item() +
                           (1 - lam) * outputs.argmax(1).eq(yb).sum().item())
        total_train += imgs.size(0)
        
    scheduler.step()
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = running_correct / total_train
    
    # --- VALIDATION (using EMA weights) ---
    ema_model.module.eval()
    val_loss, val_correct, total_val = 0.0, 0, 0
    val_preds_list, val_targets_list = [], []
    
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
                outputs = ema_model.module(imgs)
                loss = loss_fn(outputs, targets)
                
            val_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == targets).sum().item()
            total_val += imgs.size(0)
            
            val_preds_list.extend(preds.cpu().numpy())
            val_targets_list.extend(targets.cpu().numpy())
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = val_correct / total_val
    epoch_val_f1 = f1_score(val_targets_list, val_preds_list, average="macro")
    
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accs.append(epoch_train_acc)
    val_accs.append(epoch_val_acc)
    val_f1s.append(epoch_val_f1)
    
    # Log progress every epoch (matches baseline verbosity)
    elapsed = (time.time() - start_time) / 60
    print(f"  Epoch {epoch:3d}/{NUM_EPOCHS} | TrL {epoch_train_loss:.4f} TrA {epoch_train_acc:.4f} | "
          f"VlL {epoch_val_loss:.4f} VaA {epoch_val_acc:.4f} F1 {epoch_val_f1:.4f} | {elapsed:.1f}min")
        
    if epoch_val_f1 > best_val_f1:
        best_val_f1 = epoch_val_f1
        patience_counter = 0
        print(f"    >> New best F1: {best_val_f1:.4f}")
        torch.save({
            "epoch": epoch,
            "model_state_dict": ema_model.module.state_dict(),
            "val_f1": best_val_f1,
            "val_acc": epoch_val_acc,
        }, best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch} (no F1 improvement for {PATIENCE} epochs)")
            break

total_time = (time.time() - start_time) / 60
print(f"\n[*] Training Complete in {total_time:.1f} minutes! Best Validation Macro F1: {best_val_f1:.4f}")
print(f"[*] Best model checkpoint saved to: {best_model_path}")


# ==============================================================================
# 6. CLINICAL EVALUATION WITH 5-VIEW TTA & UNCERTAINTY GUARDRAIL
# ==============================================================================
print("\n" + "="*70)
print(f"  EVALUATING ON TEST SET ({len(test_dataset)} Images) WITH 5-VIEW TTA & GUARDRAILS")
print("="*70)

# Load best checkpoint
checkpoint = torch.load(best_model_path, map_location=device)
eval_model = CareFERModel(num_classes=NUM_CLASSES, pretrained=False).to(device)
eval_model.load_state_dict(checkpoint["model_state_dict"])
eval_model.eval()

tta_transforms = get_tta_transforms()
test_preds, test_targets, test_confidences = [], [], []

with torch.no_grad():
    for img_path, target in zip(test_dataset.samples, test_dataset.targets):
        try:
            raw_img = Image.open(img_path).convert("RGB")
        except Exception:
            raw_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
            
        # Run 5 augmented crops/flips through model and average probabilities
        tta_probs = torch.zeros((1, NUM_CLASSES), device=device)
        for t_idx, t_form in enumerate(tta_transforms):
            t_img = t_form(raw_img).unsqueeze(0).to(device)
            with autocast(device_type="cuda"):  # FIX: bare autocast() is deprecated
                logits = eval_model(t_img)
                probs = F.softmax(logits, dim=1)
            tta_probs += probs
            
        tta_probs /= len(tta_transforms)
        max_prob, pred_cls = torch.max(tta_probs, dim=1)
        
        test_preds.append(pred_cls.item())
        test_targets.append(target)
        test_confidences.append(max_prob.item())

# Standard Test Metrics
test_acc = accuracy_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds, average="macro")
test_prec = precision_score(test_targets, test_preds, average="macro", zero_division=0)
test_rec = recall_score(test_targets, test_preds, average="macro", zero_division=0)

print(f"\n[★] OVERALL TEST RESULTS (with 5-View TTA):")
print(f"    Accuracy:      {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"    Macro F1:      {test_f1:.4f}")
print(f"    Precision:     {test_prec:.4f}")
print(f"    Recall:        {test_rec:.4f}\n")

print(classification_report(test_targets, test_preds, target_names=CLASSES, digits=4))

# Distress Emotions Recall Audit (Clinical Safety Check)
distress_classes = ["anger", "fear", "sadness"]
print("-" * 50)
print("CLINICAL SAFETY AUDIT: Distress Emotion Recall (Sensitivity)")
print("-" * 50)
report_dict = classification_report(test_targets, test_preds, target_names=CLASSES, output_dict=True)
for d_cls in distress_classes:
    rec = report_dict[d_cls]["recall"]
    sup = report_dict[d_cls]["support"]
    print(f"  [{d_cls.upper():<8}] Recall: {rec*100:.1f}% (Support: {sup} images)")
print("-" * 50)

# Clinical Uncertainty Rejection Guardrail (>70% Confidence Check)
high_conf_indices = [i for i, conf in enumerate(test_confidences) if conf >= UNCERTAINTY_THRESH]
low_conf_indices = [i for i, conf in enumerate(test_confidences) if conf < UNCERTAINTY_THRESH]
rejection_rate = len(low_conf_indices) / len(test_confidences) * 100

print(f"\n[★] CLINICAL UNCERTAINTY GUARDRAIL (Threshold = {UNCERTAINTY_THRESH*100:.0f}% Confidence):")
print(f"    High-Confidence Diagnostic Diagnoses: {len(high_conf_indices)} / {len(test_confidences)} images")
print(f"    Flagged for Caregiver Review (Low Conf): {len(low_conf_indices)} images ({rejection_rate:.1f}% Rejection Rate)")

if high_conf_indices:
    hc_targets = [test_targets[i] for i in high_conf_indices]
    hc_preds = [test_preds[i] for i in high_conf_indices]
    hc_acc = accuracy_score(hc_targets, hc_preds)
    hc_f1 = f1_score(hc_targets, hc_preds, average="macro")
    print(f"    -> High-Confidence Subset Accuracy: {hc_acc*100:.2f}% | Macro F1: {hc_f1:.4f}")


# ==============================================================================
# 7. PUBLICATION-READY PLOTS & ARTIFACT GENERATION
# ==============================================================================
print("\n[*] Generating publication-ready figures in 300 DPI...")
sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot 1: Training & Validation Curves
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Focal Loss", color="royalblue", lw=2)
plt.plot(val_losses, label="Val Focal Loss (EMA)", color="darkorange", lw=2)
plt.title("Care-FER: Training & Validation Loss", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in train_accs], label="Train Accuracy", color="royalblue", lw=2)
plt.plot([a * 100 for a in val_accs], label="Val Accuracy (EMA)", color="darkorange", lw=2)
plt.title("Care-FER: Accuracy Curves (%)", fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "1_training_curves.png"), dpi=300)
plt.close()

# Plot 2: Confusion Matrix (Normalized)
cm = confusion_matrix(test_targets, test_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES, cbar=True)
plt.title("Care-FER: Normalized Test Confusion Matrix (TTA K=5)", fontweight="bold", pad=15)
plt.ylabel("True Emotion Label")
plt.xlabel("Predicted Emotion Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "2_test_confusion_matrix.png"), dpi=300)
plt.close()

# Plot 3: Per-Class F1, Precision, and Recall Bar Chart
metrics_df = pd.DataFrame({
    "Class": CLASSES,
    "Precision": [report_dict[c]["precision"] for c in CLASSES],
    "Recall": [report_dict[c]["recall"] for c in CLASSES],
    "F1-Score": [report_dict[c]["f1-score"] for c in CLASSES],
}).melt(id_vars="Class", var_name="Metric", value_name="Score")

plt.figure(figsize=(10, 6))
sns.barplot(data=metrics_df, x="Class", y="Score", hue="Metric", palette="Set2")
plt.title("Care-FER: Per-Class Diagnostic Performance on Test Set", fontweight="bold")
plt.ylim(0, 1.05)
plt.ylabel("Score")
plt.legend(title="Metric")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "3_per_class_metrics.png"), dpi=300)
plt.close()

print(f"[*] All publication charts saved to: {OUTPUT_DIR}")


# ==============================================================================
# 8. GRAD-CAM HEATMAPS (Clinical Explainability — VGG16 Stream A)
# ==============================================================================
print("\n[*] Generating Grad-CAM explainability heatmaps (VGG16 Stream A)...")

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for the VGG16 stream.
    Hooks the last Conv2d layer of stream_a to produce spatial saliency maps
    showing WHICH facial regions drove each emotion prediction.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, output):
        self.activations = output.detach()  # (1, C, H, W)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()  # (1, C, H, W)

    def generate(self, input_tensor, target_class):
        """Returns a normalized CAM (H x W) for the given target class."""
        self.model.eval()
        output = self.model(input_tensor)  # forward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot)   # backward to get gradients at target layer

        # Global Average Pool over spatial dimensions → importance weights per channel
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()    # (H, W)
        cam = F.relu(cam).cpu().numpy()

        # Min-max normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        for h in self._handles:
            h.remove()


# Find the last Conv2d inside stream_a (VGG16 feature extractor)
_last_conv = None
for _module in eval_model.stream_a.modules():
    if isinstance(_module, nn.Conv2d):
        _last_conv = _module  # iterates in order; ends on the last one

if _last_conv is None:
    print("[!] No Conv2d found in stream_a — Grad-CAM skipped.")
else:
    grad_cam = GradCAM(eval_model, _last_conv)

    # Collect one sample per class from the test set
    _seen, _gradcam_samples = set(), []
    for _path, _cls in zip(test_dataset.samples, test_dataset.targets):
        if _cls not in _seen:
            _seen.add(_cls)
            _gradcam_samples.append((_path, _cls))
        if len(_seen) == NUM_CLASSES:
            break

    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(3 * NUM_CLASSES, 6))
    fig.suptitle(
        "Care-FER Grad-CAM: VGG16 Stream Discriminative Facial Regions",
        fontsize=13, fontweight="bold",
    )

    for col, (_img_path, _true_cls) in enumerate(_gradcam_samples):
        # --- Original image ---
        _raw = Image.open(_img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        _tensor = val_transform(_raw).unsqueeze(0).to(device).requires_grad_(True)

        # --- Grad-CAM ---
        _cam = grad_cam.generate(_tensor, target_class=_true_cls)

        # Upsample CAM to full image resolution
        _cam_up = np.array(
            Image.fromarray((_cam * 255).astype(np.uint8)).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR
            )
        ) / 255.0

        # Blend heatmap over original image
        _img_np = np.array(_raw) / 255.0
        _heatmap = plt.cm.jet(_cam_up)[:, :, :3]   # RGBA → RGB via colormap
        _overlay = np.clip(0.5 * _img_np + 0.5 * _heatmap, 0, 1)

        # Top row: original
        axes[0, col].imshow(_raw)
        axes[0, col].set_title(CLASSES[_true_cls].capitalize(), fontsize=10, fontweight="bold")
        axes[0, col].axis("off")

        # Bottom row: Grad-CAM overlay
        axes[1, col].imshow(_overlay)
        axes[1, col].set_title("Grad-CAM", fontsize=9)
        axes[1, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_gradcam_heatmaps.png"), dpi=300, bbox_inches="tight")
    plt.close()
    grad_cam.remove()
    print(f"[*] Grad-CAM heatmaps saved to: {OUTPUT_DIR}/4_gradcam_heatmaps.png")

print("========================================================================================")
print("  Care-FER Evaluation Complete! Ready for Paper Publication.")
print("========================================================================================")
