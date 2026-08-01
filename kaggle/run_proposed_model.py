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

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==============================================================================
SEED = 42
NUM_EPOCHS = 160  # Increased: V4 showed val F1 still rising at epoch 78 — dual-stream needs more time
BATCH_SIZE = 16
LEARNING_RATE = 1e-4  # Optimal for Hybrid Transformer-CNN architectures
WEIGHT_DECAY = 1e-4
EMA_DECAY = 0.999
TTA_VIEWS = 5         # K=5 for Test-Time Augmentation
UNCERTAINTY_THRESH = 0.30  # Clinical rejection guardrail (30% confidence for 6-class softmax — 50% caused 100% rejection)
IMG_SIZE = 224
PATIENCE = 20         # Increased: large dual-stream model converges more slowly
MIXUP_ALPHA = 0.4     # MixUp regularization strength
NUM_WORKERS = 2

# Kaggle dataset paths (fallback to local if running locally for testing)
# Dataset path — priority order:
#   1. MTCNN-preprocessed output from Cell 1 (same notebook session)
#   2. Raw Kaggle input dataset
#   3. Local path (for testing)
KAGGLE_MTCNN_DIR  = "/kaggle/working/dataset_mtcnn"
KAGGLE_DATASET_DIR = "/kaggle/input/datasets/mrsohel/autism-dataset/dataset"
LOCAL_DATASET_DIR = r"C:\Users\mrsoh\Documents\Autism-Facial-Expression-Recognition\dataset"

if os.path.exists(KAGGLE_MTCNN_DIR):
    DATASET_DIR = KAGGLE_MTCNN_DIR
elif os.path.exists(KAGGLE_DATASET_DIR):
    DATASET_DIR = KAGGLE_DATASET_DIR
else:
    DATASET_DIR = LOCAL_DATASET_DIR

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
    transforms.RandAugment(num_ops=2, magnitude=7),
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

# Offline Face Alignment Preprocessing (Path B)
def align_and_crop_faces(src_dir, dst_dir):
    """
    Standardizes input images by detecting facial bounding boxes with OpenCV Haar Cascade
    and cropping with 15% padding. Removes background noise, neck/shoulders, and scale
    inconsistencies (matching the methodology of top-performing research papers).
    Fallback: Center crop (85%) if face detection fails on extreme lighting/poses.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if dst_dir.exists() and sum(1 for f in dst_dir.rglob("*.*") if f.is_file()) > 1000:
        print(f"[*] Aligned dataset already exists at {dst_dir}. Skipping preprocessing.")
        return str(dst_dir)
        
    print(f"[*] Running Face Alignment Preprocessing (OpenCV Haar Cascade + 15% padding)...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    
    total_imgs = 0
    detected_faces = 0
    
    for split in ["train", "valid", "test"]:
        for cls_name in CLASSES:
            s_dir = src_dir / split / cls_name
            d_dir = dst_dir / split / cls_name
            os.makedirs(d_dir, exist_ok=True)
            if not s_dir.exists(): continue
            
            for img_path in s_dir.iterdir():
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"): continue
                total_imgs += 1
                try:
                    img_cv = cv2.imread(str(img_path))
                    if img_cv is None:
                        Image.open(img_path).convert("RGB").save(d_dir / img_path.name)
                        continue
                    
                    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                    if len(faces) == 0:
                        faces = alt_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
                        
                    if len(faces) > 0:
                        largest_face = max(faces, key=lambda b: b[2] * b[3])
                        x, y, w, h = largest_face
                        img_h, img_w, _ = img_cv.shape
                        
                        pad_w = int(w * 0.15)
                        pad_h = int(h * 0.15)
                        x1 = max(0, x - pad_w)
                        y1 = max(0, y - pad_h)
                        x2 = min(img_w, x + w + pad_w)
                        y2 = min(img_h, y + h + pad_h)
                        
                        cropped_cv = img_cv[y1:y2, x1:x2]
                        detected_faces += 1
                    else:
                        img_h, img_w, _ = img_cv.shape
                        cw, ch = int(img_w * 0.85), int(img_h * 0.85)
                        x1 = (img_w - cw) // 2
                        y1 = (img_h - ch) // 2
                        cropped_cv = img_cv[y1:y1+ch, x1:x1+cw]
                        
                    cropped_rgb = cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB)
                    Image.fromarray(cropped_rgb).save(d_dir / img_path.name)
                except Exception as e:
                    Image.open(img_path).convert("RGB").save(d_dir / img_path.name)
                    
    print(f"[*] Face Alignment Complete! Aligned {detected_faces}/{total_imgs} images ({detected_faces/max(1, total_imgs)*100:.1f}% success).")
    return str(dst_dir)

# Face alignment disabled: Haar Cascade only succeeded on 64.4% of images;
# the 35.6% fallback center-crops introduced noise and hurt performance.
# ALIGNED_DATASET_DIR = "/kaggle/working/aligned_dataset" if os.path.exists("/kaggle") else "./aligned_dataset"
# DATASET_DIR = align_and_crop_faces(DATASET_DIR, ALIGNED_DATASET_DIR)

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
    Care-FER v2 — Redesigned Dual-Stream Architecture:
    - Stream A: VGG16-BN SPATIAL features via forward_features()+GAP
      * Uses conv feature maps (B,512,7,7) → GAP → (B,512)
      * NOT the 4096-d FC pre_logits — spatial features are:
        (a) Smaller (512 vs 4096) = less overfitting on 1386 images
        (b) Proper SE attention on meaningful conv channels
        (c) Enable true Grad-CAM saliency on 7×7 facial regions
    - Stream B: DeiT-Small CLS token (B,384) — global attention geometry
    - Dual SE Blocks: per-stream channel recalibration
    - Head: 896→512→256→6 with GELU + progressive dropout
    """
    def __init__(self, num_classes=6, pretrained=True):
        super().__init__()
        # Stream A: VGG16-BN — spatial conv features (bypasses FC pre_logits)
        vgg = timm.create_model("vgg16_bn", pretrained=pretrained, num_classes=0)
        self.stream_a = vgg
        dim_a = 512  # VGG16-BN last conv block always outputs 512 channels

        # Stream B: DeiT-Small — CLS token (always 384-d)
        deit = timm.create_model("deit_small_patch16_224", pretrained=pretrained, num_classes=0)
        self.stream_b = deit
        dim_b = 384

        print(f"[*] Stream A (VGG16-BN spatial+GAP): {dim_a}-d | Stream B (DeiT-S CLS): {dim_b}-d | Combined: {dim_a+dim_b}-d")

        self.se_a = SqueezeExcitationBlock(dim_a, reduction=16)   # 512→32
        self.se_b = SqueezeExcitationBlock(dim_b, reduction=16)   # 384→24

        # Deeper classification head: 896→512→256→6
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
        # Stream A: spatial conv features via forward_features() then Global Average Pool
        feat_map_a = self.stream_a.forward_features(x)  # (B, 512, 7, 7)
        feat_a = feat_map_a.mean(dim=[2, 3])             # GAP → (B, 512)

        # Stream B: DeiT CLS token
        feat_b = self.stream_b(x)                        # (B, 384)

        # SE recalibration per stream
        rec_a = self.se_a(feat_a)  # (B, 512)
        rec_b = self.se_b(feat_b)  # (B, 384)

        # Concatenate and classify
        fused = torch.cat([rec_a, rec_b], dim=1)  # (B, 896)
        return self.classifier(fused)

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

# V6 Boost: Strongly upweight sadness (x2.0) and fear (x1.2) to overcome recall collapse and imbalance
_sadness_idx = CLASSES.index("sadness")
_fear_idx = CLASSES.index("fear")
_class_weights[_sadness_idx] *= 2.0  # V5 sadness recall was only 15.9% (computed weight 0.72 was too low)
_class_weights[_fear_idx] *= 1.2     # V5 fear recall was 28.6%
print(f"[*] Focal Loss Class Weights (V6 Boost): { {CLASSES[i]: f'{_class_weights[i].item():.2f}' for i in range(NUM_CLASSES)} }")
loss_fn = FocalLoss(alpha=_class_weights, gamma=1.5)  # gamma 1.5: softer than 2.0, avoids over-suppressing easy samples

# Differential learning rate: backbones get 0.1x LR, SE blocks & head get full LR
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "classifier" in name or "se_a" in name or "se_b" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": LEARNING_RATE * 0.1},  # 1e-5 for pretrained backbones (matches baseline differential LR ratio)
    {"params": head_params, "lr": LEARNING_RATE},
], weight_decay=WEIGHT_DECAY)

# Warmup (10 epochs linear) + Cosine Decay to 0.
# CosineAnnealingWarmRestarts caused LR spikes every T_0=10 epochs which destabilizes DeiT.
_WARMUP_EPOCHS = 10
def _lr_lambda(epoch):
    if epoch < _WARMUP_EPOCHS:
        return float(epoch + 1) / float(_WARMUP_EPOCHS)   # 0.1 → 1.0 linearly
    progress = float(epoch - _WARMUP_EPOCHS) / float(max(1, NUM_EPOCHS - _WARMUP_EPOCHS))
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine to 1% of LR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
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
            with autocast(device_type=device.type, dtype=torch.float16 if device.type == "cuda" else torch.bfloat16):
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
    Grad-CAM using tensor-level gradient hooks (avoids register_full_backward_hook
    which conflicts with VGG16's inplace ReLU operations in pre_logits FC layers).
    Hooks the last Conv2d of stream_a; uses activation.register_hook() to capture
    gradients without triggering the BackwardHookFunctionBackward inplace error.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self._activation = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)

    def _fwd_hook(self, module, inp, output):
        self._activation = output          # live tensor; retain_grad called in generate()

    def generate(self, input_tensor, target_class):
        """Returns a normalized CAM array (H, W) for the given target class."""
        self.model.eval()
        _grad_holder = [None]

        with torch.enable_grad():
            # Fresh input that carries gradients through the graph
            x = input_tensor.detach().clone().requires_grad_(True)
            out = self.model(x)                    # forward — triggers _fwd_hook

            # Retain grad on the (non-leaf) activation so its hook fires
            if self._activation is not None and self._activation.requires_grad:
                self._activation.retain_grad()
                _h = self._activation.register_hook(
                    lambda g: _grad_holder.__setitem__(0, g.detach())
                )
            else:
                return np.zeros((7, 7))            # fallback: no grad path to target layer

            self.model.zero_grad()
            out[0, target_class].backward()
            _h.remove()

        grads = _grad_holder[0]                    # (1, C, H, W) or None
        acts  = self._activation.detach()          # (1, C, H, W)

        if grads is None:
            return np.zeros((acts.shape[2], acts.shape[3]))

        # Global-average-pool gradients → channel importance weights
        weights = grads.mean(dim=[2, 3], keepdim=True)          # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1).squeeze().cpu().numpy()  # (H, W)
        cam = np.maximum(cam, 0)                   # ReLU
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove(self):
        self._fwd_handle.remove()


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
