import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.amp import GradScaler

from dataset import get_dataloaders, CLASS_NAMES, NUM_CLASSES, get_class_counts
from models import get_model, count_parameters
from losses import get_loss_fn
from utils import EMA, mixup_data, mixup_criterion, AverageMeter, TrainingLogger, get_device, save_checkpoint
from evaluate import plot_confusion_matrix, plot_training_curves, plot_per_class_f1
from utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a facial expression recognition model")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--loss", type=str, default="ce_smooth", choices=["ce", "ce_smooth", "focal"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--data-dir", type=str, default="master_dataset_split")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--mixup", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema, device, use_mixup):
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_mixup:
            images, y_a, y_b, lam = mixup_data(images, labels)

        optimizer.zero_grad(set_to_none=True)

        if device.type == "xpu":
            with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
                outputs = model(images)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        elif device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                if use_mixup:
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if ema:
            ema.update()

        _, predicted = outputs.max(1)
        if not use_mixup:
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        else:
            correct += (lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item())
            total += labels.size(0)

        loss_meter.update(loss.item(), images.size(0))

    scheduler.step()
    acc = correct / total if total > 0 else 0.0
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, ema=None):
    if ema:
        ema.apply_shadow()

    model.eval()
    loss_meter = AverageMeter()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        loss_meter.update(loss.item(), images.size(0))

    if ema:
        ema.restore()

    all_preds = [int(p) for p in all_preds]
    all_labels = [int(l) for l in all_labels]
    return loss_meter.avg, all_preds, all_labels


def main():
    args = parse_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    device = get_device()
    print(f"\n{'='*60}")
    print(f"  MODEL: {args.model} | DEVICE: {device} | LOSS: {args.loss}")
    print(f"  EPOCHS: {args.epochs} | BATCH: {args.batch_size} | LR: {args.lr}")
    print(f"  MIXUP: {args.mixup} | EMA: {args.ema}")
    print(f"{'='*60}\n")

    # Model
    model, input_size = get_model(args.model, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")

    if input_size != 224:
        print(f"  Note: Model expects input size {input_size}x{input_size}")

    # Data
    train_loader, val_loader, test_loader, class_weights, train_ds, val_ds, test_ds = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, img_size=input_size, num_workers=args.num_workers,
    )
    class_weights = class_weights.to(device)

    train_counts = get_class_counts(train_ds)
    print("Train class distribution:")
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        print(f"  {cls_name}: {train_counts.get(cls_idx, 0)}")

    # Loss (Don't double weight since we use WeightedRandomSampler)
    criterion = get_loss_fn(args.loss, class_weights=None)

    # Optimizer: lower LR for pretrained backbone
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "head" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * 0.1},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # EMA
    ema = EMA(model, decay=0.999) if args.ema else None

    # Logger
    model_results_dir = os.path.join(args.results_dir, args.model)
    logger = TrainingLogger(os.path.join(model_results_dir, "logs"))

    # Training loop
    best_val_f1 = 0.0
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        lr = optimizer.param_groups[-1]["lr"]

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, ema, device, args.mixup,
        )
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device, ema)

        from utils import compute_metrics
        val_metrics = compute_metrics(val_labels, val_preds, CLASS_NAMES)

        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_metrics["accuracy"], lr)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1_macro']:.4f} | "
            f"LR: {lr:.6f}"
        )

        # Save best model
        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "ema_state_dict": ema.shadow if ema else None,
                "best_metric": best_val_f1,
                "args": vars(args),
            }, model_results_dir, "best_model.pth")
            print(f"  >> New best! Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Save training history
    logger.save(args.model)
    plot_training_curves(logger.history, os.path.join(model_results_dir, "plots"), args.model)

    # Final evaluation on test set with best model
    print(f"\n{'='*60}")
    print(f"  TEST SET EVALUATION - {args.model}")
    print(f"{'='*60}")

    # Load best checkpoint
    best_ckpt = torch.load(os.path.join(model_results_dir, "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if ema and best_ckpt.get("ema_state_dict"):
        ema.shadow = best_ckpt["ema_state_dict"]

    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device, ema)
    test_metrics = compute_metrics(test_labels, test_preds, CLASS_NAMES)

    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  F1-Macro:  {test_metrics['f1_macro']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"\n{test_metrics['classification_report']}")

    # Save plots
    plots_dir = os.path.join(model_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_confusion_matrix(test_labels, test_preds, os.path.join(plots_dir, f"{args.model}_cm.png"), args.model)
    plot_per_class_f1(test_metrics["per_class_f1"], plots_dir, args.model)

    # Save test metrics
    import json
    metrics_path = os.path.join(model_results_dir, f"{args.model}_test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to: {model_results_dir}")
    print(f"  - best_model.pth")
    print(f"  - {args.model}_test_metrics.json")
    print(f"  - plots/{args.model}_cm.png")
    print(f"  - plots/{args.model}_f1_per_class.png")
    print(f"  - plots/{args.model}_curves.png")

    return test_metrics


if __name__ == "__main__":
    main()
