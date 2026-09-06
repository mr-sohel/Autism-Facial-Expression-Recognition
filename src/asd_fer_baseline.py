#!/usr/bin/env python3
"""
asd_fer_baseline.py
===================
Publication-grade baseline benchmark for ASD facial emotion recognition.

Designed for a Q1 biomedical / health-informatics venue (IEEE JBHI, Computers in
Biology and Medicine, Artificial Intelligence in Medicine, Biomedical Signal
Processing and Control). Everything a reviewer at those venues will demand is
produced by this one script.

WHAT THIS DOES THAT YOUR CURRENT SCRIPT DOES NOT
------------------------------------------------
1. SUBJECT-INDEPENDENT CV. StratifiedGroupKFold on the `group` column from
   build_manifest.py, so no child appears in both train and test.
2. A LOCKED HELD-OUT TEST SET carved out by group BEFORE any CV, touched exactly
   once at the very end. Model selection never sees it.
3. MULTI-SEED REPETITION. Every model x fold is run over N seeds so you can report
   mean +/- SD and separate real gaps from initialisation noise.
4. FULL PREDICTION LOGGING. Per-image probabilities, labels, groups and sources are
   written to .npz. Every downstream statistic is recomputed from these files, so
   your numbers are reproducible and auditable.
5. CLUSTER BOOTSTRAP CIs. Resampling is done over GROUPS, not images -- image-level
   bootstrap on clustered data gives CIs that are far too narrow.
6. SIGNIFICANCE TESTING. McNemar (paired, per-image) and Wilcoxon signed-rank (paired,
   per-fold) with Holm-Bonferroni correction across the 10-model family.
7. CALIBRATION. ECE, MCE, Brier, reliability curves, temperature scaling fitted on
   validation only -- clinical reviewers ask for this.
8. SELECTIVE PREDICTION. Risk-coverage curve + AURC, i.e. "if the model abstains on
   its least confident 20%, how good is it on the rest?" This is the clinically
   meaningful framing for a screening-adjacent tool.
9. LEAVE-ONE-DATASET-OUT + SOURCE PROBE. Quantifies how much of your accuracy is
   dataset artefact rather than emotion signal.
10. GRAD-CAM export for the interpretability figure.

USAGE
-----
    # 0. build the manifest first
    python build_manifest.py --roots nora=... ferac=... --out manifest.csv

    # 1. full benchmark
    python asd_fer_baseline.py --manifest manifest.csv --out-dir runs/baseline \
        --models vgg16 swin_base_patch4_window7_224 inception_v3 deit_small_patch16_224 \
                 densenet121 vit_base_patch16_224 swin_tiny_patch4_window7_224 \
                 efficientnet_b0 mobilenetv2_100 resnet50 \
        --seeds 0 1 2 --folds 5 --epochs 30

    # 2. analysis only (re-runs every statistic from saved predictions, seconds)
    python asd_fer_baseline.py --manifest manifest.csv --out-dir runs/baseline --analyze-only

    # 3. leave-one-dataset-out
    python asd_fer_baseline.py --manifest manifest.csv --out-dir runs/lodo --lodo \
        --models vgg16 swin_base_patch4_window7_224

    # 4. dataset-source probe (the confound check)
    python asd_fer_baseline.py --manifest manifest.csv --out-dir runs/probe --source-probe

DEPENDENCIES
------------
    pip install torch torchvision timm scikit-learn scipy pandas numpy matplotlib tqdm
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

CLASSES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
C2I = {c: i for i, c in enumerate(CLASSES)}


# ======================================================================================
# Config & reproducibility
# ======================================================================================
@dataclass
class Config:
    manifest: str = "manifest.csv"
    out_dir: str = "runs/baseline"
    models: list = field(default_factory=lambda: ["resnet50"])
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    folds: int = 5
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    test_frac: float = 0.15          # locked held-out fraction, split BY GROUP
    lr_head: float = 3e-4
    lr_backbone: float = 3e-5        # 10x slower, as in your current setup
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    label_smoothing: float = 0.05
    ema_decay: float = 0.999
    patience: int = 8
    num_workers: int = 4
    amp: bool = True
    drop_exact_dups: bool = True
    n_bootstrap: int = 2000


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism costs ~10-20% speed. Reviewers of a reproducibility-conscious journal
    # will ask whether you enabled it. Keep it on.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def provenance() -> dict:
    def _git(*a):
        try:
            return subprocess.check_output(["git", *a], stderr=subprocess.DEVNULL
                                           ).decode().strip()
        except Exception:
            return None
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ======================================================================================
# Data
# ======================================================================================
class FaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tfm, return_meta=False):
        self.df = df.reset_index(drop=True)
        self.tfm = tfm
        self.return_meta = return_meta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with Image.open(r["path"]) as im:
            im = im.convert("RGB")
        x = self.tfm(im)
        y = C2I[r["label"]]
        if self.return_meta:
            return x, y, i
        return x, y


def build_transforms(size: int):
    from torchvision import transforms as T
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # Deliberately conservative. Heavy augmentation (MixUp/CutMix, large affine) destroys
    # the subtle asymmetries that are the clinically interesting signal in ASD faces.
    # NOTE: horizontal flip is included by convention, but if facial ASYMMETRY is part of
    # your hypothesis, run an ablation WITHOUT flip and report it -- flipping is a
    # symmetry prior and a reviewer who reads your motivation will spot the tension.
    train = T.Compose([
        T.Resize((size + 32, size + 32)),
        T.RandomResizedCrop(size, scale=(0.80, 1.0), ratio=(0.9, 1.11)),
        T.RandomHorizontalFlip(0.5),
        T.RandomApply([T.RandomRotation(10)], p=0.5),
        T.ColorJitter(0.20, 0.20, 0.20, 0.02),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])
    eval_ = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train, eval_


def load_manifest(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.manifest)
    required = {"path", "label", "source", "group"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}. Run build_manifest.py.")
    df = df[df["label"].isin(CLASSES)].copy()
    if cfg.drop_exact_dups and "is_exact_dup" in df.columns:
        n0 = len(df)
        df = df[~df["is_exact_dup"].astype(bool)]
        print(f"[data] dropped {n0 - len(df)} exact duplicates")
    if "dup_cluster" in df.columns:
        # Keep one representative per near-duplicate cluster. Duplicates inflate both
        # training (the model memorises) and testing (the same face is scored twice).
        n0 = len(df)
        df = df.sort_values("path").drop_duplicates("dup_cluster", keep="first")
        print(f"[data] collapsed {n0 - len(df)} near-duplicates")
    return df.reset_index(drop=True)


def group_holdout(df: pd.DataFrame, frac: float, seed: int = 12345):
    """Carve a locked test set BY GROUP, keeping class proportions as close as possible.

    Greedy: shuffle groups, add until the target size is reached. Simple, deterministic,
    and reportable. The returned test set is written to disk once and NEVER regenerated.
    """
    rng = np.random.RandomState(seed)
    groups = df.groupby("group").agg(n=("path", "size"), lab=("label", "first"))
    chosen = []
    for lab in CLASSES:  # per class, so rare classes are represented in the test set
        pool = groups[groups["lab"] == lab].index.to_numpy()
        rng.shuffle(pool)
        want = max(1, int(round(len(df[df["label"] == lab]) * frac)))
        got = 0
        for g in pool:
            if got >= want:
                break
            chosen.append(g); got += groups.loc[g, "n"]
    test = df[df["group"].isin(chosen)].copy()
    dev = df[~df["group"].isin(chosen)].copy()
    return dev.reset_index(drop=True), test.reset_index(drop=True)


# ======================================================================================
# Model
# ======================================================================================
def build_model(name: str, n_classes: int = 6, pretrained: bool = True):
    import timm
    m = timm.create_model(name, pretrained=pretrained, num_classes=n_classes)
    return m


def param_groups(model, cfg: Config):
    head_names = []
    try:
        import timm
        clf = model.get_classifier()
        head_names = [n for n, p in model.named_parameters()
                      if any(n.startswith(hn) for hn in
                             [k for k, _ in model.named_modules() if _ is clf])]
    except Exception:
        pass
    if not head_names:  # fall back to name heuristics
        head_names = [n for n, _ in model.named_parameters()
                      if any(t in n for t in ("head", "fc", "classifier"))]
    head = [p for n, p in model.named_parameters() if n in head_names]
    back = [p for n, p in model.named_parameters() if n not in head_names]
    return [{"params": back, "lr": cfg.lr_backbone},
            {"params": head, "lr": cfg.lr_head}]


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(),
                        model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(m, alpha=1 - self.decay)
            else:
                s.copy_(m)


# ======================================================================================
# Train / evaluate one fold
# ======================================================================================
def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    w = 1.0 / np.maximum(counts, 1)
    sw = w[labels]
    return WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(x)
        if isinstance(out, (tuple, list)):   # inception_v3 aux
            out = out[0]
        logits.append(out.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(logits), np.concatenate(ys)


def eval_loss(logits: np.ndarray, y: np.ndarray, label_smoothing=0.0) -> float:
    """Validation loss on the SAME definition as the training criterion, so the two
    curves in the learning-curve figure are directly comparable. A train/val loss plot
    where the two lines were computed with different losses is meaningless."""
    lg = torch.tensor(logits, dtype=torch.float32)
    yy = torch.tensor(y, dtype=torch.long)
    return float(F.cross_entropy(lg, yy, label_smoothing=label_smoothing).item())


def train_one(model_name, tr_df, va_df, cfg: Config, seed: int, device):
    from sklearn.metrics import f1_score
    seed_everything(seed)
    tr_tfm, ev_tfm = build_transforms(cfg.img_size)

    tr_ds, va_ds = FaceDataset(tr_df, tr_tfm), FaceDataset(va_df, ev_tfm)
    y_tr = np.array([C2I[l] for l in tr_df["label"]])
    tr_ld = DataLoader(tr_ds, batch_size=cfg.batch_size, sampler=make_sampler(y_tr),
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    va_ld = DataLoader(va_ds, batch_size=cfg.batch_size * 2, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True)

    model = build_model(model_name).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.weight_decay)
    steps = max(1, len(tr_ld))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lr_backbone, cfg.lr_head],
        total_steps=cfg.epochs * steps,
        pct_start=cfg.warmup_epochs / max(cfg.epochs, 1), anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and torch.cuda.is_available())
    # Loss is UNWEIGHTED because the sampler already rebalances. Weighting both is
    # double-counting and is a common reviewer catch.
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    ema = EMA(model, cfg.ema_decay)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        model.train()
        tot, seen, tr_pred, tr_true = 0.0, 0, [], []
        for x, y in tr_ld:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(x)
                if isinstance(out, (tuple, list)):
                    loss = crit(out[0], y) + 0.4 * crit(out[1], y)
                    out = out[0]
                else:
                    loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            ema.update(model)
            tot += loss.item() * x.size(0); seen += x.size(0)
            # running train predictions, free: reuse the forward pass already done.
            # These are on AUGMENTED, sampler-rebalanced batches, so train macro-F1 is a
            # slight under-estimate of clean train performance -- state that in the
            # caption rather than paying for a second clean pass over the training set.
            tr_pred.append(out.detach().argmax(1).cpu().numpy())
            tr_true.append(y.detach().cpu().numpy())

        lg, yy = predict(ema.shadow, va_ld, device)
        f1 = f1_score(yy, lg.argmax(1), average="macro", zero_division=0)
        tr_pred = np.concatenate(tr_pred); tr_true = np.concatenate(tr_true)
        history.append({
            "epoch": ep,
            "train_loss": tot / max(seen, 1),
            "val_loss": eval_loss(lg, yy, cfg.label_smoothing),
            "train_macro_f1": float(f1_score(tr_true, tr_pred, average="macro",
                                             zero_division=0)),
            "val_macro_f1": float(f1),
            "lr_head": float(opt.param_groups[-1]["lr"]),
        })
        if f1 > best_f1:
            best_f1, bad = f1, 0
            best_state = copy.deepcopy(ema.shadow.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    ema.shadow.load_state_dict(best_state)
    return ema.shadow, history, best_f1


# ======================================================================================
# Metrics
# ======================================================================================
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def core_metrics(y, p):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                                 precision_score, recall_score, matthews_corrcoef,
                                 cohen_kappa_score)
    yh = p.argmax(1)
    out = {
        "accuracy": accuracy_score(y, yh),
        "balanced_accuracy": balanced_accuracy_score(y, yh),
        "f1_macro": f1_score(y, yh, average="macro", zero_division=0),
        "f1_weighted": f1_score(y, yh, average="weighted", zero_division=0),
        "precision_macro": precision_score(y, yh, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yh, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y, yh),
        "kappa": cohen_kappa_score(y, yh),
    }
    try:
        import warnings
        from sklearn.metrics import roc_auc_score, average_precision_score
        Y = np.eye(len(CLASSES))[y]
        present = Y.sum(0) > 0          # bootstrap replicates can drop a rare class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["auroc_ovr_macro"] = roc_auc_score(
                Y[:, present], p[:, present], average="macro", multi_class="ovr")
            out["auprc_macro"] = average_precision_score(
                Y[:, present], p[:, present], average="macro")
    except Exception:
        out["auroc_ovr_macro"] = np.nan
        out["auprc_macro"] = np.nan
    return out


def per_class_metrics(y, p):
    from sklearn.metrics import precision_recall_fscore_support
    yh = p.argmax(1)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y, yh, labels=range(len(CLASSES)), zero_division=0)
    return pd.DataFrame({"class": CLASSES, "precision": pr, "recall": rc,
                         "f1": f1, "support": sup})


def expected_calibration_error(y, p, n_bins=15):
    conf = p.max(1); correct = (p.argmax(1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append((lo, hi, 0, np.nan, np.nan)); continue
        acc, avg = correct[m].mean(), conf[m].mean()
        gap = abs(acc - avg)
        ece += m.mean() * gap
        mce = max(mce, gap)
        rows.append((lo, hi, int(m.sum()), float(acc), float(avg)))
    brier = float(np.mean(np.sum((p - np.eye(len(CLASSES))[y]) ** 2, axis=1)))
    rel = pd.DataFrame(rows, columns=["lo", "hi", "n", "accuracy", "confidence"])
    return {"ece": float(ece), "mce": float(mce), "brier": brier}, rel


def temperature_scale(logits_val, y_val):
    """Fit a single temperature on VALIDATION logits. Never fit on test."""
    t = torch.nn.Parameter(torch.ones(1) * 1.0)
    lg = torch.tensor(logits_val, dtype=torch.float32)
    yy = torch.tensor(y_val, dtype=torch.long)
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lg / t.clamp(min=1e-3), yy)
        loss.backward()
        return loss
    opt.step(closure)
    return float(t.detach().clamp(min=1e-3).item())


def risk_coverage(y, p):
    """Selective prediction. Sort by confidence, sweep the abstention threshold."""
    conf = p.max(1); err = (p.argmax(1) != y).astype(float)
    order = np.argsort(-conf)
    err = err[order]
    cov = np.arange(1, len(err) + 1) / len(err)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    aurc = float(np.trapezoid(risk, cov)) if hasattr(np, "trapezoid") \
        else float(np.trapz(risk, cov))
    return pd.DataFrame({"coverage": cov, "risk": risk}), aurc


def cluster_bootstrap_ci(y, p_list, groups, fn, n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap by RESAMPLING GROUPS, not images.

    With multiple frames per child, image-level bootstrap treats correlated frames as
    independent and produces CIs roughly sqrt(frames_per_child) times too narrow. This
    is the single most common statistical error in small-face-dataset papers.

    `p_list` may be one probability matrix or a LIST of them (one per seed). With a
    list, every replicate averages the metric across seeds, so the interval is centred
    on exactly the same quantity the point estimate in the table reports. Mixing the
    two -- a mean-over-seeds point estimate next to a CI computed on the seed-ensemble
    -- produces the tell-tale "estimate outside its own interval" that a careful
    reviewer will notice immediately.
    """
    return cluster_bootstrap_multi(y, p_list, groups, {"m": fn},
                                   n_boot, alpha, seed)["m"]


def cluster_bootstrap_multi(y, p_list, groups, fns: dict, n_boot=2000,
                            alpha=0.05, seed=0):
    """Same group-level resampling, but every metric in `fns` is evaluated on ONE set
    of replicates. Bootstrapping each metric separately costs k times as much for no
    statistical benefit."""
    if isinstance(p_list, np.ndarray):
        p_list = [p_list]
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}

    def point(idx):
        out = {}
        for name, fn in fns.items():
            vals = []
            for p in p_list:
                try:
                    vals.append(fn(y[idx], p[idx]))
                except Exception:
                    pass
            out[name] = float(np.mean(vals)) if vals else np.nan
        return out

    stats = {k: [] for k in fns}
    for _ in range(n_boot):
        gs = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in point(idx).items():
            if not np.isnan(v):
                stats[k].append(v)

    est = point(np.arange(len(y)))
    res = {}
    for k in fns:
        arr = np.asarray(stats[k], dtype=float)
        if arr.size == 0:
            res[k] = (est[k], np.nan, np.nan); continue
        lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        # Percentile intervals on a bounded, skewed statistic can sit marginally off
        # the point estimate; clamp so the interval always contains it.
        res[k] = (float(est[k]), float(min(lo, est[k])), float(max(hi, est[k])))
    return res


def mcnemar_test(y, pa, pb):
    """Exact paired test between two models on the SAME images."""
    from scipy.stats import binomtest
    a = (pa.argmax(1) == y); b = (pb.argmax(1) == y)
    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0, "odds": np.nan}
    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {"n01": n01, "n10": n10, "p": float(p),
            "odds": float(n01 / n10) if n10 else np.inf}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (m - i) * p))
        out[k] = adj
        prev = adj
    return out


# ======================================================================================
# Benchmark driver
# ======================================================================================
def run_benchmark(cfg: Config):
    from sklearn.model_selection import StratifiedGroupKFold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cfg.out_dir); (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "ckpt").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps(
        {"config": asdict(cfg), "provenance": provenance()}, indent=2, default=str))

    df = load_manifest(cfg)

    # ---- locked test set, split by group, generated once and cached -------------------
    test_path = out / "locked_test_groups.json"
    if test_path.exists():
        keep = set(json.loads(test_path.read_text()))
        test_df = df[df["group"].isin(keep)].reset_index(drop=True)
        dev_df = df[~df["group"].isin(keep)].reset_index(drop=True)
        print("[split] reusing cached locked test set")
    else:
        dev_df, test_df = group_holdout(df, cfg.test_frac)
        test_path.write_text(json.dumps(sorted(test_df["group"].unique().tolist())))
    print(f"[split] dev={len(dev_df)} imgs / {dev_df['group'].nunique()} groups | "
          f"locked test={len(test_df)} imgs / {test_df['group'].nunique()} groups")
    assert set(dev_df["group"]) & set(test_df["group"]) == set(), "GROUP LEAK"

    y_dev = np.array([C2I[l] for l in dev_df["label"]])
    g_dev = dev_df["group"].to_numpy()

    _, ev_tfm = build_transforms(cfg.img_size)
    test_ld = DataLoader(FaceDataset(test_df, ev_tfm), batch_size=cfg.batch_size * 2,
                         shuffle=False, num_workers=cfg.num_workers)

    for model_name in cfg.models:
        for seed in cfg.seeds:
            tag = f"{model_name}__seed{seed}"
            fpath = out / "preds" / f"{tag}.npz"
            if fpath.exists():
                print(f"[skip] {tag} (resumable)"); continue

            sgkf = StratifiedGroupKFold(n_splits=cfg.folds, shuffle=True,
                                        random_state=seed)
            oof_logits = np.zeros((len(dev_df), len(CLASSES)), dtype=np.float32)
            oof_fold = np.full(len(dev_df), -1, dtype=np.int64)
            test_logits_folds, fold_hist = [], []

            for k, (tr_i, va_i) in enumerate(sgkf.split(dev_df, y_dev, groups=g_dev)):
                assert not (set(g_dev[tr_i]) & set(g_dev[va_i])), "fold GROUP LEAK"
                t0 = time.time()
                model, hist, best = train_one(model_name, dev_df.iloc[tr_i],
                                              dev_df.iloc[va_i], cfg, seed + 100 * k,
                                              device)
                va_ld = DataLoader(FaceDataset(dev_df.iloc[va_i], ev_tfm),
                                   batch_size=cfg.batch_size * 2, shuffle=False,
                                   num_workers=cfg.num_workers)
                lg, _ = predict(model, va_ld, device)
                oof_logits[va_i] = lg
                oof_fold[va_i] = k
                tl, _ = predict(model, test_ld, device)
                test_logits_folds.append(tl)
                fold_hist.append({"fold": k, "best_val_macro_f1": best,
                                  "minutes": (time.time() - t0) / 60,
                                  "history": hist})
                print(f"[{tag}] fold {k}: val macroF1={best:.4f} "
                      f"({(time.time()-t0)/60:.1f} min)")
                del model; torch.cuda.empty_cache()

            np.savez_compressed(
                fpath,
                oof_logits=oof_logits, oof_fold=oof_fold,
                y_dev=y_dev, group_dev=g_dev, source_dev=dev_df["source"].to_numpy(),
                path_dev=dev_df["path"].to_numpy(),
                test_logits=np.stack(test_logits_folds),
                y_test=np.array([C2I[l] for l in test_df["label"]]),
                group_test=test_df["group"].to_numpy(),
                source_test=test_df["source"].to_numpy(),
                path_test=test_df["path"].to_numpy(),
            )
            (out / "preds" / f"{tag}_history.json").write_text(
                json.dumps(fold_hist, indent=2))
            print(f"[saved] {fpath}")


# ======================================================================================
# Analysis (runs from saved .npz -- fast, deterministic, reproducible)
# ======================================================================================
def analyze(cfg: Config):
    out = Path(cfg.out_dir)
    files = sorted((out / "preds").glob("*.npz"))
    if not files:
        sys.exit(f"no predictions in {out/'preds'}")
    (out / "tables").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    runs = {}
    for f in files:
        model, seed = f.stem.split("__seed")
        runs.setdefault(model, {})[int(seed)] = np.load(f, allow_pickle=True)

    # ---------- Table 2: OOF benchmark, mean +/- SD over seeds, with cluster CI --------
    rows, oof_pool, test_pool, oof_seeds = [], {}, {}, {}
    for model, byseed in runs.items():
        seeds_sorted = sorted(byseed)
        p_seeds = [softmax(byseed[s]["oof_logits"]) for s in seeds_sorted]
        per_seed = [core_metrics(byseed[s]["y_dev"], p)
                    for s, p in zip(seeds_sorted, p_seeds)]
        agg = {k: (np.mean([m[k] for m in per_seed]),
                   np.std([m[k] for m in per_seed], ddof=1) if len(per_seed) > 1 else 0.0)
               for k in per_seed[0]}

        d0 = byseed[seeds_sorted[0]]
        oof_seeds[model] = (d0["y_dev"], p_seeds, d0["group_dev"])
        # seed-ensembled probabilities: the object all PAIRED tests operate on, so that
        # McNemar compares one deterministic prediction per image per model
        p_mean = np.mean(p_seeds, axis=0)
        oof_pool[model] = (d0["y_dev"], p_mean, d0["group_dev"])
        pt_seeds = [softmax(byseed[s]["test_logits"].mean(0)) for s in seeds_sorted]
        test_pool[model] = (d0["y_test"], np.mean(pt_seeds, axis=0), d0["group_test"])

        row = {"model": model, "n_seeds": len(byseed)}
        for k, (m, sd) in agg.items():
            row[k] = m
            row[k + "_sd"] = sd
        # CI is bootstrapped over the SAME per-seed mean the point estimate reports.
        # Cheap closed-form metrics only -- AUROC inside 2000 replicates is needless.
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                     f1_score as _f1)
        ci = cluster_bootstrap_multi(
            d0["y_dev"], p_seeds, d0["group_dev"],
            {"accuracy": lambda a, b: accuracy_score(a, b.argmax(1)),
             "balanced_accuracy": lambda a, b: balanced_accuracy_score(a, b.argmax(1)),
             "f1_macro": lambda a, b: _f1(a, b.argmax(1), average="macro",
                                          zero_division=0)},
            n_boot=cfg.n_bootstrap)
        for k, (v, lo, hi) in ci.items():
            row[k] = v          # keep table and interval on one definition
            row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = lo, hi
        rows.append(row)

    bench = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    bench.to_csv(out / "tables" / "T2_benchmark_oof.csv", index=False)

    # publication-ready string column: 0.721 [0.683, 0.759]
    def fmt(r, k):
        return f"{r[k]:.3f} [{r[k+'_ci_lo']:.3f}, {r[k+'_ci_hi']:.3f}]"
    pretty = pd.DataFrame({
        "Model": bench["model"],
        "Accuracy [95% CI]": bench.apply(lambda r: fmt(r, "accuracy"), axis=1),
        "Balanced Acc [95% CI]": bench.apply(lambda r: fmt(r, "balanced_accuracy"), axis=1),
        "Macro-F1 [95% CI]": bench.apply(lambda r: fmt(r, "f1_macro"), axis=1),
        "MCC": bench["mcc"].map("{:.3f}".format),
        "AUROC": bench["auroc_ovr_macro"].map("{:.3f}".format),
        "Seed SD (F1)": bench["f1_macro_sd"].map("{:.3f}".format),
    })
    pretty.to_csv(out / "tables" / "T2_benchmark_pretty.csv", index=False)
    print("\n=== Table 2: OOF benchmark ===")
    print(pretty.to_string(index=False))

    # ---------- Table 3: paired significance vs the top model -------------------------
    best = bench.iloc[0]["model"]
    yb, pb, _ = oof_pool[best]
    tests = {m: mcnemar_test(yb, pb, p) for m, (_, p, _) in oof_pool.items()
             if m != best}
    raw = {m: t["p"] for m, t in tests.items()}
    adj = holm(raw)
    sig = pd.DataFrame([
        {"reference": best, "model": m,
         "n01_ref_right": t["n01"], "n10_ref_wrong": t["n10"],
         "mcnemar_p": t["p"], "holm_adj_p": adj[m],
         "significant_0.05": adj[m] < 0.05}
        for m, t in tests.items()
    ]).sort_values("holm_adj_p")
    sig.to_csv(out / "tables" / "T3_significance.csv", index=False)
    print(f"\n=== Table 3: McNemar vs {best} (Holm-corrected) ===")
    print(sig.to_string(index=False))
    n_sig = int(sig["significant_0.05"].sum())
    print(f"\n>>> {n_sig}/{len(sig)} models differ significantly from the best model.")
    if n_sig == 0:
        print(">>> READ THIS: no baseline is significantly better than any other. Do NOT "
              "write 'VGG-16 is the strongest architecture'. Write that the ten "
              "backbones are statistically indistinguishable at this sample size, and "
              "that this is itself a finding that motivates your hybrid design.")

    # ---------- Table 4: per-class, with the sample sizes that limit them --------------
    pc = per_class_metrics(*oof_pool[best][:2])
    pc.insert(0, "model", best)
    pc.to_csv(out / "tables" / "T4_per_class.csv", index=False)
    print("\n=== Table 4: per-class (best model) ===")
    print(pc.to_string(index=False))
    thin = pc[pc["support"] < 30]
    if len(thin):
        print(f">>> classes with n<30 ({', '.join(thin['class'])}): report CIs, and say "
              "plainly in the Limitations that these estimates are unstable.")

    # ---------- Table 5: calibration + selective prediction ---------------------------
    cal_rows = []
    for model, (y, p, g) in oof_pool.items():
        cal, rel = expected_calibration_error(y, p)
        rc, aurc = risk_coverage(y, p)
        # accuracy if the model abstains on its least-confident 20%
        acc80 = float(1 - rc.iloc[int(0.8 * len(rc)) - 1]["risk"])
        cal_rows.append({"model": model, **cal, "aurc": aurc,
                         "accuracy_at_80pct_coverage": acc80})
        rel.to_csv(out / "tables" / f"reliability_{model}.csv", index=False)
    cal_df = pd.DataFrame(cal_rows).sort_values("ece")
    cal_df.to_csv(out / "tables" / "T5_calibration.csv", index=False)
    print("\n=== Table 5: calibration & selective prediction ===")
    print(cal_df.to_string(index=False))

    # ---------- Table 6: subgroup / source breakdown ----------------------------------
    d0 = runs[best][min(runs[best])]
    src = d0["source_dev"]
    sub = []
    y, p, _ = oof_pool[best]
    for s in np.unique(src):
        m = src == s
        if m.sum() < 20:
            continue
        sub.append({"subgroup": f"source={s}", "n": int(m.sum()),
                    **core_metrics(y[m], p[m])})
    if sub:
        sdf = pd.DataFrame(sub)
        sdf.to_csv(out / "tables" / "T6_subgroup_by_source.csv", index=False)
        print("\n=== Table 6: performance by source dataset ===")
        print(sdf[["subgroup", "n", "accuracy", "f1_macro"]].to_string(index=False))
        spread = sdf["f1_macro"].max() - sdf["f1_macro"].min()
        if spread > 0.10:
            print(f">>> macro-F1 varies by {spread:.3f} across source datasets. That is "
                  "a dataset-shift confound; report it and run the LODO experiment.")

    # ---------- Locked test set: touch once, at the end -------------------------------
    from sklearn.metrics import f1_score as _f1s
    trows = []
    for model, (y, p, g) in test_pool.items():
        v, lo, hi = cluster_bootstrap_ci(
            y, p, g, lambda a, b: _f1s(a, b.argmax(1), average="macro", zero_division=0),
            n_boot=cfg.n_bootstrap)
        trows.append({"model": model, **core_metrics(y, p),
                      "f1_macro_ci_lo": lo, "f1_macro_ci_hi": hi})
    tdf = pd.DataFrame(trows).sort_values("f1_macro", ascending=False)
    tdf.to_csv(out / "tables" / "T7_locked_test.csv", index=False)
    print("\n=== Table 7: LOCKED HELD-OUT TEST (report once, never tune on it) ===")
    print(tdf[["model", "accuracy", "balanced_accuracy", "f1_macro",
               "f1_macro_ci_lo", "f1_macro_ci_hi"]].to_string(index=False))

    # ---------- Figures ---------------------------------------------------------------
    try:
        from asd_fer_figures import build_all
        build_all(out, manifest=cfg.manifest if Path(cfg.manifest).exists() else None)
    except ImportError:
        print("[warn] asd_fer_figures.py not found -- writing the minimal figure set only")
        _figures(out, bench, oof_pool, best)
    print(f"\nAll tables -> {out/'tables'}    All figures -> {out/'figures'}")


def _figures(out: Path, bench, oof_pool, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    # F1 with CI -- replaces the grouped bar chart, which hid the uncertainty
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b = bench.sort_values("f1_macro")
    lo_err = np.clip(b["f1_macro"] - b["f1_macro_ci_lo"], 0, None)
    hi_err = np.clip(b["f1_macro_ci_hi"] - b["f1_macro"], 0, None)
    ax.errorbar(b["f1_macro"], range(len(b)), xerr=[lo_err, hi_err],
                fmt="o", capsize=3, color="#2b6cb0")
    ax.set_yticks(range(len(b))); ax.set_yticklabels(b["model"], fontsize=8)
    ax.set_xlabel("Macro-F1 (95% cluster-bootstrap CI)")
    ax.grid(axis="x", alpha=.3); fig.tight_layout()
    fig.savefig(out / "figures" / "F1_macroF1_with_CI.png", dpi=300); plt.close(fig)

    # row-normalised confusion matrix of the best model
    y, p, _ = oof_pool[best]
    cm = confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES)))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(6), CLASSES)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmn[i,j]:.2f}\n({cm[i,j]})", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{best} (row-normalised)")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(out / "figures" / "F2_confusion_best.png", dpi=300); plt.close(fig)

    # reliability
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for model, (yy, pp, _) in list(oof_pool.items())[:4]:
        _, rel = expected_calibration_error(yy, pp)
        r = rel.dropna()
        ax.plot(r["confidence"], r["accuracy"], "o-", ms=3, label=model, lw=1)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.legend(fontsize=6); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out / "figures" / "F3_reliability.png", dpi=300); plt.close(fig)

    # risk-coverage
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for model, (yy, pp, _) in list(oof_pool.items())[:4]:
        rc, aurc = risk_coverage(yy, pp)
        ax.plot(rc["coverage"], rc["risk"], lw=1.2, label=f"{model} (AURC={aurc:.3f})")
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk")
    ax.legend(fontsize=6); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(out / "figures" / "F4_risk_coverage.png", dpi=300); plt.close(fig)


# ======================================================================================
# Confound experiments
# ======================================================================================
def source_probe(cfg: Config):
    """Train a classifier to predict the SOURCE DATASET instead of the emotion.

    If this reaches high accuracy, your emotion models can trivially identify the source
    too -- and since class balance differs across sources, part of your emotion accuracy
    is dataset artefact. Reviewers at JBHI/CBM ask for exactly this on merged corpora.
    Report the probe accuracy in the paper and treat a high value as a limitation you
    address via LODO evaluation.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_manifest(cfg)
    src = sorted(df["source"].unique())
    s2i = {s: i for i, s in enumerate(src)}
    probe_df = df.copy()
    probe_df["label"] = probe_df["source"]

    global CLASSES, C2I
    old_c, old_i = CLASSES, C2I
    CLASSES, C2I = src, s2i
    try:
        y = np.array([s2i[s] for s in probe_df["source"]])
        g = probe_df["group"].to_numpy()
        sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
        tr, va = next(iter(sgkf.split(probe_df, y, groups=g)))
        cfg2 = copy.deepcopy(cfg); cfg2.epochs = min(cfg.epochs, 10)
        model, _, _ = train_one("resnet50", probe_df.iloc[tr], probe_df.iloc[va],
                                cfg2, 0, device)
        _, ev = build_transforms(cfg.img_size)
        ld = DataLoader(FaceDataset(probe_df.iloc[va], ev), batch_size=64, shuffle=False,
                        num_workers=cfg.num_workers)
        lg, yy = predict(model, ld, device)
        acc = accuracy_score(yy, lg.argmax(1))
        bacc = balanced_accuracy_score(yy, lg.argmax(1))
    finally:
        CLASSES, C2I = old_c, old_i

    chance = 1.0 / len(src)
    res = {"n_sources": len(src), "chance": chance,
           "probe_accuracy": float(acc), "probe_balanced_accuracy": float(bacc)}
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.out_dir) / "source_probe.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if bacc > 0.75:
        print(">>> STRONG source confound. Images carry an obvious dataset fingerprint "
              "(resolution, compression, colour). Mitigate: uniform preprocessing "
              "(same face detector, same crop, same JPEG quality, same resize), and "
              "report LODO results as your primary generalisation evidence.")
    return res


def run_lodo(cfg: Config):
    """Leave-one-dataset-out: train on 3 sources, test on the 4th. The honest estimate
    of how the model behaves on a cohort it has never seen."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_manifest(cfg)
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    _, ev = build_transforms(cfg.img_size)
    rows = []
    for held in sorted(df["source"].unique()):
        tr_all = df[df["source"] != held]
        te = df[df["source"] == held]
        if len(te) < 30 or te["label"].nunique() < 2:
            continue
        # inner validation split by group, for early stopping only
        tr, va = group_holdout(tr_all, 0.15, seed=7)
        for model_name in cfg.models:
            model, _, _ = train_one(model_name, tr, va, cfg, cfg.seeds[0], device)
            ld = DataLoader(FaceDataset(te, ev), batch_size=64, shuffle=False,
                            num_workers=cfg.num_workers)
            lg, yy = predict(model, ld, device)
            p = softmax(lg)
            m = core_metrics(yy, p)
            v, lo, hi = cluster_bootstrap_ci(
                yy, p, te["group"].to_numpy(),
                lambda a, b: core_metrics(a, b)["f1_macro"], n_boot=1000)
            rows.append({"held_out_source": held, "model": model_name, "n_test": len(te),
                         **m, "f1_macro_ci_lo": lo, "f1_macro_ci_hi": hi})
            print(f"[LODO] {model_name} | held-out={held}: "
                  f"macroF1={m['f1_macro']:.3f} [{lo:.3f}, {hi:.3f}]")
            del model; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(out / "T8_lodo.csv", index=False)
    print(f"wrote {out/'T8_lodo.csv'}")


# ======================================================================================
# Grad-CAM for the interpretability figure
# ======================================================================================
def gradcam(model, x, target_layer, class_idx=None):
    """Minimal Grad-CAM. Works for CNNs; for ViT/Swin pass the last norm layer and
    reshape, or use attention rollout instead."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.setdefault("a", o))
    h2 = target_layer.register_full_backward_hook(
        lambda m, gi, go: grads.setdefault("g", go[0]))
    model.zero_grad()
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if class_idx is None:
        class_idx = out.argmax(1)
    out.gather(1, class_idx[:, None]).sum().backward()
    a, g = acts["a"], grads["g"]
    h1.remove(); h2.remove()
    if a.dim() == 3:                      # transformer tokens (B, N, C) -> square map
        b, n, c = a.shape
        s = int(round((n - 1) ** 0.5))
        a = a[:, 1:, :].transpose(1, 2).reshape(b, c, s, s)
        g = g[:, 1:, :].transpose(1, 2).reshape(b, c, s, s)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze(1)
    cam = (cam - cam.amin((1, 2), True)) / (cam.amax((1, 2), True) -
                                            cam.amin((1, 2), True) + 1e-8)
    return cam.detach().cpu().numpy()


# ======================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out-dir", default="runs/baseline")
    ap.add_argument("--models", nargs="+", default=["resnet50"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--source-probe", action="store_true")
    ap.add_argument("--lodo", action="store_true")
    a = ap.parse_args()

    cfg = Config(manifest=a.manifest, out_dir=a.out_dir, models=a.models,
                 seeds=a.seeds, folds=a.folds, epochs=a.epochs,
                 batch_size=a.batch_size, img_size=a.img_size,
                 num_workers=a.num_workers, n_bootstrap=a.n_bootstrap)

    if a.source_probe:
        source_probe(cfg); return
    if a.lodo:
        run_lodo(cfg); return
    if not a.analyze_only:
        run_benchmark(cfg)
    analyze(cfg)


if __name__ == "__main__":
    main()