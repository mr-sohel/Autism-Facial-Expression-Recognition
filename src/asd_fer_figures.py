#!/usr/bin/env python3
"""
asd_fer_figures.py
==================
Every figure the manuscript needs, generated from the saved predictions and training
histories that asd_fer_baseline.py writes. No GPU, no re-training.

    python asd_fer_figures.py --run-dir runs/baseline
    python asd_fer_figures.py --run-dir runs/baseline --compare-dir runs/leaked

DESIGN RULES BAKED IN (they are also journal requirements)
----------------------------------------------------------
* Six emotion classes are NEVER six overlaid hues. Per-class curves are drawn as
  small multiples, one class per panel, single series. Six overlapping colour-coded
  curves are unreadable for a colour-blind reader and illegible in greyscale print.
* Model overlays are capped at three, drawn in a CVD-validated three-hue order with
  distinct dash patterns as secondary encoding, so identity survives greyscale.
* Folds use one hue stepped light-to-dark (ordinal data gets an ordinal ramp).
* Magnitude (confusion matrices, agreement) uses one hue light-to-dark. Signed
  differences use blue-red diverging with a grey zero. Never a rainbow.
* No dual-axis panels anywhere. Loss and F1 have different units, so they get
  separate panels rather than two y-scales on one.
* Every figure is written at 300 dpi with a white ground and vector PDF alongside.

Palettes below were checked with a CVD/contrast validator, not chosen by eye.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CLASSES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]

# --- validated palettes ---------------------------------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # all-pairs CVD-safe; 3 is the cap
DASH = [(None, None), (5, 2), (1.5, 1.6)]        # secondary encoding for greyscale
FOLD_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # ordinal, one hue
SEQ = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUE_CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "divbr", ["#0d366b", "#3987e5", "#cde2fb", "#f0efec", "#f6c3bf", "#e34948", "#8f1f1e"])
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
GRID, RULE = "#e8e8e4", "#d4d4cf"
GOOD, WARN, CRIT = "#1baf7a", "#eda100", "#d9403f"


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": RULE, "axes.linewidth": .8, "axes.labelcolor": INK2,
        "text.color": INK, "xtick.color": INK3, "ytick.color": INK3,
        "grid.color": GRID, "grid.linewidth": .7,
        "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")          # vector, for the camera-ready
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def short(name: str) -> str:
    """timm identifiers are long enough to blow out legends and tick labels.
    'swin_base_patch4_window7_224' -> 'swin_base'. Keep the full name in the CSVs."""
    import re
    s = re.sub(r"_patch\d+.*$", "", str(name))
    s = re.sub(r"_window\d+.*$", "", s)
    s = re.sub(r"_(224|256|384|100)$", "", s)
    return s


# ======================================================================================
# Loading
# ======================================================================================
def load_run(run_dir: Path):
    runs, hist = {}, {}
    for f in sorted((run_dir / "preds").glob("*.npz")):
        model, seed = f.stem.split("__seed")
        runs.setdefault(model, {})[int(seed)] = np.load(f, allow_pickle=True)
    for f in sorted((run_dir / "preds").glob("*_history.json")):
        model, seed = f.stem.replace("_history", "").split("__seed")
        hist.setdefault(model, {})[int(seed)] = json.loads(f.read_text())
    if not runs:
        raise SystemExit(f"no predictions found in {run_dir/'preds'}")
    return runs, hist


def pooled(runs):
    """model -> (y, mean-prob over seeds, groups, [per-seed probs])"""
    out = {}
    for m, byseed in runs.items():
        ss = sorted(byseed)
        ps = [softmax(byseed[s]["oof_logits"]) for s in ss]
        d0 = byseed[ss[0]]
        out[m] = (d0["y_dev"], np.mean(ps, 0), d0["group_dev"], ps)
    return out


def rank_models(pool):
    """Rank on the mean over seeds of per-seed macro-F1 -- the same definition the
    benchmark table and the forest plot use. Ranking on the seed-ENSEMBLE instead would
    quietly give a different 'best model' in different figures of the same paper."""
    from sklearn.metrics import f1_score
    def score(m):
        y, _, _, ps = pool[m]
        return np.mean([f1_score(y, p.argmax(1), average="macro", zero_division=0)
                        for p in ps])
    return sorted(pool, key=lambda m: -score(m))


# ======================================================================================
# F01 / F02  Learning curves  -- "train vs validation"
# ======================================================================================
def fig_learning_curves(hist, model, out: Path):
    """Train vs validation, loss and macro-F1, mean over folds with a min-max band.

    Loss and F1 get separate panels. Putting them on one panel with two y-axes is the
    single most common chart error in ML papers: the crossing point of the two lines is
    an artefact of the arbitrary scaling, and readers reliably over-read it.
    """
    seeds = hist.get(model)
    if not seeds:
        return
    folds = seeds[sorted(seeds)[0]]
    # legend goes where the curves are not: loss descends (upper right free),
    # F1 ascends (lower right free)
    keys = [("train_loss", "val_loss", "Cross-entropy loss", "upper right"),
            ("train_macro_f1", "val_macro_f1", "Macro-F1", "lower right")]

    n_ep = max(len(f["history"]) for f in folds)
    def stack(key):
        M = np.full((len(folds), n_ep), np.nan)
        for i, f in enumerate(folds):
            v = [h.get(key, np.nan) for h in f["history"]]
            M[i, :len(v)] = v
        return M

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, (ktr, kva, ylab, legend_loc) in zip(axes, keys):
        Tr, Va = stack(ktr), stack(kva)
        if np.all(np.isnan(Tr)) and np.all(np.isnan(Va)):
            ax.set_visible(False); continue
        x = np.arange(1, n_ep + 1)
        for M, c, lab, dash in ((Tr, SERIES[0], "Train", DASH[0]),
                                (Va, SERIES[1], "Validation", DASH[1])):
            mu = np.nanmean(M, 0)
            lo, hi = np.nanmin(M, 0), np.nanmax(M, 0)
            ax.fill_between(x, lo, hi, color=c, alpha=.13, linewidth=0)
            ax.plot(x, mu, color=c, label=lab,
                    dashes=dash if dash[0] else (None, None))
        # early-stopping epoch: where validation macro-F1 peaked, on average
        vf = stack("val_macro_f1")
        if not np.all(np.isnan(vf)):
            be = int(np.nanargmax(np.nanmean(vf, 0))) + 1
            ax.axvline(be, color=INK3, lw=.8, ls=":", zorder=1)
            # placed OUTSIDE the axes, above the top spine: the only position that can
            # never collide with a curve or the legend, whatever the data does
            ax.annotate(f"selected epoch {be}", xy=(be, 1.0),
                        xycoords=("data", "axes fraction"), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, color=INK3)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
        ax.legend(loc=legend_loc)
    fig.suptitle(f"{short(model)} — training vs validation, mean over {len(folds)} folds "
                 f"(band = fold min–max)", y=1.10, fontsize=9)
    save(fig, out, "F01_learning_curves_train_vs_val")


def fig_learning_curves_per_fold(hist, model, out: Path):
    """Per-fold validation macro-F1. Folds are ordered, so they get an ordinal ramp
    rather than arbitrary categorical hues."""
    seeds = hist.get(model)
    if not seeds:
        return
    folds = seeds[sorted(seeds)[0]]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, key, ylab in ((axes[0], "train_loss", "Train loss"),
                          (axes[1], "val_macro_f1", "Validation macro-F1")):
        for i, f in enumerate(folds):
            v = [h.get(key, np.nan) for h in f["history"]]
            ax.plot(np.arange(1, len(v) + 1), v,
                    color=FOLD_RAMP[i % len(FOLD_RAMP)], label=f"Fold {i}", lw=1.3)
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylab)
    axes[1].legend(ncol=2, loc="lower right")
    fig.suptitle(f"{short(model)} — per-fold learning curves", y=1.04, fontsize=9)
    save(fig, out, "F02_learning_curves_per_fold")


def fig_seed_stability(pool, out: Path):
    """Seed-to-seed spread per model. If this is comparable to the between-model gaps,
    your ranking is measuring initialisation, not architecture -- and this figure is
    how you show it in one glance."""
    from sklearn.metrics import f1_score
    order = rank_models(pool)
    fig, ax = plt.subplots(figsize=(6.4, max(2.6, .34 * len(order) + 1.0)))
    for i, m in enumerate(reversed(order)):
        y, _, _, ps = pool[m]
        vals = [f1_score(y, p.argmax(1), average="macro", zero_division=0) for p in ps]
        ax.plot(vals, [i] * len(vals), "o", ms=5, color=SERIES[0], alpha=.75,
                markeredgecolor="white", markeredgewidth=.8)
        ax.plot([min(vals), max(vals)], [i, i], color=SERIES[0], lw=1.2, alpha=.35,
                zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short(m) for m in reversed(order)])
    ax.set_xlabel("Macro-F1, one point per seed")
    ax.grid(axis="y", visible=False)
    ax.set_title("Seed-to-seed variation within each backbone", loc="left")
    save(fig, out, "F03_seed_stability")


# ======================================================================================
# F04-F05  Confusion matrices
# ======================================================================================
def _cm_panel(ax, cm, norm_rows=True, title="", show_counts=True, ylab=True):
    M = cm / np.maximum(cm.sum(1, keepdims=True), 1) if norm_rows else cm
    vmax = 1.0 if norm_rows else cm.max()
    im = ax.imshow(M, cmap=BLUE_CMAP, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(CLASSES))); ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=42, ha="right")
    ax.set_yticklabels(CLASSES if ylab else [])
    ax.grid(False)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            v = M[i, j]
            txt = f"{v:.2f}" if norm_rows else f"{int(v)}"
            if show_counts and norm_rows:
                txt += f"\n{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.4,
                    color="white" if v > vmax * .55 else INK)
    ax.set_xlabel("Predicted")
    if ylab:
        ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=8.5)
    return im


def fig_confusion(pool, model, out: Path):
    from sklearn.metrics import confusion_matrix
    y, p, _, _ = pool[model]
    cm = confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES)))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    _cm_panel(axes[0], cm, True, "Row-normalised (recall), counts beneath")
    im = _cm_panel(axes[1], cm, False, "Raw counts", show_counts=False, ylab=False)
    fig.colorbar(im, ax=axes[1], fraction=.046, pad=.03)
    fig.subplots_adjust(wspace=.12)
    fig.suptitle(f"Confusion matrix — {short(model)}", y=1.01, fontsize=9.5)
    save(fig, out, "F04_confusion_matrix")


def fig_confusion_all(pool, out: Path, k=6):
    from sklearn.metrics import confusion_matrix
    order = rank_models(pool)[:k]
    ncol = 3
    nrow = int(np.ceil(len(order) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow))
    for ax, m in zip(np.ravel(axes), order):
        y, p, _, _ = pool[m]
        cm = confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES)))
        M = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(M, cmap=BLUE_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.grid(False)
        ax.set_xticklabels([c[:3] for c in CLASSES], fontsize=6)
        ax.set_yticklabels([c[:3] for c in CLASSES], fontsize=6)
        ax.set_title(short(m), loc="left", fontsize=7.5)
    for ax in np.ravel(axes)[len(order):]:
        ax.set_visible(False)
    fig.suptitle("Row-normalised confusion matrices across backbones", y=1.0, fontsize=9.5)
    save(fig, out, "F05_confusion_grid")


def fig_confusion_delta(pool_a, pool_b, model, out: Path,
                        label_a="subject-independent", label_b="image-level"):
    """The leakage figure. Difference between two protocols on the same model.

    Signed data, so a diverging map with a neutral zero -- never a sequential ramp,
    which would hide the sign.
    """
    from sklearn.metrics import confusion_matrix
    if model not in pool_a or model not in pool_b:
        return
    def rn(pool):
        y, p, _, _ = pool[model]
        cm = confusion_matrix(y, p.argmax(1), labels=range(len(CLASSES)))
        return cm / np.maximum(cm.sum(1, keepdims=True), 1)
    D = rn(pool_a) - rn(pool_b)
    lim = max(.05, np.abs(D).max())
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(D, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -lim, lim))
    ax.set_xticks(range(6), CLASSES, rotation=42, ha="right")
    ax.set_yticks(range(6), CLASSES); ax.grid(False)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{D[i,j]:+.2f}", ha="center", va="center", fontsize=6.4,
                    color="white" if abs(D[i, j]) > lim * .6 else INK)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{short(model)}: {label_a} minus {label_b}", loc="left", fontsize=8.5)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label="Δ recall")
    save(fig, out, "F06_confusion_delta_protocol")


# ======================================================================================
# F07-F09  ROC and precision-recall
# ======================================================================================
def _roc_band(y_bin, score, groups, n_boot=400, seed=0, n_grid=200):
    """Group-bootstrap band for the ROC curve AND an interval for the AUC.

    The band is a real vertical-averaging bootstrap: at each FPR on a common grid, the
    2.5th and 97.5th percentiles of the resampled TPR. A shaded area under a single ROC
    curve is NOT a confidence band, and captioning it as one is a reporting error --
    which is exactly the kind of thing a statistical reviewer writes up.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    grid = np.linspace(0, 1, n_grid)
    tprs, aucs = [], []
    for _ in range(n_boot):
        gs = rng.choice(uniq, len(uniq), True)
        idx = np.concatenate([idx_by_g[g] for g in gs])
        if len(np.unique(y_bin[idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[idx], score[idx])
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(roc_auc_score(y_bin[idx], score[idx]))
    if not tprs:
        return grid, None, None, np.nan, np.nan
    T = np.vstack(tprs)
    return (grid, np.percentile(T, 2.5, 0), np.percentile(T, 97.5, 0),
            float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def fig_roc_per_class(pool, model, out: Path, n_boot=400):
    """Six classes as small multiples, not six overlaid hues.

    Each panel is one class, one series, with its AUC and a group-bootstrap CI. The
    reader can actually read the rare classes, and it prints in greyscale.
    """
    from sklearn.metrics import roc_curve, roc_auc_score
    y, p, g, _ = pool[model]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.2), sharex=True, sharey=True)
    for k, (ax, cls) in enumerate(zip(np.ravel(axes), CLASSES)):
        yb = (y == k).astype(int)
        n_pos = int(yb.sum())
        if n_pos < 2:
            ax.set_visible(False); continue
        fpr, tpr, _ = roc_curve(yb, p[:, k])
        auc = roc_auc_score(yb, p[:, k])
        grid, blo, bhi, lo, hi = _roc_band(yb, p[:, k], g, n_boot)
        ax.plot([0, 1], [0, 1], color=INK3, lw=.9, ls=(0, (2, 2)), zorder=1)
        if blo is not None:
            ax.fill_between(grid, blo, bhi, color=SERIES[0], alpha=.18, linewidth=0,
                            zorder=2)
        ax.plot(fpr, tpr, color=SERIES[0], lw=1.7, zorder=3)
        ax.set_title(f"{cls}  (n={n_pos})", loc="left", fontsize=8.5)
        ax.text(.97, .06, f"AUC {auc:.3f}\n[{lo:.3f}, {hi:.3f}]", ha="right",
                va="bottom", transform=ax.transAxes, fontsize=7, color=INK2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.003)
        if k >= 3: ax.set_xlabel("False positive rate")
        if k % 3 == 0: ax.set_ylabel("True positive rate")
    fig.suptitle(f"One-vs-rest ROC by emotion — {short(model)}   "
                 f"(band = 95% group-bootstrap interval on TPR)", y=1.0, fontsize=9.5)
    save(fig, out, "F07_roc_per_class")


def fig_roc_models(pool, out: Path, k=3):
    """Macro-average ROC for the top three models only. Three is the cap at which a
    categorical palette still separates for every colour-vision type on overlapping
    curves; dash patterns carry identity in greyscale."""
    from sklearn.metrics import roc_curve, auc as _auc
    order = rank_models(pool)[:k]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    for i, m in enumerate(order):
        y, p, _, _ = pool[m]
        Y = np.eye(len(CLASSES))[y]
        # macro: interpolate each class onto a common grid, then average
        grid = np.linspace(0, 1, 400)
        tprs = []
        for c in range(len(CLASSES)):
            if Y[:, c].sum() < 2: continue
            fpr, tpr, _ = roc_curve(Y[:, c], p[:, c])
            tprs.append(np.interp(grid, fpr, tpr))
        mac = np.mean(tprs, 0)
        axes[0].plot(grid, mac, color=SERIES[i], lw=1.7,
                     dashes=DASH[i] if DASH[i][0] else (None, None),
                     label=f"{short(m)}  (AUC {_auc(grid, mac):.3f})")
        fpr, tpr, _ = roc_curve(Y.ravel(), p.ravel())      # micro
        axes[1].plot(fpr, tpr, color=SERIES[i], lw=1.7,
                     dashes=DASH[i] if DASH[i][0] else (None, None),
                     label=f"{short(m)}  (AUC {_auc(fpr, tpr):.3f})")
    for ax, t in zip(axes, ("Macro-average (each class weighted equally)",
                            "Micro-average (each image weighted equally)")):
        ax.plot([0, 1], [0, 1], color=INK3, lw=.9, ls=(0, (2, 2)), zorder=1)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_title(t, loc="left", fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.003)
        ax.legend(loc="lower right")
    save(fig, out, "F08_roc_models")


def fig_pr_per_class(pool, model, out: Path):
    """Precision-recall by class, with each class's prevalence drawn as the no-skill
    baseline. On a 10:1 imbalanced problem PR is the more honest curve, and without the
    prevalence line a reader cannot tell a good AP from a trivial one."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    y, p, _, _ = pool[model]
    fig, axes = plt.subplots(2, 3, figsize=(7.8, 5.2), sharex=True, sharey=True)
    for k, (ax, cls) in enumerate(zip(np.ravel(axes), CLASSES)):
        yb = (y == k).astype(int)
        if yb.sum() < 2:
            ax.set_visible(False); continue
        pr, rc, _ = precision_recall_curve(yb, p[:, k])
        ap = average_precision_score(yb, p[:, k])
        base = yb.mean()
        ax.axhline(base, color=INK3, lw=.9, ls=(0, (2, 2)), zorder=1)
        ax.fill_between(rc, pr, color=SERIES[0], alpha=.10, linewidth=0, step="post")
        ax.step(rc, pr, where="post", color=SERIES[0], lw=1.7)
        ax.set_title(f"{cls}  (n={int(yb.sum())})   AP {ap:.3f}", loc="left",
                     fontsize=8.5)
        # the prevalence label goes at recall~0, where the curve is pinned near 1.0 and
        # can never collide with it, whatever the class balance
        ax.text(.012, base + .022, f"chance {base:.3f}", transform=ax.get_yaxis_transform(),
                fontsize=6.6, color=INK3, va="bottom", ha="left")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.003)
        if k >= 3: ax.set_xlabel("Recall")
        if k % 3 == 0: ax.set_ylabel("Precision")
    fig.suptitle(f"Precision–recall by emotion — {short(model)} "
                 f"(dashed line = class prevalence)", y=1.0, fontsize=9.5)
    save(fig, out, "F09_pr_per_class")


# ======================================================================================
# F10-F12  Uncertainty behaviour
# ======================================================================================
def fig_calibration(pool, out: Path, k=3, n_bins=12):
    from asd_fer_baseline import expected_calibration_error
    order = rank_models(pool)[:k]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=INK3, lw=.9, ls=(0, (2, 2)), zorder=1,
            label="perfect calibration")
    for i, m in enumerate(order):
        y, p, _, _ = pool[m]
        cal, rel = expected_calibration_error(y, p, n_bins)
        # A bin holding a handful of images produces a wild accuracy estimate that reads
        # as a calibration failure. Plot only bins with >=10 images; ECE still uses all.
        r = rel.dropna()
        r = r[r["n"] >= 10]
        ax.plot(r["confidence"], r["accuracy"], "o-", ms=4.5, color=SERIES[i], lw=1.5,
                dashes=DASH[i] if DASH[i][0] else (None, None),
                markeredgecolor="white", markeredgewidth=.7,
                label=f"{short(m)}  (ECE {cal['ece']:.3f})")
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Reliability  (bins with n≥10)", loc="left", fontsize=8.5)
    ax.legend(loc="lower right")

    ax = axes[1]
    y, p, _, _ = pool[order[0]]
    conf = p.max(1); ok = p.argmax(1) == y
    bins = np.linspace(0, 1, 26)
    ax.hist([conf[ok], conf[~ok]], bins=bins, stacked=True,
            color=[SERIES[2], SERIES[1]], label=["correct", "incorrect"],
            edgecolor="white", linewidth=.4)
    ax.set_xlabel("Predicted confidence"); ax.set_ylabel("Images")
    ax.set_title(f"Confidence distribution — {short(order[0])}", loc="left", fontsize=8.5)
    ax.legend(loc="upper left")
    save(fig, out, "F10_calibration")


def fig_risk_coverage(pool, out: Path, k=3):
    from asd_fer_baseline import risk_coverage
    order = rank_models(pool)[:k]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for i, m in enumerate(order):
        y, p, _, _ = pool[m]
        rc, aurc = risk_coverage(y, p)
        ax.plot(rc["coverage"], rc["risk"], color=SERIES[i], lw=1.7,
                dashes=DASH[i] if DASH[i][0] else (None, None),
                label=f"{short(m)}  (AURC {aurc:.3f})")
    ax.axvline(.8, color=INK3, lw=.8, ls=":")
    ax.annotate("80% coverage", (.8, ax.get_ylim()[1]), xytext=(-4, -3),
                textcoords="offset points", ha="right", va="top",
                fontsize=6.8, color=INK3)
    ax.set_xlabel("Coverage (fraction of images the model answers on)")
    ax.set_ylabel("Selective risk (error rate on those)")
    ax.set_title("Risk–coverage: what abstention buys you", loc="left", fontsize=8.5)
    ax.legend(loc="lower right")
    save(fig, out, "F11_risk_coverage")


def fig_per_class_f1(pool, model, out: Path, n_boot=600):
    """Per-class F1 with a group-bootstrap CI and the support printed on each bar.

    A per-class F1 bar chart without support and without an interval is the figure that
    lets a 14-sample class look like a finding."""
    from sklearn.metrics import f1_score
    y, p, g, _ = pool[model]
    rng = np.random.RandomState(0)
    uniq = np.unique(g); idx_by_g = {q: np.where(g == q)[0] for q in uniq}
    boots = np.full((n_boot, len(CLASSES)), np.nan)
    for b in range(n_boot):
        gs = rng.choice(uniq, len(uniq), True)
        idx = np.concatenate([idx_by_g[q] for q in gs])
        boots[b] = f1_score(y[idx], p[idx].argmax(1), labels=range(len(CLASSES)),
                            average=None, zero_division=0)
    est = f1_score(y, p.argmax(1), labels=range(len(CLASSES)), average=None,
                   zero_division=0)
    lo = np.nanpercentile(boots, 2.5, 0); hi = np.nanpercentile(boots, 97.5, 0)
    sup = np.bincount(y, minlength=len(CLASSES))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(CLASSES))
    # thin bars, 2px surface gap between neighbours, rounded data end
    ax.bar(xs, est, width=.56, color=SERIES[0], edgecolor="white", linewidth=1.6,
           zorder=2)
    ax.errorbar(xs, est, yerr=[np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)],
                fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3, zorder=3)
    for x, v, n in zip(xs, est, sup):
        ax.text(x, .015, f"n={n}", ha="center", va="bottom", fontsize=6.6,
                color="white" if v > .18 else INK3, zorder=4)
        ax.text(x, hi[x] + .02, f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                color=INK2)
    thin = sup < 30
    if thin.any():
        for x in xs[thin]:
            ax.text(x, -.085, "unstable", ha="center", fontsize=6.2, color=WARN,
                    transform=ax.get_xaxis_transform())
    ax.set_xticks(xs, CLASSES, rotation=18, ha="right")
    ax.set_ylabel("F1 (95% CI, group bootstrap)")
    ax.set_ylim(0, min(1.06, max(hi) + .12))
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Per-class F1 with support — {short(model)}", loc="left", fontsize=8.5)
    save(fig, out, "F12_per_class_f1")


# ======================================================================================
# F13-F16  Comparison, agreement, errors
# ======================================================================================
def fig_forest(pool, out: Path, n_boot=800):
    from sklearn.metrics import f1_score, accuracy_score
    from asd_fer_baseline import cluster_bootstrap_multi
    order = rank_models(pool)
    rows = []
    for m in order:
        y, _, g, ps = pool[m]
        ci = cluster_bootstrap_multi(
            y, ps, g,
            {"f1": lambda a, b: f1_score(a, b.argmax(1), average="macro", zero_division=0),
             "acc": lambda a, b: accuracy_score(a, b.argmax(1))}, n_boot=n_boot)
        rows.append({"model": m, **{f"{k}_{s}": v for k, t in ci.items()
                                    for s, v in zip(("est", "lo", "hi"), t)}})
    df = pd.DataFrame(rows).sort_values("f1_est")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, max(2.8, .34 * len(df) + 1.1)),
                             sharey=True)
    for ax, k, lab in ((axes[0], "f1", "Macro-F1"), (axes[1], "acc", "Accuracy")):
        ax.errorbar(df[f"{k}_est"], range(len(df)),
                    xerr=[np.clip(df[f"{k}_est"] - df[f"{k}_lo"], 0, None),
                          np.clip(df[f"{k}_hi"] - df[f"{k}_est"], 0, None)],
                    fmt="o", ms=6, color=SERIES[0], ecolor=INK2, elinewidth=1.1,
                    capsize=3, markeredgecolor="white", markeredgewidth=.9)
        ax.set_xlabel(f"{lab} (95% CI)")
        ax.grid(axis="y", visible=False)
    # the top model's interval, extended, shows which rows it overlaps
    top = df.iloc[-1]
    axes[0].axvspan(top["f1_lo"], top["f1_hi"], color=SERIES[0], alpha=.07, zorder=0)
    axes[0].set_yticks(range(len(df)), [short(m) for m in df["model"]])
    fig.suptitle("Backbone comparison — shaded band is the best model's interval; "
                 "any model overlapping it is not distinguishable from it",
                 y=1.02, fontsize=8.8)
    save(fig, out, "F13_forest_comparison")


def fig_significance_matrix(pool, out: Path):
    """Pairwise McNemar, Holm-corrected, as a matrix. Status colours, not a ramp:
    'significant' is a state, not a magnitude."""
    from asd_fer_baseline import mcnemar_test, holm
    order = rank_models(pool)
    n = len(order)
    raw = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = order[i], order[j]
            raw[(a, b)] = mcnemar_test(pool[a][0], pool[a][1], pool[b][1])["p"]
    adj = holm(raw)
    M = np.full((n, n), np.nan)
    for (a, b), q in adj.items():
        M[order.index(a), order.index(b)] = q
        M[order.index(b), order.index(a)] = q
    fig, ax = plt.subplots(figsize=(max(4.8, .52 * n + 2.4), max(4.2, .52 * n + 1.9)))
    disp = np.where(np.isnan(M), np.nan, np.clip(M, 1e-30, 1))
    L = -np.log10(disp)
    # Cap the colour scale at p=1e-4. A single p=1e-18 pair would otherwise compress
    # every meaningful gradation near 0.05 into one indistinguishable shade.
    VMAX = 4.0
    im = ax.imshow(L, cmap=BLUE_CMAP, vmin=0, vmax=VMAX)
    ax.set_xticks(range(n), [short(m) for m in order], rotation=42, ha="right", fontsize=7)
    ax.set_yticks(range(n), [short(m) for m in order], fontsize=7); ax.grid(False)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=INK3, fontsize=7)
                continue
            q = M[i, j]
            sig = q < .05
            ax.text(j, i, f"{q:.3f}" if q >= .001 else "<.001", ha="center",
                    va="center", fontsize=6.2,
                    color="white" if L[i, j] / VMAX > .55 else INK,
                    fontweight="bold" if sig else "normal")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label="−log₁₀ adjusted p  (capped at 4)")
    ax.set_title("Pairwise McNemar, Holm-corrected (bold = significant at 0.05)",
                 loc="left", fontsize=8.5)
    save(fig, out, "F14_significance_matrix")


def fig_agreement(pool, out: Path):
    """Error-pattern agreement between backbones.

    This replaces a raw 'model correlation heatmap'. Two models agreeing on which
    images they get WRONG is what tells you an ensemble or hybrid has nothing to gain;
    correlation of scores does not. Cohen's kappa on the correct/incorrect indicator.
    """
    from sklearn.metrics import cohen_kappa_score
    order = rank_models(pool)
    n = len(order)
    K = np.eye(n)
    corr = {m: (pool[m][1].argmax(1) == pool[m][0]).astype(int) for m in order}
    for i in range(n):
        for j in range(i + 1, n):
            k = cohen_kappa_score(corr[order[i]], corr[order[j]])
            K[i, j] = K[j, i] = k
    fig, ax = plt.subplots(figsize=(max(4.8, .52 * n + 2.4), max(4.2, .52 * n + 1.9)))
    im = ax.imshow(K, cmap=BLUE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(n), [short(m) for m in order], rotation=42, ha="right", fontsize=7)
    ax.set_yticks(range(n), [short(m) for m in order], fontsize=7); ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{K[i,j]:.2f}", ha="center", va="center", fontsize=6.2,
                    color="white" if K[i, j] > .55 else INK)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label="Cohen's κ on correctness")
    ax.set_title("Do backbones fail on the same images?  (low κ ⇒ complementary)",
                 loc="left", fontsize=8.5)
    save(fig, out, "F15_error_agreement")


def fig_subgroup(pool, runs, model, out: Path):
    """Performance by source dataset. A wide spread here is the visual form of the
    merged-corpus confound."""
    from sklearn.metrics import f1_score, accuracy_score
    y, p, _, _ = pool[model]
    d0 = runs[model][min(runs[model])]
    src = d0["source_dev"]
    names, f1s, accs, ns = [], [], [], []
    for s in np.unique(src):
        m = src == s
        if m.sum() < 20: continue
        names.append(str(s)); ns.append(int(m.sum()))
        f1s.append(f1_score(y[m], p[m].argmax(1), average="macro", zero_division=0))
        accs.append(accuracy_score(y[m], p[m].argmax(1)))
    if not names:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    xs = np.arange(len(names))
    w = .34
    ax.bar(xs - w / 2 - .01, f1s, w, color=SERIES[0], edgecolor="white", linewidth=1.6,
           label="Macro-F1")
    ax.bar(xs + w / 2 + .01, accs, w, color=SERIES[1], edgecolor="white", linewidth=1.6,
           label="Accuracy")
    overall = f1_score(y, p.argmax(1), average="macro", zero_division=0)
    ax.axhline(overall, color=INK3, lw=.9, ls=(0, (2, 2)))
    ax.annotate(f"pooled macro-F1 {overall:.3f}", (len(names) - .5, overall),
                xytext=(0, 3), textcoords="offset points", ha="right",
                fontsize=6.8, color=INK3)
    ax.set_xticks(xs, [f"{a}\nn={b}" for a, b in zip(names, ns)])
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False); ax.legend(ncol=2, loc="upper right")
    ax.set_title(f"Performance by source dataset — {short(model)}", loc="left", fontsize=8.5)
    save(fig, out, "F16_subgroup_by_source")


def fig_lodo(csv_path: Path, out: Path):
    """Leave-one-dataset-out, as a model x held-out-source matrix."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    piv = df.pivot_table(index="model", columns="held_out_source", values="f1_macro")
    fig, ax = plt.subplots(figsize=(max(4.4, .9 * piv.shape[1] + 2.6),
                                    max(2.6, .5 * piv.shape[0] + 1.7)))
    im = ax.imshow(piv.values, cmap=BLUE_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=25, ha="right")
    ax.set_yticks(range(piv.shape[0]), [short(m) for m in piv.index], fontsize=7.5); ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if v > .55 else INK)
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03, label="Macro-F1 on held-out source")
    ax.set_title("Leave-one-dataset-out generalisation", loc="left", fontsize=8.5)
    save(fig, out, "F17_lodo_matrix")


def fig_dataset(manifest_csv: Path, out: Path):
    """Class x source composition. Figure 1 of the paper."""
    if not manifest_csv.exists():
        return
    df = pd.read_csv(manifest_csv)
    piv = (df.pivot_table(index="label", columns="source", values="path",
                          aggfunc="count").reindex(CLASSES).fillna(0))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    bottom = np.zeros(len(piv))
    ramp = [SEQ[2], SEQ[3], SEQ[4], SEQ[6]]
    for i, s in enumerate(piv.columns):
        ax.bar(range(len(piv)), piv[s].values, .58, bottom=bottom,
               color=ramp[i % len(ramp)], edgecolor="white", linewidth=1.6, label=s)
        bottom += piv[s].values
    for i, t in enumerate(bottom):
        ax.text(i, t + max(bottom) * .015, f"{int(t)}", ha="center", fontsize=7,
                color=INK2)
    ax.set_xticks(range(len(piv)), piv.index, rotation=18, ha="right")
    ax.set_ylabel("Images"); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(bottom) * 1.32)          # headroom so the legend clears the bars
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(.5, 1.0))
    ax.set_title("Class composition by source", loc="left", fontsize=8.5)

    ax = axes[1]
    if "group" in df.columns:
        per = df.groupby("group").size().value_counts().sort_index()
        ax.bar(per.index, per.values, .7, color=SERIES[0], edgecolor="white",
               linewidth=1.4)
        ax.set_xlabel("Images contributed by one subject group")
        ax.set_ylabel("Number of groups"); ax.grid(axis="x", visible=False)
        ax.set_title(f"Subject groups (n={df['group'].nunique()}) — "
                     f"why image-level splits leak", loc="left", fontsize=8.5)
    else:
        ax.set_visible(False)
    save(fig, out, "F18_dataset_composition")


# ======================================================================================
def build_all(run_dir: Path, compare_dir: Path | None = None,
              manifest: Path | None = None):
    style()
    out = run_dir / "figures"
    runs, hist = load_run(run_dir)
    pool = pooled(runs)
    best = rank_models(pool)[0]
    print(f"[figures] {len(pool)} models, best = {best}")

    fig_learning_curves(hist, best, out)
    fig_learning_curves_per_fold(hist, best, out)
    fig_seed_stability(pool, out)
    fig_confusion(pool, best, out)
    fig_confusion_all(pool, out)
    if compare_dir:
        cruns, _ = load_run(Path(compare_dir))
        fig_confusion_delta(pool, pooled(cruns), best, out)
    fig_roc_per_class(pool, best, out)
    fig_roc_models(pool, out)
    fig_pr_per_class(pool, best, out)
    fig_calibration(pool, out)
    fig_risk_coverage(pool, out)
    fig_per_class_f1(pool, best, out)
    fig_forest(pool, out)
    fig_significance_matrix(pool, out)
    fig_agreement(pool, out)
    fig_subgroup(pool, runs, best, out)
    fig_lodo(run_dir / "T8_lodo.csv", out)
    if manifest:
        fig_dataset(Path(manifest), out)
    print(f"[figures] all written to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--compare-dir", default=None,
                    help="a second run (e.g. image-level splits) for the delta figure")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args()
    build_all(Path(a.run_dir), a.compare_dir, a.manifest)