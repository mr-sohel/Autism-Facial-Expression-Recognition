#!/usr/bin/env python3
"""
build_manifest.py
=================
Step 0 of the publication-grade pipeline: turn a pile of merged ASD facial-emotion
images into an auditable manifest with (a) exact + near-duplicate flags, (b) inferred
subject/identity groups, and (c) source-dataset provenance.

WHY THIS FILE EXISTS
--------------------
Your current protocol splits/folds at the IMAGE level. When four face datasets are
merged, the same child almost certainly appears in many frames. Image-level splitting
puts frames of the same child in both train and test, which inflates every number you
reported. A Q1 biomedical reviewer will ask for subject-independent evaluation, and if
you cannot produce it the paper is rejected. This script produces the `group` column
that makes subject-independent evaluation possible.

It also produces the `source` column so you can run the two experiments reviewers
always ask for on merged corpora:
  1. Source-probe: can a classifier predict which dataset an image came from?
     (High AUC => your emotion model may be exploiting dataset artifacts.)
  2. Leave-one-dataset-out (LODO) generalization.

INPUT LAYOUT (flexible)
-----------------------
Pass one or more roots as `name=path`, each containing class subfolders:

    dataset_nora/anger/*.jpg
    dataset_nora/fear/*.jpg
    ...

USAGE
-----
    python build_manifest.py \
        --roots nora=/kaggle/input/nora_mendeley \
                ferac=/kaggle/input/ferac \
                talaat=/kaggle/input/talaat \
                hasibur=/kaggle/input/hasibur \
        --out manifest.csv \
        --phash-threshold 6 \
        --identity-threshold 0.55

DEPENDENCIES
------------
    pip install pillow imagehash numpy pandas scikit-learn tqdm
    pip install facenet-pytorch          # optional but STRONGLY recommended
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CANONICAL_CLASSES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]

# Map the label spellings that appear across the four source datasets onto one scheme.
# EXTEND THIS as you inspect your folders -- an unmapped folder name raises an error
# rather than silently creating a 7th class.
LABEL_ALIASES = {
    "anger": "anger", "angry": "anger", "ang": "anger",
    "fear": "fear", "fearful": "fear", "afraid": "fear",
    "joy": "joy", "happy": "joy", "happiness": "joy", "smile": "joy",
    "natural": "natural", "neutral": "natural", "normal": "natural", "calm": "natural",
    "sadness": "sadness", "sad": "sadness",
    "surprise": "surprise", "surprised": "surprise", "surprize": "surprise",
}


# --------------------------------------------------------------------------------------
# 1. Scan
# --------------------------------------------------------------------------------------
def scan_roots(roots: dict[str, str]) -> pd.DataFrame:
    rows = []
    unmapped = set()
    for source, root in roots.items():
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"root '{source}' -> {root} does not exist")
        for p in sorted(root.rglob("*")):
            if p.suffix.lower() not in IMG_EXT or not p.is_file():
                continue
            raw_label = p.parent.name.strip().lower().replace(" ", "_")
            label = LABEL_ALIASES.get(raw_label)
            if label is None:
                unmapped.add(f"{source}:{raw_label}")
                continue
            rows.append({"path": str(p), "source": source,
                         "raw_label": raw_label, "label": label})
    if unmapped:
        raise ValueError(
            "Unmapped folder names found. Add them to LABEL_ALIASES (or exclude the "
            f"folders) before continuing:\n  " + "\n  ".join(sorted(unmapped))
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images found. Check --roots.")
    return df


# --------------------------------------------------------------------------------------
# 2. Duplicate detection (exact + perceptual)
# --------------------------------------------------------------------------------------
def md5_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()


def compute_hashes(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import imagehash
    except ImportError:
        sys.exit("pip install imagehash")

    md5s, phashes, sizes, bad = [], [], [], []
    for path in tqdm(df["path"], desc="hashing"):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                sizes.append(f"{im.width}x{im.height}")
                phashes.append(str(imagehash.phash(im, hash_size=8)))
            md5s.append(md5_of(path))
        except Exception as e:  # corrupt file
            bad.append((path, repr(e)))
            sizes.append(""); phashes.append(""); md5s.append("")
    df = df.copy()
    df["md5"] = md5s
    df["phash"] = phashes
    df["resolution"] = sizes
    if bad:
        print(f"[warn] {len(bad)} unreadable images dropped", file=sys.stderr)
        df = df[df["md5"] != ""].reset_index(drop=True)
    return df


def _hex_to_bits(h: str) -> np.ndarray:
    return np.array([int(b) for b in bin(int(h, 16))[2:].zfill(len(h) * 4)], dtype=np.uint8)


def flag_duplicates(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    """Exact (md5) and near (pHash Hamming <= threshold) duplicate flagging.

    Near-duplicates get a shared `dup_cluster`. Keep ONE representative per cluster
    for training, but never let two members of a cluster land in different CV folds.
    """
    df = df.copy()

    # exact
    df["is_exact_dup"] = df.duplicated(subset=["md5"], keep="first")

    # near: bucket by the first 16 bits to keep pairwise cost sane, then exact Hamming.
    bits = np.stack([_hex_to_bits(h) for h in df["phash"]])
    n = len(df)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    buckets = defaultdict(list)
    for i in range(n):
        # multi-probe: 4 rotated 16-bit prefixes so a few flipped bits do not hide a pair
        for shift in (0, 16, 32, 48):
            key = (shift, bits[i, shift:shift + 16].tobytes())
            buckets[key].append(i)

    for idxs in tqdm(buckets.values(), desc="near-dup"):
        if len(idxs) < 2 or len(idxs) > 4000:
            continue
        sub = bits[idxs]
        d = (sub[:, None, :] != sub[None, :, :]).sum(-1)
        ii, jj = np.where(np.triu(d <= threshold, k=1))
        for a, b in zip(ii, jj):
            union(idxs[a], idxs[b])

    df["dup_cluster"] = [f"D{find(i):06d}" for i in range(n)]
    sizes = df["dup_cluster"].value_counts()
    df["dup_cluster_size"] = df["dup_cluster"].map(sizes)
    return df


# --------------------------------------------------------------------------------------
# 3. Identity (subject) clustering  -- the column that makes your splits defensible
# --------------------------------------------------------------------------------------
def infer_identities(df: pd.DataFrame, threshold: float = 0.55,
                     batch_size: int = 64, device: str = "cuda") -> pd.DataFrame:
    """Cluster faces by identity using FaceNet embeddings + agglomerative clustering.

    `threshold` is a cosine distance cut-off. 0.5-0.6 is the usual operating range for
    VGGFace2-trained InceptionResnetV1. TUNE IT: over-merging (too high) throws away
    data by making groups huge; under-merging (too low) leaves subject leakage.

    Validate the choice by eyeballing ~30 random clusters and reporting, in the paper,
    the purity you observed. Reviewers accept inferred identities if you show the
    validation; they do not accept unvalidated ones.
    """
    try:
        import torch
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print("[warn] facenet-pytorch not installed -- falling back to dup_cluster as "
              "the grouping key. This is WEAKER: it removes duplicate leakage but NOT "
              "subject leakage. Install facenet-pytorch before submitting.",
              file=sys.stderr)
        df = df.copy()
        df["group"] = df["dup_cluster"]
        df["group_source"] = "dup_cluster_fallback"
        return df

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    device = device if torch.cuda.is_available() else "cpu"
    net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    embs, ok_idx = [], []
    buf, buf_idx = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.no_grad():
            e = net(x).cpu().numpy()
        embs.append(e)
        ok_idx.extend(buf_idx)
        buf.clear(); buf_idx.clear()

    for i, path in enumerate(tqdm(df["path"], desc="embedding")):
        try:
            with Image.open(path) as im:
                im = im.convert("RGB").resize((160, 160))
            t = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float()
            t = (t - 127.5) / 128.0
            buf.append(t); buf_idx.append(i)
        except Exception:
            continue
        if len(buf) == batch_size:
            flush()
    flush()

    E = normalize(np.concatenate(embs, 0))
    clust = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="cosine", linkage="average",
    ).fit(E)

    df = df.copy()
    df["group"] = [f"UNK{i}" for i in range(len(df))]
    df.loc[df.index[ok_idx], "group"] = [f"ID{c:05d}" for c in clust.labels_]
    df["group_source"] = "facenet_agglomerative"

    # A duplicate cluster must never straddle two identity groups.
    for dc, sub in df.groupby("dup_cluster"):
        if sub["group"].nunique() > 1:
            df.loc[sub.index, "group"] = sub["group"].iloc[0]
    return df


# --------------------------------------------------------------------------------------
# 4. Audit report
# --------------------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    g = df.groupby("group")
    rep = {
        "n_images": int(len(df)),
        "n_exact_duplicates": int(df["is_exact_dup"].sum()),
        "n_near_dup_clusters_gt1": int((df["dup_cluster_size"] > 1).sum()),
        "n_groups": int(df["group"].nunique()),
        "images_per_group_mean": float(g.size().mean()),
        "images_per_group_max": int(g.size().max()),
        "class_counts": df["label"].value_counts().to_dict(),
        "source_counts": df["source"].value_counts().to_dict(),
        "class_x_source": df.pivot_table(index="label", columns="source",
                                         values="path", aggfunc="count").fillna(0)
                            .astype(int).to_dict(),
        # groups that span >1 source are a red flag: same child in two datasets
        "groups_spanning_sources": int(
            (g["source"].nunique() > 1).sum()),
        # a group that spans >1 label is either a clustering error or a labelling
        # inconsistency between source datasets -- both must be discussed in the paper
        "groups_spanning_labels": int((g["label"].nunique() > 1).sum()),
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="source_name=path pairs")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--audit-out", default="manifest_audit.json")
    ap.add_argument("--phash-threshold", type=int, default=6)
    ap.add_argument("--identity-threshold", type=float, default=0.55)
    ap.add_argument("--skip-identity", action="store_true")
    args = ap.parse_args()

    roots = dict(r.split("=", 1) for r in args.roots)

    df = scan_roots(roots)
    print(f"[1/4] scanned {len(df)} images from {len(roots)} sources")

    df = compute_hashes(df)
    print("[2/4] hashes computed")

    df = flag_duplicates(df, args.phash_threshold)
    print(f"[3/4] {df['is_exact_dup'].sum()} exact dups, "
          f"{(df['dup_cluster_size'] > 1).sum()} images in near-dup clusters")

    if args.skip_identity:
        df["group"] = df["dup_cluster"]
        df["group_source"] = "dup_cluster_only"
    else:
        df = infer_identities(df, args.identity_threshold)
    print(f"[4/4] {df['group'].nunique()} identity groups")

    df.to_csv(args.out, index=False)
    rep = audit(df)
    Path(args.audit_out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out} and {args.audit_out}")


if __name__ == "__main__":
    main()