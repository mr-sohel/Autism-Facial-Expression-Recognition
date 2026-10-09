#!/usr/bin/env python3
"""
build_manifest.py
=================
Step 0 of the publication-grade pipeline: turn a pile of merged ASD facial-emotion
images into an auditable manifest with (a) exact + near-duplicate flags, (b) a
`photo_id` shared by every copy of one photograph, (c) inferred subject/identity groups,
and (d) source-dataset provenance.

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
  2. Leave-one-dataset-out (LODO) generalization -- ONLY if the audit shows the
     sources do not share photographs or children. On the current four sources they
     do (see `groups_spanning_sources`), so LODO is invalid there.

PHOTOGRAPHS, NOT FILES
----------------------
pHash (`dup_cluster`) survives resizing and recompression but not the mirror / crop /
rotation / noise augmentations these sources were built with: it reports 5,139 "unique"
images where there are about 960 photographs. `photo_id` comes from the geometric
matcher in audit_unique_images.py and is what everything downstream must use:
train on `is_photo_rep` rows only, and identities are clustered per photograph so
copies of one photo can never land in two groups.

INPUT LAYOUT (flexible)
-----------------------
Pass one or more roots as `name=path`, each containing class subfolders:

    dataset_nora/anger/*.jpg
    dataset_nora/fear/*.jpg
    ...

USAGE
-----
    python src/build_manifest.py \
        --roots nora=data/processed/nora ferac=data/processed/ferac \
                talaat=data/processed/talaat hasibur=data/processed/hasibur \
        --source-priority talaat ferac nora hasibur \
        --out runs/manifest.csv

DEPENDENCIES
------------
    pip install pillow imagehash numpy pandas scikit-learn tqdm
    pip install torch timm opencv-python # photograph matching (audit_unique_images.py)
    pip install facenet-pytorch          # optional but STRONGLY recommended
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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

# Sanity check: every alias must map to a canonical class name
_alias_targets = set(LABEL_ALIASES.values())
_canonical_set = set(CANONICAL_CLASSES)
assert _alias_targets == _canonical_set, (
    f"LABEL_ALIASES maps to {_alias_targets} but CANONICAL_CLASSES is {_canonical_set}"
)

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
            # forward slashes: the manifest must load on Windows AND Linux/Kaggle
            rows.append({"path": p.as_posix(), "source": source,
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

def compute_hashes(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import imagehash
    except ImportError:
        sys.exit("pip install imagehash")

    from PIL import ImageOps

    md5s, phashes, sizes, bad = [], [], [], []
    for path in tqdm(df["path"], desc="hashing"):
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                sizes.append(f"{im.width}x{im.height}")
                phashes.append(str(imagehash.phash(im, hash_size=8)))
                md5s.append(hashlib.md5(im.tobytes()).hexdigest())
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


def flag_duplicates(df: pd.DataFrame, threshold: int = 6) -> pd.DataFrame:
    """Exact (pixel md5) and near (pHash Hamming <= threshold) duplicate flagging.

    Near-duplicates get a shared `dup_cluster`. Keep ONE representative per cluster
    for training, but never let two members of a cluster land in different CV folds.
    """
    df = df.copy()

    # exact
    df["is_exact_dup"] = df.duplicated(subset=["md5"], keep="first")

    # Warn about pixel-identical images with conflicting labels
    label_per_md5 = df.groupby("md5")["label"].nunique()
    n_conflicts = int((label_per_md5 > 1).sum())
    if n_conflicts:
        print(f"[WARN] {n_conflicts} pixel-identical image groups have CONFLICTING "
              f"labels across sources -- review these manually!", file=sys.stderr)

    # near: chunked brute force over 64-bit phashes for 100% recall
    bits = np.array([int(h, 16) for h in df["phash"]], dtype=np.uint64)
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

    # Brute force over all pairs in pure python (~56M pairs, about a minute at n=10k)
    bits_list = bits.tolist()
    for i in tqdm(range(n), desc="near-dup"):
        h = bits_list[i]
        for j in range(i + 1, n):
            if (h ^ bits_list[j]).bit_count() <= threshold:
                union(i, j)

    df["dup_cluster"] = [f"D{find(i):06d}" for i in range(n)]
    sizes = df["dup_cluster"].value_counts()
    df["dup_cluster_size"] = df["dup_cluster"].map(sizes)
    return df


# --------------------------------------------------------------------------------------
# 3. Photographs: augmentation-robust copy detection
# --------------------------------------------------------------------------------------
def mark_representatives(df: pd.DataFrame, source_priority: list[str]) -> pd.DataFrame:
    """Pick the ONE copy of each photograph that training and evaluation may use.

    Preference: the earliest source in `source_priority` (list the un-augmented
    originals first), then a file not named `*_aug_*`, then path order. Also flags
    photographs whose copies sit in different emotion folders -- the representative's
    label is then only one of several claimed labels and needs adjudication.
    """
    df = df.copy()
    rank = {s: i for i, s in enumerate(source_priority)}
    order = pd.DataFrame({
        "photo_id": df["photo_id"],
        "src": df["source"].map(rank).fillna(len(rank)),
        "aug": df["path"].map(lambda p: "_aug_" in Path(p).name),
        "path": df["path"],
    }).sort_values(["photo_id", "src", "aug", "path"])
    df["is_photo_rep"] = df.index.isin(order.drop_duplicates("photo_id").index)
    df["photo_label_conflict"] = df.groupby("photo_id")["label"].transform("nunique") > 1
    return df


def assign_photos(df: pd.DataFrame, cache_dir: str, min_inliers: int,
                  parents: dict[str, str], device: str = "auto") -> pd.DataFrame:
    """Add `photo_id` using the embedding-shortlist + SIFT/RANSAC matcher."""
    from audit_unique_images import filename_edges, match_pairs, photo_ids

    uniq, pairs, inl = match_pairs(df, cache_dir, device=device)
    df = df.copy()
    df["photo_id"] = photo_ids(df, uniq, pairs, inl, filename_edges(df, uniq, parents),
                               min_inliers)
    return df


# --------------------------------------------------------------------------------------
# 4. Identity (subject) clustering  -- the column that makes your splits defensible
# --------------------------------------------------------------------------------------
def infer_identities(df: pd.DataFrame, threshold: float = 0.55,
                     batch_size: int = 64, device: str = "auto") -> pd.DataFrame:
    """Cluster faces by identity using FaceNet embeddings + agglomerative clustering.

    ONE embedding per photograph (its representative), and every copy inherits that
    group. Embedding all files instead lets the augmented copies vote: heavy-noise
    copies of different children look alike to FaceNet and collapse into one giant
    "child", while copies of a single photo get scattered over several groups.

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
        print("[warn] facenet-pytorch not installed -- falling back to photo_id as "
              "the grouping key. This is WEAKER: it removes duplicate leakage but NOT "
              "subject leakage. Install facenet-pytorch before submitting.",
              file=sys.stderr)
        df = df.copy()
        df["group"] = df["photo_id"]
        df["group_source"] = "photo_id_fallback"
        return df

    from PIL import ImageOps
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    if device == "auto":
        device = ("cuda" if torch.cuda.is_available() else
                  "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    net = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    reps = df[df["is_photo_rep"]]
    embs, ok_photo = [], []
    buf, buf_photo = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(device)
        with torch.no_grad():
            e = net(x).cpu().numpy()
        embs.append(e)
        ok_photo.extend(buf_photo)
        buf.clear(); buf_photo.clear()

    for path, photo in zip(tqdm(reps["path"], desc="embedding"), reps["photo_id"]):
        try:
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB").resize((160, 160))
            t = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float()
            t = (t - 127.5) / 128.0
            buf.append(t); buf_photo.append(photo)
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

    group_of = {p: f"ID{c:05d}" for p, c in zip(ok_photo, clust.labels_)}
    df = df.copy()
    # every copy of a photograph inherits its group; an unreadable one stands alone
    df["group"] = [group_of.get(p, f"UNK_{p}") for p in df["photo_id"]]
    df["group_source"] = "facenet_agglomerative_per_photo"
    return df


# --------------------------------------------------------------------------------------
# 5. Audit report
# --------------------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    g = df.groupby("group")
    rep = {
        "n_images": int(len(df)),
        "n_exact_duplicates": int(df["is_exact_dup"].sum()),
        "n_images_in_near_dup_clusters": int((df["dup_cluster_size"] > 1).sum()),
        "n_near_dup_clusters_gt1": int(df[df["dup_cluster_size"] > 1]["dup_cluster"].nunique()),
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
        # pixel-identical images that got different labels across source datasets
        "n_label_conflicts_exact_dup": int(
            (df.groupby("md5")["label"].nunique() > 1).sum()),
    }

    # ---- photograph level: these, not the file counts above, are the dataset size ----
    r = df[df["is_photo_rep"]]
    ph = df.groupby("photo_id")
    per_group = r.groupby("group").size()
    clean = r[~r["photo_label_conflict"]]
    rep.update({
        "n_pixel_unique_images": int(df["md5"].nunique()),
        "n_phash_clusters": int(df["dup_cluster"].nunique()),
        "n_unique_photos": int(len(r)),
        "copies_per_photo_mean": float(ph.size().mean()),
        "unique_photos_per_source": {s: int(d["photo_id"].nunique())
                                     for s, d in df.groupby("source")},
        "photos_exclusive_to_source": {
            s: int(ph["source"].agg(lambda x: set(x) == {s}).sum())
            for s in sorted(df["source"].unique())},
        "photos_in_multiple_sources": int((ph["source"].nunique() > 1).sum()),
        "representative_source_counts": r["source"].value_counts().to_dict(),
        # label of the representative copy; arbitrary for the conflicting photos
        "class_counts_unique_photos": r["label"].value_counts().to_dict(),
        # the same photograph filed under different emotions by different files
        "photos_with_conflicting_labels": int(r["photo_label_conflict"].sum()),
        "class_counts_unique_photos_no_conflict": clean["label"].value_counts().to_dict(),
        "photos_per_group_mean": float(per_group.mean()),
        "photos_per_group_median": float(per_group.median()),
        "photos_per_group_max": int(per_group.max()),
        "n_single_photo_groups": int((per_group == 1).sum()),
    })

    # Loud warnings for issues that affect downstream splits
    n_span_labels = rep["groups_spanning_labels"]
    n_span_sources = rep["groups_spanning_sources"]
    n_groups = rep["n_groups"]
    if n_span_labels:
        print(f"\n[WARN] {n_span_labels}/{n_groups} identity groups "
              f"({100*n_span_labels/n_groups:.0f}%) span MULTIPLE LABELS. "
              f"This means the same child has different emotion labels across "
              f"images. Downstream StratifiedGroupKFold class balancing will be "
              f"approximate. Discuss this in the paper.", file=sys.stderr)
    if n_span_sources:
        print(f"[WARN] {n_span_sources}/{n_groups} identity groups "
              f"({100*n_span_sources/n_groups:.0f}%) appear in MULTIPLE SOURCE "
              f"DATASETS. Leave-One-Dataset-Out (LODO) evaluation is invalid.",
              file=sys.stderr)

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
    ap.add_argument("--skip-photo-match", action="store_true",
                    help="use pHash clusters as photo_id (fast, but misses augmented "
                         "copies -- for smoke tests only)")
    ap.add_argument("--photo-cache", default="runs/unique_audit",
                    help="where verified image pairs are cached")
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--parent-source", nargs="*", default=["hasibur=talaat"],
                    help="child=parent: `<stem>_aug_<k>` in child is a copy of `<stem>` "
                         "in the same label folder of parent")
    ap.add_argument("--source-priority", nargs="*", default=None,
                    help="sources in order of preference for a photo's representative "
                         "copy; put un-augmented originals first (default: --roots order)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    roots = dict(r.split("=", 1) for r in args.roots)
    priority = args.source_priority or list(roots)
    unknown = set(priority) - set(roots)
    if unknown:
        ap.error(f"--source-priority names unknown sources: {sorted(unknown)}")

    df = scan_roots(roots)
    print(f"[1/5] scanned {len(df)} images from {len(roots)} sources")

    df = compute_hashes(df)
    print("[2/5] hashes computed")

    df = flag_duplicates(df, args.phash_threshold)
    print(f"[3/5] {df['is_exact_dup'].sum()} exact dups, "
          f"{(df['dup_cluster_size'] > 1).sum()} images in near-dup clusters")

    if args.skip_photo_match:
        df["photo_id"] = df["dup_cluster"]
    else:
        df = assign_photos(df, args.photo_cache, args.min_inliers,
                           dict(p.split("=", 1) for p in args.parent_source), args.device)
    df = mark_representatives(df, priority)
    print(f"[4/5] {df['photo_id'].nunique()} distinct photographs")

    if args.skip_identity:
        df["group"] = df["photo_id"]
        df["group_source"] = "photo_id_only"
    else:
        df = infer_identities(df, args.identity_threshold, device=args.device)
    print(f"[5/5] {df['group'].nunique()} identity groups")

    df.to_csv(args.out, index=False)
    rep = audit(df)
    Path(args.audit_out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {args.out} and {args.audit_out}")


if __name__ == "__main__":
    main()