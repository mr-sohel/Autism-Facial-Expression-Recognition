#!/usr/bin/env python3
"""
audit_unique_images.py
======================
Count how many DISTINCT PHOTOGRAPHS the merged corpus really contains, and give every
image a `photo_id` shared by all copies of the same photograph.

WHY THIS FILE EXISTS
--------------------
build_manifest.py finds near-duplicates with a 64-bit pHash. pHash survives recompression
and resizing but NOT the augmentations these sources were built with (mirror, crop,
rotation, colour shift, heavy Gaussian noise). The Hasibur source is the Talaat source
augmented 10x -- the filenames say so (`0368_aug_7.jpg` <- `0368.jpg`) -- yet pHash
splits each 10-image family into ~6 "unique" clusters. Every count downstream of that
(unique images, images per child, fold sizes) is inflated.

METHOD
------
1. Embed every pixel-unique image and its mirror with a pretrained backbone. This is
   ONLY a shortlist: on its own it finds the true parent 82% of the time.
2. For each image, take its top-K neighbours and verify each pair geometrically:
   SIFT keypoints -> ratio test -> RANSAC similarity transform -> count inliers. Two
   different photographs of a face do not share a rigid keypoint layout; an augmented
   copy does. Mirrored copies are handled by matching against the flipped image too.
3. Union every pair with >= `--min-inliers` inliers. Connected components = photographs.
4. Validate the matcher against the Hasibur filename families (ground truth that needs
   no labelling), THEN add those filename links as certain edges: `<stem>_aug_<k>` files
   in one label folder are one photograph, and `--parent-source hasibur=talaat` ties each
   family to the same-named file in the source it was augmented from.

USAGE
-----
    python src/audit_unique_images.py --manifest runs/manifest.csv --out-dir runs/unique_audit

Outputs (in --out-dir):
    photo_ids.csv        path, md5, photo_id, photo_size   (join to the manifest on path)
    pair_inliers.csv     every verified pair and its inlier count (reusable)
    unique_audit.json    the numbers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FAMILY_RE = re.compile(r"^(\d+)_aug_\d+$")
DEFAULT_MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


# --------------------------------------------------------------------------------------
# 1. Embedding shortlist
# --------------------------------------------------------------------------------------
def pick_device(device: str) -> str:
    import torch
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def embed(paths, model_name: str, device: str, batch_size: int = 64):
    import timm
    import torch
    from PIL import Image, ImageOps
    from tqdm import tqdm

    net = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_data_config({}, model=net)
    size = cfg["input_size"][-1]
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(device)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(device)

    def load(p):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size),
                                                                   Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

    e, ef = [], []
    for i in tqdm(range(0, len(paths), batch_size), desc="embedding"):
        x = torch.stack([load(p) for p in paths[i:i + batch_size]]).to(device).float()
        x = (x / 255 - mean) / std
        with torch.no_grad():
            e.append(net(x).float().cpu().numpy())
            ef.append(net(torch.flip(x, dims=[3])).float().cpu().numpy())
    e, ef = np.concatenate(e), np.concatenate(ef)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    ef /= np.linalg.norm(ef, axis=1, keepdims=True)
    return e, ef


def shortlist(e, ef, k: int) -> np.ndarray:
    """Unordered candidate pairs (i < j): each image's top-k neighbours, mirror-aware."""
    n = len(e)
    pairs = set()
    for lo in range(0, n, 1024):
        s = np.maximum(e[lo:lo + 1024] @ e.T, e[lo:lo + 1024] @ ef.T)
        s[np.arange(len(s)), np.arange(lo, lo + len(s))] = -1
        top = np.argpartition(-s, k, axis=1)[:, :k]
        for r, row in enumerate(top):
            i = lo + r
            pairs.update((min(i, j), max(i, j)) for j in row.tolist())
    return np.array(sorted(pairs), dtype=np.int64)


# --------------------------------------------------------------------------------------
# 2. Geometric verification
# --------------------------------------------------------------------------------------
def sift_features(path: str, n_features: int):
    """SIFT on a denoised, contrast-normalised 256x256 grey image, plus its mirror.

    The blur matters: several sources add heavy Gaussian noise, and without it most
    keypoints land on noise. CLAHE removes the brightness/colour augmentations.
    """
    sift = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02)
    g = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None, None
    g = cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
    out = []
    for img in (g, cv2.flip(g, 1)):
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            out.append(None)
        else:   # uint8 descriptors: SIFT values are integral, and this is 4x smaller
            out.append((np.float32([k.pt for k in kp]), des.astype(np.uint8)))
    return out[0], out[1]


def count_inliers(fa, fb) -> int:
    if fa is None or fb is None or len(fa[0]) < 4 or len(fb[0]) < 4:
        return 0
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fa[1].astype(np.float32),
                                                  fb[1].astype(np.float32), k=2)
    good = [m[0] for m in matches
            if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 4:
        return 0
    a = fa[0][[g.queryIdx for g in good]]
    b = fb[0][[g.trainIdx for g in good]]
    m, mask = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                          ransacReprojThreshold=4.0)
    if m is None:
        return 0
    scale = float(np.hypot(m[0, 0], m[0, 1]))
    if not 0.5 < scale < 2.0:       # augmentations crop/zoom mildly; reject wild fits
        return 0
    return int(mask.sum())


def verify_pairs(paths, pairs, n_features: int, workers: int) -> np.ndarray:
    from tqdm import tqdm
    with ThreadPoolExecutor(workers) as ex:
        feats = list(tqdm(ex.map(lambda p: sift_features(p, n_features), paths),
                          total=len(paths), desc="sift"))

    def one(ij):
        i, j = ij
        return max(count_inliers(feats[i][0], feats[j][0]),
                   count_inliers(feats[i][0], feats[j][1]))

    with ThreadPoolExecutor(workers) as ex:
        return np.fromiter(tqdm(ex.map(one, pairs.tolist(), chunksize=256),
                                total=len(pairs), desc="verify"),
                           dtype=np.int32, count=len(pairs))


# --------------------------------------------------------------------------------------
# 3. Components
# --------------------------------------------------------------------------------------
def components(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in pairs.tolist():
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])


# --------------------------------------------------------------------------------------
# 4. Report
# --------------------------------------------------------------------------------------
def report(df: pd.DataFrame) -> dict:
    g = df.groupby("photo_id")
    first = df.drop_duplicates("photo_id")
    rep = {
        "n_images": int(len(df)),
        "n_pixel_unique_images": int(df["md5"].nunique()),
        "n_phash_clusters": int(df["dup_cluster"].nunique()) if "dup_cluster" in df else None,
        "n_unique_photos": int(df["photo_id"].nunique()),
        "copies_per_photo_mean": float(g.size().mean()),
        "copies_per_photo_max": int(g.size().max()),
        "unique_photos_per_source": {s: int(d["photo_id"].nunique())
                                     for s, d in df.groupby("source")},
        "photos_by_n_sources": {int(k): int(v) for k, v in
                                g["source"].nunique().value_counts().sort_index().items()},
        "photos_exclusive_to_source": {
            s: int((g["source"].agg(lambda x: set(x) == {s})).sum())
            for s in sorted(df["source"].unique())},
        "photos_per_label_first_seen": first["label"].value_counts().to_dict(),
        # the same photograph filed under different emotions
        "photos_with_conflicting_labels": int((g["label"].nunique() > 1).sum()),
    }
    if "group" in df:
        # copies of ONE photograph assigned to DIFFERENT children by the identity step:
        # every such photo can sit on both sides of a "subject-independent" split
        rep["photos_split_across_identity_groups"] = int((g["group"].nunique() > 1).sum())
    return rep


def _stems(df: pd.DataFrame):
    stem = df["path"].map(lambda p: Path(p.replace("\\", "/")).stem)
    fam_stem = stem.map(lambda s: m.group(1) if (m := FAMILY_RE.match(s)) else None)
    return stem, fam_stem


def filename_edges(df: pd.DataFrame, uniq: pd.DataFrame, parents: dict) -> np.ndarray:
    """Certain same-photo links read from filenames, as index pairs into `uniq`."""
    col = {m: i for i, m in enumerate(uniq["md5"])}
    stem, fam_stem = _stems(df)
    edges = []
    for (src, label, st), d in df[fam_stem.notna()].groupby(
            ["source", "label", fam_stem[fam_stem.notna()]]):
        members = [col[m] for m in d["md5"]]
        par = parents.get(src)
        if par:
            hit = df[(df["source"] == par) & (df["label"] == label) & (stem == st)]
            members += [col[m] for m in hit["md5"]]
        edges += [(members[0], m) for m in members[1:]]
    return np.array(edges, dtype=np.int64).reshape(-1, 2)


def validate_families(df: pd.DataFrame, parents: dict) -> dict | None:
    """Ground truth from filenames: `<stem>_aug_<k>` in one label folder is one photo."""
    stem, fam_stem = _stems(df)
    out = {}
    for src, d in df[fam_stem.notna()].groupby("source"):
        fam = d.assign(stem=fam_stem[d.index]).groupby(["label", "stem"])
        sizes = fam.size()
        if sizes.max() < 5:          # only sources whose families are large enough to test
            continue
        n_ids = fam["photo_id"].nunique()
        out[src] = {
            "n_families": int(len(sizes)),
            "family_size": int(sizes.median()),
            "families_fully_recovered": int((n_ids == 1).sum()),
            "families_fully_recovered_frac": float((n_ids == 1).mean()),
            "mean_photo_ids_per_family": float(n_ids.mean()),
        }
        if "dup_cluster" in d:
            out[src]["mean_phash_clusters_per_family"] = float(
                fam["dup_cluster"].nunique().mean())
        par = parents.get(src)
        if par:     # image-level: is each augmented copy tied to its named parent?
            p = df[df["source"] == par].assign(stem=stem)
            pid = p.drop_duplicates(["label", "stem"]).set_index(["label", "stem"])["photo_id"]
            want = pd.MultiIndex.from_arrays([d["label"], fam_stem[d.index]]).map(pid)
            out[src]["images_linked_to_named_parent_frac"] = float(
                (d["photo_id"].to_numpy() == np.asarray(want)).mean())
    return out or None


def match_pairs(df: pd.DataFrame, cache_dir, model: str = DEFAULT_MODEL, top_k: int = 50,
                sift_features: int = 500, workers: int = 10, device: str = "auto"):
    """Shortlist + verify. Returns (pixel-unique rows, candidate pairs, inlier counts).

    The verified pairs are cached in `cache_dir`, keyed on the exact image set and the
    parameters that affect them, so a changed corpus is re-matched rather than silently
    reusing stale pairs.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    uniq = df.drop_duplicates("md5").reset_index(drop=True)
    paths = uniq["path"].tolist()
    key = {"images": hashlib.sha1("".join(sorted(uniq["md5"])).encode()).hexdigest(),
           "model": model, "top_k": top_k, "sift_features": sift_features}
    pair_path, key_path = cache_dir / "pair_inliers.csv", cache_dir / "pair_inliers.key.json"
    print(f"[photo] {len(df)} images, {len(uniq)} pixel-unique")

    if pair_path.exists() and key_path.exists() and json.loads(key_path.read_text()) == key:
        pr = pd.read_csv(pair_path)
        col = {m: i for i, m in enumerate(uniq["md5"])}
        pairs = np.c_[pr["md5_a"].map(col), pr["md5_b"].map(col)].astype(np.int64)
        inl = pr["inliers"].to_numpy()
        print(f"[photo] reusing {len(pr)} verified pairs from {pair_path}")
    else:
        e, ef = embed(paths, model, pick_device(device))
        pairs = shortlist(e, ef, top_k)
        print(f"[photo] {len(pairs)} candidate pairs (top-{top_k})")
        t0 = time.time()
        inl = verify_pairs(paths, pairs, sift_features, workers)
        print(f"[photo] verified in {(time.time() - t0) / 60:.1f} min")
        md5 = uniq["md5"].to_numpy()
        pd.DataFrame({"md5_a": md5[pairs[:, 0]], "md5_b": md5[pairs[:, 1]],
                      "inliers": inl}).to_csv(pair_path, index=False)
        key_path.write_text(json.dumps(key))
    return uniq, pairs, inl


def photo_ids(df, uniq, pairs, inl, known, min_inliers: int) -> pd.Series:
    """One `photo_id` per row of `df`: components of verified pairs + filename links."""
    edges = pairs[inl >= min_inliers]
    if known is not None and len(known):
        edges = np.concatenate([edges, known])
    comp = components(len(uniq), edges)
    ids = pd.Series([f"P{c:05d}" for c in comp], index=uniq["md5"].to_numpy())
    return df["md5"].map(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="runs/manifest.csv")
    ap.add_argument("--out-dir", default="runs/unique_audit")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="timm backbone used only to shortlist candidate pairs")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--min-inliers", type=int, default=12)
    ap.add_argument("--sift-features", type=int, default=500)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--parent-source", nargs="*", default=["hasibur=talaat"],
                    help="child=parent: `<stem>_aug_<k>` in child is a copy of `<stem>` "
                         "in the same label folder of parent")
    args = ap.parse_args()
    parents = dict(p.split("=", 1) for p in args.parent_source)

    out = Path(args.out_dir)
    df = pd.read_csv(args.manifest)
    uniq, pairs, inl = match_pairs(df, out, args.model, args.top_k, args.sift_features,
                                   args.workers, args.device)
    known = filename_edges(df, uniq, parents)

    def assign(th, use_filenames=True):
        return photo_ids(df, uniq, pairs, inl, known if use_filenames else None, th)

    # score the matcher ALONE against the filename ground truth, before trusting it
    df["photo_id"] = assign(args.min_inliers, use_filenames=False)
    validation = validate_families(df, parents)
    matcher_only = int(df["photo_id"].nunique())

    df["photo_id"] = assign(args.min_inliers)
    df["photo_size"] = df["photo_id"].map(df["photo_id"].value_counts())
    df[["path", "md5", "photo_id", "photo_size"]].to_csv(out / "photo_ids.csv", index=False)

    rep = report(df)
    rep["params"] = {k: getattr(args, k) for k in
                     ("model", "top_k", "min_inliers", "sift_features")}
    rep["params"]["parent_source"] = parents
    rep["n_filename_edges"] = int(len(known))
    rep["n_unique_photos_matcher_only"] = matcher_only
    rep["matcher_validation_vs_filenames"] = validation
    # how much the headline number depends on the one free parameter
    rep["n_unique_photos_by_min_inliers"] = {
        int(th): int(assign(th).nunique()) for th in (8, 10, 12, 15, 20, 30)}
    (out / "unique_audit.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"\nwrote {out / 'photo_ids.csv'} and {out / 'unique_audit.json'}")


if __name__ == "__main__":
    main()
