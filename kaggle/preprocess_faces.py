"""
==============================================================================
  Offline Face Preprocessing — MTCNN + CLAHE
==============================================================================
  Run this ONCE before training to produce a clean, face-cropped dataset.
  Output is a parallel directory tree that is a drop-in replacement for the
  raw dataset in any training script.

  Steps applied per image:
    1. MTCNN face detection  -> crop to largest/most-confident face + 20% pad
    2. CLAHE                 -> normalise local contrast (handles multi-dataset
                                lighting variation from 4 source datasets)
    3. Fallback              -> if no face detected, centre-crop 85% of image
    4. Resize to 224x224     -> matches training pipeline input size

  Requirements (install once):
    pip install facenet-pytorch opencv-python pillow

  Usage — local:
    python kaggle/preprocess_faces.py

  Usage — Kaggle notebook (add a code cell BEFORE the training cell):
    import subprocess
    subprocess.run(["pip", "install", "-q", "facenet-pytorch"], check=True)
    exec(open("/kaggle/input/<your-dataset>/preprocess_faces.py").read())

  After running, update DATA_DIR / DATASET_DIR in your training scripts to
  point at the output directory (default: dataset_mtcnn/).
==============================================================================
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")


# ==============================================================================
# Paths  (auto-detects Kaggle vs local)
# ==============================================================================
if os.path.exists("/kaggle"):
    RAW_DATASET = "/kaggle/input/datasets/mrsohel/autism-dataset/dataset"
    OUT_DATASET = "/kaggle/working/dataset_mtcnn"
else:
    _repo       = Path(__file__).resolve().parent.parent   # repo root
    RAW_DATASET = str(_repo / "dataset")
    OUT_DATASET = str(_repo / "dataset_mtcnn")

SPLITS   = ["train", "valid", "test"]
CLASSES  = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# ==============================================================================
# Config
# ==============================================================================
FACE_PADDING  = 0.20   # fractional padding added around detected face box
FALLBACK_FRAC = 0.85   # centre-crop fraction when no face found
CLAHE_CLIP    = 2.0    # CLAHE clipLimit  (higher = stronger contrast boost)
CLAHE_TILE    = 8      # CLAHE tileGridSize in pixels
OUTPUT_SIZE   = 224    # final image size saved to disk
MIN_FACE_PX   = 30     # reject detections smaller than this (false positives)


# ==============================================================================
# MTCNN — lazy import
# ==============================================================================
def load_mtcnn(device="cpu"):
    try:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(
            keep_all=True,                    # detect all faces; we pick the best
            min_face_size=MIN_FACE_PX,
            thresholds=[0.6, 0.7, 0.7],      # P-Net / R-Net / O-Net thresholds
            post_process=False,
            device=device,
        )
        print(f"[OK] MTCNN loaded on {device}")
        return mtcnn
    except ImportError:
        print("[!] facenet-pytorch not installed.")
        print("    Run:  pip install facenet-pytorch")
        print("    Falling back to centre-crop only (no face detection).")
        return None


# ==============================================================================
# Step 1 — Face detection & crop
# ==============================================================================
def detect_and_crop(img_pil, mtcnn):
    """
    Returns (cropped_PIL_image, status_string).
    status is one of: 'detected' | 'fallback_no_face' | 'fallback_small'
    """
    w, h = img_pil.size

    if mtcnn is not None:
        try:
            boxes, probs = mtcnn.detect(img_pil)
        except Exception:
            boxes, probs = None, None

        if boxes is not None and len(boxes) > 0:
            # Choose face with highest detection confidence
            best = int(np.argmax(probs))
            x1, y1, x2, y2 = boxes[best]
            bw, bh = x2 - x1, y2 - y1

            # Reject implausibly small detections
            if bw < MIN_FACE_PX or bh < MIN_FACE_PX:
                return _centre_crop(img_pil), "fallback_small"

            # Add padding around the face
            x1 = max(0,  x1 - bw * FACE_PADDING)
            y1 = max(0,  y1 - bh * FACE_PADDING)
            x2 = min(w,  x2 + bw * FACE_PADDING)
            y2 = min(h,  y2 + bh * FACE_PADDING)

            return img_pil.crop((x1, y1, x2, y2)), "detected"

    return _centre_crop(img_pil), "fallback_no_face"


def _centre_crop(img_pil):
    w, h  = img_pil.size
    cw    = int(w * FALLBACK_FRAC)
    ch    = int(h * FALLBACK_FRAC)
    x0    = (w - cw) // 2
    y0    = (h - ch) // 2
    return img_pil.crop((x0, y0, x0 + cw, y0 + ch))


# ==============================================================================
# Step 2 — CLAHE contrast normalisation (L channel of LAB colour space)
# ==============================================================================
def apply_clahe(img_pil):
    img_np  = np.array(img_pil.convert("RGB"))
    img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe   = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                               tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])   # only L channel
    result  = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


# ==============================================================================
# Main loop
# ==============================================================================
def process_dataset(raw_root, out_root, mtcnn, use_clahe=True):
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    print(f"\n{'='*64}")
    print(f"  Source  : {raw_root}")
    print(f"  Output  : {out_root}")
    print(f"  CLAHE   : {'ON' if use_clahe else 'OFF'}")
    print(f"  Padding : {int(FACE_PADDING*100)}%  |  Fallback crop: {int(FALLBACK_FRAC*100)}%")
    print(f"{'='*64}\n")

    total_stats = defaultdict(int)

    for split in SPLITS:
        split_stats = defaultdict(int)

        for cls_name in CLASSES:
            src_dir = raw_root / split / cls_name
            dst_dir = out_root / split / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src_dir.exists():
                continue

            img_paths = [p for p in src_dir.iterdir()
                         if p.suffix.lower() in IMG_EXTS]

            for img_path in img_paths:
                dst_path = dst_dir / img_path.name

                # Re-run safe: skip already-processed images
                if dst_path.exists():
                    split_stats["skipped"] += 1
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")

                    # 1. Face detection + crop
                    cropped, status = detect_and_crop(img, mtcnn)
                    split_stats[status] += 1

                    # 2. Optional CLAHE
                    if use_clahe:
                        cropped = apply_clahe(cropped)

                    # 3. Resize and save
                    cropped = cropped.resize((OUTPUT_SIZE, OUTPUT_SIZE),
                                            Image.LANCZOS)
                    cropped.save(dst_path, quality=95)

                except Exception as e:
                    split_stats["error"] += 1
                    # Hard fallback: save resized original
                    try:
                        Image.open(img_path).convert("RGB") \
                             .resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS) \
                             .save(dst_path)
                    except Exception:
                        pass

        # Per-split summary line
        processed = (split_stats["detected"]
                     + split_stats["fallback_no_face"]
                     + split_stats["fallback_small"])
        detected  = split_stats["detected"]
        fallback  = split_stats["fallback_no_face"] + split_stats["fallback_small"]
        det_pct   = detected / max(1, processed) * 100

        print(f"  [{split:5s}]  Processed={processed:4d} | "
              f"Face detected={detected:4d} ({det_pct:5.1f}%) | "
              f"Fallback={fallback:3d} | "
              f"Errors={split_stats['error']:2d} | "
              f"Skipped={split_stats['skipped']:4d}")

        for k, v in split_stats.items():
            total_stats[k] += v

    # Overall summary
    total_processed = (total_stats["detected"]
                       + total_stats["fallback_no_face"]
                       + total_stats["fallback_small"])
    overall_pct = total_stats["detected"] / max(1, total_processed) * 100
    print(f"\n  [TOTAL]  Processed={total_processed} | "
          f"Face detected={total_stats['detected']} ({overall_pct:.1f}%) | "
          f"Fallback={total_stats['fallback_no_face'] + total_stats['fallback_small']} | "
          f"Errors={total_stats['error']}")

    return str(out_root)


# ==============================================================================
# Per-class report (train split only)
# ==============================================================================
def per_class_report(raw_root, out_root):
    raw_root, out_root = Path(raw_root), Path(out_root)
    print(f"\n{'─'*52}")
    print(f"  Per-class images saved (train split)")
    print(f"{'─'*52}")
    for cls_name in CLASSES:
        src = raw_root / "train" / cls_name
        dst = out_root / "train" / cls_name
        n_src = len([p for p in src.iterdir()
                     if p.suffix.lower() in IMG_EXTS]) if src.exists() else 0
        n_dst = len([p for p in dst.iterdir()
                     if p.suffix.lower() in IMG_EXTS]) if dst.exists() else 0
        filled = int(n_dst / max(1, n_src) * 30)
        bar    = "█" * filled + "░" * (30 - filled)
        pct    = n_dst / max(1, n_src) * 100
        print(f"  {cls_name:<10} [{bar}]  {n_dst:3d}/{n_src:3d}  ({pct:.0f}%)")
    print(f"{'─'*52}")


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MTCNN + CLAHE offline face preprocessing"
    )
    parser.add_argument("--raw",      default=RAW_DATASET,
                        help="Raw dataset root directory")
    parser.add_argument("--out",      default=OUT_DATASET,
                        help="Output directory for processed dataset")
    parser.add_argument("--no-clahe", action="store_true",
                        help="Disable CLAHE contrast normalisation")
    parser.add_argument("--device",   default="cpu",
                        help="Device for MTCNN: 'cpu' or 'cuda'")

    # parse_known_args() ignores Jupyter/Kaggle kernel args in sys.argv
    args, _ = parser.parse_known_args()

    mtcnn    = load_mtcnn(device=args.device)
    out_path = process_dataset(
        raw_root  = args.raw,
        out_root  = args.out,
        mtcnn     = mtcnn,
        use_clahe = not args.no_clahe,
    )
    per_class_report(args.raw, out_path)

    print(f"\n[*] Done! Update DATA_DIR in your training scripts to:")
    print(f"      {out_path}\n")
