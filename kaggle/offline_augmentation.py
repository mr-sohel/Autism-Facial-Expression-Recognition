import os
import glob
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

# ==============================================================================
# Configuration
# ==============================================================================
# Detect if running in Kaggle or locally
if os.path.exists("/kaggle"):
    # Target the raw dataset, not the MTCNN one, so MTCNN can process the augmented images later
    DATASET_DIR = "/kaggle/input/datasets/mrsohel/autism-dataset/dataset/train"
    OUT_DIR = "/kaggle/working/dataset_augmented/train"
else:
    _repo = Path(__file__).resolve().parent.parent
    DATASET_DIR = str(_repo / "dataset" / "train")
    OUT_DIR = str(_repo / "dataset_augmented" / "train")

# We want all classes to have at least this many images
TARGET_COUNT = 600

# Strong augmentations to create diverse synthetic copies
augment_pipeline = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=20),
    T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=10),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
    T.RandomPerspective(distortion_scale=0.2, p=0.5),
])

def augment_class(class_name, target_count):
    src_dir = Path(DATASET_DIR) / class_name
    dst_dir = Path(OUT_DIR) / class_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    if not src_dir.exists():
        print(f"Skipping {class_name}, source dir not found.")
        return

    images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.png")) + list(src_dir.glob("*.jpeg"))
    current_count = len(images)
    
    if current_count == 0:
        print(f"No images found for {class_name}.")
        return

    # First, copy all original images
    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img.save(dst_dir / img_path.name)
        
    print(f"[{class_name.upper()}] Copied {current_count} original images.")
    
    if current_count >= target_count:
        print(f"[{class_name.upper()}] Already has {current_count} images (target: {target_count}). No augmentation needed.")
        return
        
    # Generate new images until we hit the target count
    needed = target_count - current_count
    print(f"[{class_name.upper()}] Generating {needed} synthetic augmented images...")
    
    generated = 0
    while generated < needed:
        # Loop through existing images and create variations
        for img_path in images:
            if generated >= needed:
                break
                
            img = Image.open(img_path).convert("RGB")
            # Apply random transformation
            aug_img = augment_pipeline(img)
            
            # Save new image
            new_name = f"aug_{generated}_{img_path.name}"
            aug_img.save(dst_dir / new_name)
            generated += 1

    print(f"[{class_name.upper()}] Finished! Total images now: {len(list(dst_dir.glob('*')))}")

def copy_unmodified_splits():
    """Copy the valid and test splits unmodified so the full dataset structure is intact."""
    import shutil
    for split in ["valid", "test"]:
        src = Path(DATASET_DIR).parent / split
        dst = Path(OUT_DIR).parent / split
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[*] Copied {split} split unmodified.")

if __name__ == "__main__":
    print("="*60)
    print(" OFFLINE DATA AUGMENTATION (CLASS BALANCING)")
    print("="*60)
    
    # We explicitly want to boost the minority classes up to ~600 (like Joy)
    classes_to_process = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
    
    for c in classes_to_process:
        augment_class(c, TARGET_COUNT)
        
    copy_unmodified_splits()
    
    print("\n[*] Offline augmentation complete!")
    print(f"[*] Your balanced raw dataset is now located at: {Path(OUT_DIR).parent}")
    print("[*] NEXT STEP: Update preprocess_faces.py to point RAW_DATASET to this new folder, then re-run it.")
