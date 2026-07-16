import os
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image


CLASS_NAMES = ["anger", "fear", "joy", "natural", "sadness", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


def get_train_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
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


def get_val_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FacialExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []

        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            class_idx = CLASS_TO_IDX[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    self.samples.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def compute_class_weights(dataset):
    counts = Counter(dataset.labels)
    total = len(dataset.labels)
    weights = []
    for i in range(NUM_CLASSES):
        count = counts.get(i, 1)
        weights.append(total / (NUM_CLASSES * count))
    return torch.FloatTensor(weights)


def get_class_counts(dataset):
    return Counter(dataset.labels)


def create_weighted_sampler(dataset):
    class_counts = get_class_counts(dataset)
    sample_weights = [1.0 / class_counts[label] for label in dataset.labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)


def get_dataloaders(data_dir, batch_size=16, img_size=224, num_workers=0):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_ds = FacialExpressionDataset(train_dir, transform=get_train_transforms(img_size))
    val_ds = FacialExpressionDataset(val_dir, transform=get_val_transforms(img_size))
    test_ds = FacialExpressionDataset(test_dir, transform=get_val_transforms(img_size))

    train_sampler = create_weighted_sampler(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    class_weights = compute_class_weights(train_ds)

    return train_loader, val_loader, test_loader, class_weights, train_ds, val_ds, test_ds
