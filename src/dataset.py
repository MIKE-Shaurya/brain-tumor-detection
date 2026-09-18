"""Dataset loading and transform pipelines for the MRI classification task.

Expects data organized as an ImageFolder layout:

    data/train/<class_name>/*.jpg
    data/test/<class_name>/*.jpg
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import config


def get_transforms(train: bool):
    """Return the torchvision transform pipeline for train or eval mode."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD),
    ])


def get_dataloaders(batch_size: int = config.BATCH_SIZE, num_workers: int = 2):
    """Build train/val/test DataLoaders from the data/ directory.

    The train/ folder is split into train + validation subsets;
    test/ is used only for final evaluation.
    """
    full_train_dataset = datasets.ImageFolder(
        config.TRAIN_DIR, transform=get_transforms(train=True)
    )

    # Sanity check that folder class order matches config.CLASS_NAMES
    assert full_train_dataset.classes == sorted(config.CLASS_NAMES), (
        f"Class mismatch: found {full_train_dataset.classes}, "
        f"expected {sorted(config.CLASS_NAMES)}. Update config.CLASS_NAMES "
        f"or your data folder names."
    )

    val_size = int(len(full_train_dataset) * config.VAL_SPLIT)
    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_subset, val_subset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )

    # Validation should use eval-time transforms (no augmentation), so wrap
    # a second dataset instance pointed at the same folder but with eval
    # transforms, and reuse the split indices.
    eval_transform_dataset = datasets.ImageFolder(
        config.TRAIN_DIR, transform=get_transforms(train=False)
    )
    val_subset.dataset = eval_transform_dataset

    test_dataset = datasets.ImageFolder(
        config.TEST_DIR, transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, full_train_dataset.classes


def compute_class_weights(dataset_classes, train_dir=config.TRAIN_DIR):
    """Compute inverse-frequency class weights for a weighted loss,
    to counteract mild class imbalance in the MRI dataset.
    """
    import os

    counts = []
    for cls in dataset_classes:
        cls_dir = os.path.join(train_dir, cls)
        counts.append(len(os.listdir(cls_dir)))

    counts = torch.tensor(counts, dtype=torch.float)
    weights = counts.sum() / (len(counts) * counts)
    return weights
