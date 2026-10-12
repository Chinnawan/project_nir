import glob
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import CHANNEL_GROUPS, CLASS_TO_IDX, CLASSES, DATA_DIR, OUTPUT_DIR, SEED


def list_files(split, root=None):
    """Return [(filepath, label_idx), ...] for a split, sorted for reproducibility.
    `root` selects another dataset tree with the same layout (e.g. Processed_Dataset_SS)."""
    items = []
    for cls in CLASSES:
        pattern = os.path.join(root or DATA_DIR, split, cls, "*.npy")
        for fp in sorted(glob.glob(pattern)):
            items.append((fp, CLASS_TO_IDX[cls]))
    return items


def _channel_stats_path():
    return os.path.join(OUTPUT_DIR, "channel_stats.json")


def compute_stats_for_files(files):
    """Per-channel mean/std (all 4 bands) computed over an arbitrary list of
    (filepath, label) items. Not cached — used for e.g. k-fold splits where the
    train pool changes per fold."""
    sums = np.zeros(4, dtype=np.float64)
    sq_sums = np.zeros(4, dtype=np.float64)
    n_pixels = 0
    for fp, _ in files:
        arr = np.load(fp).astype(np.float64) / 255.0
        sums += arr.sum(axis=(0, 1))
        sq_sums += (arr ** 2).sum(axis=(0, 1))
        n_pixels += arr.shape[0] * arr.shape[1]

    mean = sums / n_pixels
    var = sq_sums / n_pixels - mean ** 2
    std = np.sqrt(np.clip(var, 1e-8, None))
    return mean, std


def compute_channel_stats():
    """Per-channel mean/std over the fixed train split (all 4 bands), cached to disk."""
    cache_path = _channel_stats_path()
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            stats = json.load(f)
        return np.array(stats["mean"]), np.array(stats["std"])

    mean, std = compute_stats_for_files(list_files("train"))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    return mean, std


class ApplesDataset(Dataset):
    """Loads preprocessed (224,224,4) uint8 .npy cubes, selects a channel group,
    normalizes with train-set stats, and applies light augmentation for training."""

    def __init__(self, split, channel_group="full", augment=None, mean=None, std=None, items=None):
        assert channel_group in CHANNEL_GROUPS, f"unknown channel_group {channel_group}"
        self.split = split
        self.channels = CHANNEL_GROUPS[channel_group]
        self.items = items if items is not None else list_files(split)
        self.augment = augment if augment is not None else (split == "train")

        if mean is None or std is None:
            mean, std = compute_channel_stats()
        self.mean = np.asarray(mean, dtype=np.float32)[self.channels].reshape(-1, 1, 1)
        self.std = np.asarray(std, dtype=np.float32)[self.channels].reshape(-1, 1, 1)

    def __len__(self):
        return len(self.items)

    def _augment(self, arr):
        # arr: (C,H,W) float32 in [0,1], applied to the train split only
        if random.random() < 0.5:
            arr = arr[:, :, ::-1]  # horizontal flip
        if random.random() < 0.5:
            arr = arr[:, ::-1, :]  # vertical flip
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # 90/180/270 degrees
            arr = np.rot90(arr, k, axes=(1, 2))
        if random.random() < 0.5:
            factor = random.uniform(0.85, 1.15)  # brightness scaling
            arr = np.clip(arr * factor, 0.0, 1.0)
        return np.ascontiguousarray(arr)

    def __getitem__(self, idx):
        fp, label = self.items[idx]
        arr = np.load(fp).astype(np.float32) / 255.0  # (H,W,4)
        arr = np.transpose(arr, (2, 0, 1))  # (4,H,W)
        arr = arr[self.channels]

        if self.augment:
            arr = self._augment(arr)

        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr.copy()), label


def class_counts(split="train"):
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    for _, label in list_files(split):
        counts[label] += 1
    return counts


def get_dataloaders(channel_group="full", batch_size=32, num_workers=0, use_weighted_sampler=True):
    torch.manual_seed(SEED)
    mean, std = compute_channel_stats()

    train_ds = ApplesDataset("train", channel_group, augment=True, mean=mean, std=std)
    val_ds = ApplesDataset("val", channel_group, augment=False, mean=mean, std=std)
    test_ds = ApplesDataset("test", channel_group, augment=False, mean=mean, std=std)

    if use_weighted_sampler:
        counts = class_counts("train")
        weight_per_class = 1.0 / counts
        sample_weights = [weight_per_class[label] for _, label in train_ds.items]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    counts = class_counts("train")
    class_weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights


def get_dataloaders_from_items(train_items, val_items, test_items, channel_group="full",
                                batch_size=16, num_workers=0, use_weighted_sampler=True,
                                augment_train=True):
    """Same as get_dataloaders() but built from explicit (filepath, label) item
    lists instead of the fixed train/val/test folders — used by k-fold CV, where
    the pool is re-split per fold. Channel stats are computed from train_items
    only (no leakage from val/test of that fold). Set augment_train=False for
    ablation runs that need the exact same split logic without augmentation."""
    mean, std = compute_stats_for_files(train_items)

    train_ds = ApplesDataset("train", channel_group, augment=augment_train, mean=mean, std=std, items=train_items)
    val_ds = ApplesDataset("val", channel_group, augment=False, mean=mean, std=std, items=val_items)
    test_ds = ApplesDataset("test", channel_group, augment=False, mean=mean, std=std, items=test_items)

    if use_weighted_sampler:
        counts = np.zeros(len(CLASSES), dtype=np.int64)
        for _, label in train_items:
            counts[label] += 1
        weight_per_class = 1.0 / counts
        sample_weights = [weight_per_class[label] for _, label in train_ds.items]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    mean, std = compute_channel_stats()
    print("channel mean:", mean)
    print("channel std:", std)
    for split in ["train", "val", "test"]:
        print(split, "counts:", class_counts(split))
