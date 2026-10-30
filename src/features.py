"""Hand-crafted feature extraction for the classical ML baseline (Phase 2).

For each 4-band image cube (BL, GR, RE, IR) we extract, per channel:
  - color statistics: mean, std, median, min, max, p10, p90, skew, kurtosis
  - GLCM texture: contrast, dissimilarity, homogeneity, energy, correlation, ASM
      (averaged over 4 directions, computed on a 32-level quantized channel)
plus 6 pairwise normalized-difference band ratios (one per channel pair), which
is the classical-ML analogue of the per-wavelength importance the SE block
gives the deep-learning side.
"""
import itertools
import os

import numpy as np
from scipy import stats as sstats
from skimage.feature import graycomatrix, graycoprops

from .config import CHANNEL_NAMES, CLASS_TO_IDX, CLASSES, OUTPUT_DIR
from .consent import require_consent
from .dataset import list_files

GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
GLCM_LEVELS = 32
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _color_stats(channel):
    flat = channel.astype(np.float64).ravel()
    return [
        flat.mean(), flat.std(), np.median(flat), flat.min(), flat.max(),
        np.percentile(flat, 10), np.percentile(flat, 90),
        sstats.skew(flat), sstats.kurtosis(flat),
    ]


def _glcm_features(channel):
    quantized = (channel.astype(np.float64) / 256.0 * GLCM_LEVELS).astype(np.uint8)
    quantized = np.clip(quantized, 0, GLCM_LEVELS - 1)
    glcm = graycomatrix(quantized, GLCM_DISTANCES, GLCM_ANGLES, levels=GLCM_LEVELS, symmetric=True, normed=True)
    return [graycoprops(glcm, prop).mean() for prop in GLCM_PROPS]


def feature_names():
    names = []
    for ch in CHANNEL_NAMES:
        names += [f"{ch}_{s}" for s in ["mean", "std", "median", "min", "max", "p10", "p90", "skew", "kurt"]]
    for ch in CHANNEL_NAMES:
        names += [f"{ch}_glcm_{p}" for p in GLCM_PROPS]
    for i, j in itertools.combinations(range(len(CHANNEL_NAMES)), 2):
        names.append(f"NDI_{CHANNEL_NAMES[i]}_{CHANNEL_NAMES[j]}")
    return names


def extract_features(arr):
    """arr: (H,W,4) uint8 -> 1D feature vector."""
    feats = []
    means = []
    for c in range(arr.shape[-1]):
        channel = arr[:, :, c]
        feats += _color_stats(channel)
        means.append(channel.astype(np.float64).mean())
    for c in range(arr.shape[-1]):
        feats += _glcm_features(arr[:, :, c])
    eps = 1e-6
    for i, j in itertools.combinations(range(len(means)), 2):
        feats.append((means[i] - means[j]) / (means[i] + means[j] + eps))
    return np.array(feats, dtype=np.float64)


def _cache_path(split):
    return os.path.join(OUTPUT_DIR, "features", f"{split}.npz")


def build_feature_dataset(split, force=False):
    require_consent()
    cache = _cache_path(split)
    if os.path.exists(cache) and not force:
        data = np.load(cache, allow_pickle=True)
        return data["X"], data["y"], list(data["paths"])

    items = list_files(split)
    X, y, paths = [], [], []
    for fp, label in items:
        arr = np.load(fp)
        X.append(extract_features(arr))
        y.append(label)
        paths.append(fp)
    X = np.stack(X)
    y = np.array(y)

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, X=X, y=y, paths=np.array(paths))
    return X, y, paths


if __name__ == "__main__":
    names = feature_names()
    print(f"{len(names)} features:", names)
    for split in ["train", "val", "test"]:
        X, y, paths = build_feature_dataset(split, force=True)
        print(split, X.shape, y.shape)
        assert X.shape[1] == len(names)
