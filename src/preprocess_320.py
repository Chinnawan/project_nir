"""Build Processed_Dataset_320/: the same RawA images as Processed_Dataset/ but at 320x320.

Identical to preprocess_pipeline.py (quadrant split, centre crop 512x512, stack BL/GR/RE/IR,
INTER_AREA) except the final resize target is 320 instead of 224, so small surface defects keep
more pixels. Mirrors Processed_Dataset/ file for file (same split, class folders, names), so
k-fold folds line up 1:1 with every other experiment. Does not touch Processed_Dataset/.

Usage: python -m src.preprocess_320
"""
import glob
import os
import re
import time

import cv2
import numpy as np

from .config import CLASSES, DATA_DIR, PROJECT_ROOT

RAW_DIR = os.path.join(PROJECT_ROOT, "images", "RawImages")
OUT_DIR = os.path.join(PROJECT_ROOT, "Processed_Dataset_320")
SIZE = 320


def crop_and_resize(quadrant, size=SIZE):
    cropped = quadrant[:, 64:576]  # 512x512 centre crop of the 512x640 quadrant (same as the 224 pipeline)
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)


def process(filepath, size=SIZE):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    assert img is not None and img.shape == (1024, 1280), filepath
    ir = crop_and_resize(img[0:512, 0:640], size)
    gr = crop_and_resize(img[0:512, 640:1280], size)
    bl = crop_and_resize(img[512:1024, 0:640], size)
    re_ = crop_and_resize(img[512:1024, 640:1280], size)
    return np.stack([bl, gr, re_, ir], axis=-1)  # BL, GR, RE, IR


def main():
    t0 = time.time()
    n = 0
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)
            for fp in sorted(glob.glob(os.path.join(DATA_DIR, split, cls, "*.npy"))):
                name = os.path.basename(fp)
                rawno = int(re.search(r"(\d+)", name).group(1))
                cube = process(os.path.join(RAW_DIR, f"RawA{rawno:04d}.tif"))
                np.save(os.path.join(OUT_DIR, split, cls, name), cube)
                n += 1
    assert n == 819, n

    # sanity: downsampling a 320 cube to 224 reproduces the existing 224 cube (same pipeline, same crop)
    a = np.load(glob.glob(os.path.join(DATA_DIR, "train", CLASSES[1], "*.npy"))[0])
    b = np.load(os.path.join(OUT_DIR, "train", CLASSES[1], os.path.basename(glob.glob(
        os.path.join(DATA_DIR, "train", CLASSES[1], "*.npy"))[0])))
    down = np.stack([cv2.resize(b[..., c], (224, 224), interpolation=cv2.INTER_AREA) for c in range(4)], axis=-1)
    print(f"wrote {n} cubes ({SIZE}x{SIZE}x4) to {OUT_DIR} in {time.time() - t0:.0f}s; "
          f"mean |320->224 vs original 224| = {np.abs(down.astype(float) - a.astype(float)).mean():.2f} gray levels")


if __name__ == "__main__":
    main()
