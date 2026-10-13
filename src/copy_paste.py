"""Defect Copy-Paste on SingleSpectra cubes (224x224x4, BL/GR/RE/IR), on-the-fly.

A real defect is cut from a donor apple (pixel-level mask from segmentations/...SBW_DefectNNNN.bmp,
processed with the same pad+resize as the images) and pasted at the same position in ALL four bands
onto a Healthy apple. Pasting is multiplicative, per band:
    target_b' = target_b * (1 + alpha * (ratio_b - 1)),   ratio_b = donor_b / (donor surface level_b)
where the donor surface level is the median of a ring of apple surface around the defect and alpha
is the soft (Gaussian) defect mask. So the defect keeps its own contrast/texture but takes the
brightness of the target apple's surface.

Rules (PROJECT_PLAN.md, round 2 #6):
  - donors and targets come ONLY from the training items passed in (never val/test of the fold)
  - target = Healthy tissue (not Stem/Calix); defect placed >= `min_edge_dist` px inside the apple
  - random position, rotation 0-360, scale 0.8-1.2; label = class of the donor defect
  - donors Rot / Russet / Other / Hail_damage_perf are drawn with weight 2
Uses only Python's `random` (seeded by repro.set_seed) and deterministic cv2 ops.
"""
import os
import random

import cv2
import numpy as np

from .config import PROJECT_ROOT
from .preprocess_ss import pad_square_resize

MASK_DIR = os.path.join(PROJECT_ROOT, "segmentations", "Segmented_defects", "jona", "im", "segmentedBW")
UPWEIGHT_TISSUES = {"Rot", "Russet", "Other", "Hail_damage_perf"}
APPLE_IR_THRESHOLD = 50


def load_defect_mask(defect_no):
    m = cv2.imread(os.path.join(MASK_DIR, f"SBW_Defect{defect_no:04d}.bmp"), cv2.IMREAD_UNCHANGED)
    assert m is not None, defect_no
    m = m[..., 0] if m.ndim == 3 else m
    return pad_square_resize((m > 127).astype(np.uint8) * 255) > 127


def apple_region(cube):
    """uint8 (H,W) {0,1}: filled apple silhouette from the IR band."""
    m = (cube[..., 3] > APPLE_IR_THRESHOLD).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 1, -1)
    return out


class _Donor:
    __slots__ = ("fp", "defect_no", "label", "tissue", "ratio", "alpha", "mask", "centroid", "radius")


def _prepare_donor(fp, defect_no, label, tissue, cube, mask):
    apple = apple_region(cube).astype(bool)
    mask_u8 = mask.astype(np.uint8)
    ring = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(bool)
    ring &= ~cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(bool)
    ring &= apple
    if ring.sum() < 50:
        ring = apple & ~mask
    level = np.array([np.median(cube[..., b][ring]) for b in range(4)], dtype=np.float32)
    level = np.maximum(level, 1.0)
    ratio_full = np.clip(cube.astype(np.float32) / level, 0.2, 3.0)

    ys, xs = np.nonzero(mask)
    m = 8
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, mask.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, mask.shape[1])
    d = _Donor()
    d.fp, d.defect_no, d.label, d.tissue = fp, defect_no, label, tissue
    d.ratio = np.ascontiguousarray(ratio_full[y0:y1, x0:x1])
    d.mask = np.ascontiguousarray(mask_u8[y0:y1, x0:x1])
    d.alpha = cv2.GaussianBlur(d.mask.astype(np.float32), (0, 0), 1.5)
    cy, cx = ys.mean() - y0, xs.mean() - x0
    d.centroid = (float(cx), float(cy))
    d.radius = float(np.sqrt((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2).max()) + 2.0
    return d


class DefectCopyPaster:
    def __init__(self, donor_specs, min_edge_dist=20, max_tries=20):
        """donor_specs: [(fp, defect_no, label3, tissue)] -- must all be TRAINING items."""
        self.donors, self.weights = [], []
        for fp, defect_no, label, tissue in donor_specs:
            cube = np.load(fp)
            self.donors.append(_prepare_donor(fp, defect_no, label, tissue, cube, load_defect_mask(defect_no)))
            self.weights.append(2.0 if tissue in UPWEIGHT_TISSUES else 1.0)
        self.min_edge_dist = min_edge_dist
        self.max_tries = max_tries
        self.n_attempts = 0
        self.n_success = 0
        self.last_info = None

    def apply(self, cube, rng=random):
        """cube: uint8 (H,W,4). Returns (new_cube, donor_label) or None if no valid placement."""
        self.n_attempts += 1
        H, W, _ = cube.shape
        dist = cv2.distanceTransform(apple_region(cube), cv2.DIST_L2, 3)
        for _ in range(self.max_tries):
            donor = rng.choices(self.donors, weights=self.weights)[0]
            s = rng.uniform(0.8, 1.2)
            ang = rng.uniform(0.0, 360.0)
            ys, xs = np.nonzero(dist >= self.min_edge_dist + donor.radius * s)
            if len(ys) == 0:
                continue
            k = rng.randrange(len(ys))
            cy, cx = float(ys[k]), float(xs[k])
            mcx, mcy = donor.centroid
            M = cv2.getRotationMatrix2D((mcx, mcy), ang, s)
            M[0, 2] += cx - mcx
            M[1, 2] += cy - mcy
            alpha = cv2.warpAffine(donor.alpha, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
            alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
            ratio = cv2.warpAffine(donor.ratio, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(1, 1, 1, 1))
            out = cube.astype(np.float32) * (1.0 + alpha[..., None] * (ratio - 1.0))
            self.n_success += 1
            self.last_info = {"donor_fp": donor.fp, "defect_no": donor.defect_no, "tissue": donor.tissue,
                              "center": (cx, cy), "scale": s, "angle": ang}
            return np.clip(out, 0, 255).astype(np.uint8), donor.label
        return None
