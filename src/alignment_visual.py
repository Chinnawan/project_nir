"""N5 pre-check: look at the band misregistration with your own eyes (not only phase correlation).

For 3 apples, the exact 224x224 training inputs (raw Processed_Dataset cube and SingleSpectra cube) are shown as
false-colour overlays: magenta = BL (450 nm), green = RE (750 nm). Where the two bands line up the apple looks grey/
white; a shift shows as a magenta fringe on one side and a green fringe on the other. A third column shifts RE by the
phase-correlation estimate to check that the estimate is what removes the fringes.

  python -m src.alignment_visual  ->  outputs/figures/private/alignment_overlay.png  (gitignored: dataset pixels)
"""
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR
from .round2 import Context

OUT = os.path.join(OUTPUT_DIR, "figures", "private", "alignment_overlay.png")
APPLES = ["RawA0004", "RawA0076", "RawA0500"]


def norm(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)


def overlay(a, b):
    a, b = norm(a), norm(b)
    return np.stack([a, b, a], axis=-1)  # magenta = a, green = b


def grad(img):
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def shift_est(a, b):
    h, w = a.shape
    (dx, dy), _ = cv2.phaseCorrelate(grad(a), grad(b), cv2.createHanningWindow((w, h), cv2.CV_32F))
    return dx, dy


def main():
    ctx = Context()
    names = [os.path.basename(p[0]).replace(".npy", "") for p in ctx.pool_raw]
    pools = {"raw": ctx.pool("raw"), "ss": ctx.pool("ss")}
    fig, axes = plt.subplots(len(APPLES), 4, figsize=(16, 4.2 * len(APPLES)))
    report = []
    for r, name in enumerate(APPLES):
        i = names.index(name)
        raw, ss = (np.load(pools[s][i][0]) for s in ("raw", "ss"))
        dx, dy = shift_est(raw[..., 0], raw[..., 2])
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        re_fixed = cv2.warpAffine(raw[..., 2], m, raw.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
        sdx, sdy = shift_est(ss[..., 0], ss[..., 2])
        panels = [(overlay(raw[..., 0], raw[..., 2]), f"{name} raw (training input)\nmagenta=BL, green=RE"),
                  (overlay(raw[..., 0], re_fixed), f"raw, RE shifted back by ({-dx:+.1f}, {-dy:+.1f}) px"),
                  (overlay(ss[..., 0], ss[..., 2]), f"SingleSpectra (training input)\nphase-corr shift ({sdx:+.1f}, {sdy:+.1f}) px"),
                  (overlay(raw[..., 2], raw[..., 3]), "raw, magenta=RE, green=IR")]
        for c, (img, title) in enumerate(panels):
            axes[r, c].imshow(img)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
        report.append(f"{name}: raw BL->RE shift ({dx:+.1f},{dy:+.1f}) px, ss ({sdx:+.1f},{sdy:+.1f}) px")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print("\n".join(report))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
