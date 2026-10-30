"""Quantitative Grad-CAM check against the pixel-level defect masks.

Uses the 5 per-fold checkpoints of nir_SE_ss (trained on the SingleSpectra images,
whose geometry matches the masks in segmentations/Segmented_defects). For every
Defect image, Grad-CAM (w.r.t. the ground-truth class) is computed ONLY with the
fold model that held that image out as its test fold (out-of-fold), so no image
is explained by a model that trained on it. Every Defect image lands in exactly
one test fold, so all 246 are covered once.

Metrics (mask and CAM both at 224x224, mask padded/resized with the same transform
as the images):
  - Pointing Game : argmax of the CAM falls inside the defect mask (hit rate)
  - IoU           : CAM >= its 80th percentile (top 20% of pixels) vs the mask
Null baselines: the same CAM scored against the mask of a different, randomly
chosen Defect image (20 draws per image). "null_any" draws from all Defect images,
"null_cls" only from images of the same class (Internal/Surface) so the mask size
distribution matches (Surface masks are ~2% of the image, Internal ~9% -- a null that
mixes them is unfair to Surface). A real result must beat the matched null_cls.

Usage: python -m src.gradcam_mask_eval
Needs: outputs/dl/kfold_ckpt/nir_SE_ss_fold{0..4}.pt  (python -m src.train_dl_kfold --variant nir_SE_ss)
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .config import CLASSES, OUTPUT_DIR, PROJECT_ROOT, SEED
from .consent import require_consent
from .dataset import ApplesDataset
from .models import build_model
from .preprocess_ss import MAPPING_PATH, pad_square_resize
from .train_dl_kfold import CKPT_DIR, N_SPLITS, build_pool, get_fold_splits

VARIANT = "nir_SE_ss"
MASK_DIR = os.path.join(PROJECT_ROOT, "segmentations", "Segmented_defects", "jona", "im", "segmentedBW")
KLEYNEN_PATH = os.path.join(PROJECT_ROOT, "segmentations", "KleynenDefectClasses_Simplified.txt")
OUT_JSON = os.path.join(OUTPUT_DIR, "dl", "gradcam_mask_eval.json")
OVERLAY_DIR = os.path.join(OUTPUT_DIR, "dl", "gradcam_mask")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_FRAC = 0.20
N_NULL = 20
KLEYNEN_NAMES = {1: "rejection", 2: "bruise", 3: "serious", 4: "slight"}


def load_kleynen():
    require_consent()
    with open(KLEYNEN_PATH, encoding="latin-1") as f:
        return {int(a): int(b) for a, b in re.findall(r"Defect(\d+)\s+(\d)", f.read())}


def load_mask(defect_no):
    require_consent()
    m = cv2.imread(os.path.join(MASK_DIR, f"SBW_Defect{defect_no:04d}.bmp"), cv2.IMREAD_UNCHANGED)
    assert m is not None, defect_no
    m = m[..., 0] if m.ndim == 3 else m
    return pad_square_resize((m > 127).astype(np.uint8) * 255) > 127


def pointing_hit(cam, mask):
    y, x = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return bool(mask[y, x]), (int(x), int(y))


def top_iou(cam, mask, frac=TOP_FRAC):
    thr = np.quantile(cam, 1.0 - frac)
    pred = cam >= thr if thr > 0 else cam > 0
    union = np.logical_or(pred, mask).sum()
    return float(np.logical_and(pred, mask).sum() / union) if union else 0.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def boot_ci(x, n_boot=2000, seed=SEED):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summarize(recs):
    n = len(recs)
    if n == 0:
        return {"n": 0}
    hits = np.array([r["hit"] for r in recs])
    ious = np.array([r["iou"] for r in recs])
    return {
        "n": n,
        "pointing_hit_rate": float(hits.mean()),
        "pointing_hit_ci95": wilson(int(hits.sum()), n),
        "null_pointing_hit_rate": float(np.mean([r["null_hit"] for r in recs])),
        "null_cls_pointing_hit_rate": float(np.mean([r["null_cls_hit"] for r in recs])),
        "iou_mean": float(ious.mean()), "iou_std": float(ious.std()), "iou_ci95": boot_ci(ious),
        "null_iou_mean": float(np.mean([r["null_iou"] for r in recs])),
        "null_cls_iou_mean": float(np.mean([r["null_cls_iou"] for r in recs])),
        "mask_area_frac_mean": float(np.mean([r["mask_frac"] for r in recs])),
        "pred_correct_rate": float(np.mean([r["correct"] for r in recs])),
    }


def save_overlay(path, ir, cam, mask, peak, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base = np.stack([ir] * 3, axis=-1)
    over = (0.5 * base + 0.5 * heat).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, t in ((axes[0], base, "IR band + defect mask"), (axes[1], over, "Grad-CAM (GT class)")):
        im = img.copy()
        cv2.drawContours(im, cnts, -1, (0, 255, 0), 1)
        ax.imshow(im)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    axes[1].plot(peak[0], peak[1], "w*", markersize=12, markeredgecolor="k")
    fig.suptitle(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    global OUT_JSON, OVERLAY_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=-1,
                    help="index into model.backbone.features used as the Grad-CAM layer: -1 = last block "
                         "(7x7 map, default), 5 = last 14x14 block")
    ap.add_argument("--out", default=None, help="results json (default outputs/dl/gradcam_mask_eval.json)")
    ap.add_argument("--overlay-dir", default=None)
    args = ap.parse_args()
    if args.out:
        OUT_JSON = args.out
    if args.overlay_dir:
        OVERLAY_DIR = args.overlay_dir
    ckpts = [os.path.join(CKPT_DIR, f"{VARIANT}_fold{f}.pt") for f in range(N_SPLITS)]
    missing = [c for c in ckpts if not os.path.exists(c)]
    if missing:
        raise SystemExit(f"missing checkpoints (run: python -m src.train_dl_kfold --variant {VARIANT}):\n" +
                         "\n".join(missing))

    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    kleynen = load_kleynen()
    pool, labels = build_pool("ss")
    splits = get_fold_splits(pool, labels)

    recs, cams, masks, irs, peaks = [], [], [], [], []
    for fold, (_, test_idx) in enumerate(splits):
        ckpt = torch.load(ckpts[fold], map_location=DEVICE, weights_only=False)
        model = build_model("nir", num_classes=len(CLASSES), pretrained=False, use_se=True).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        cam_algo = GradCAM(model=model, target_layers=[model.backbone.features[args.layer]])

        for i in test_idx:
            fp, label = pool[i]
            rawno = int(re.search(r"(\d+)", os.path.basename(fp)).group(1))
            info = mapping[str(rawno)]
            if info["defect_no"] is None:
                continue
            ds = ApplesDataset("test", "nir", augment=False, mean=ckpt["mean"], std=ckpt["std"],
                                items=[(fp, label)])
            x, _ = ds[0]
            xb = x.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = int(model(xb).argmax(1).item())
            cam = cam_algo(input_tensor=xb, targets=[ClassifierOutputTarget(label)])[0]
            mask = load_mask(info["defect_no"])
            cube = np.load(fp)
            ir = cube[..., 3]
            hit, peak = pointing_hit(cam, mask)
            recs.append({
                "defect_no": info["defect_no"], "rawno": rawno, "fold": fold, "tissue": info["tissue"],
                "cls": CLASSES[label], "kleynen_class": kleynen[info["defect_no"]],
                "pred": CLASSES[pred], "correct": pred == label, "hit": hit, "iou": top_iou(cam, mask),
                "mask_frac": float(mask.mean()), "peak_xy": peak,
            })
            cams.append(cam.astype(np.float32))
            masks.append(mask)
            irs.append(ir)
            peaks.append(peak)
        del model, cam_algo
        print(f"fold {fold}: {sum(r['fold'] == fold for r in recs)} defect images explained (out-of-fold)")

    assert len(recs) == 246 and len({r["defect_no"] for r in recs}) == 246, len(recs)

    rng = np.random.RandomState(SEED)
    n = len(recs)
    for i, r in enumerate(recs):
        others = rng.choice([j for j in range(n) if j != i], N_NULL, replace=False)
        r["null_hit"] = float(np.mean([masks[j][peaks[i][1], peaks[i][0]] for j in others]))
        r["null_iou"] = float(np.mean([top_iou(cams[i], masks[j]) for j in others]))
        same = [j for j in range(n) if j != i and recs[j]["cls"] == r["cls"]]
        others_c = rng.choice(same, N_NULL, replace=False)
        r["null_cls_hit"] = float(np.mean([masks[j][peaks[i][1], peaks[i][0]] for j in others_c]))
        r["null_cls_iou"] = float(np.mean([top_iou(cams[i], masks[j]) for j in others_c]))

    def by(key_fn):
        groups = {}
        for r in recs:
            groups.setdefault(key_fn(r), []).append(r)
        return {str(k): summarize(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}

    results = {
        "variant": VARIANT, "gradcam_layer_index": args.layer, "n_defect_images": n, "top_fraction": TOP_FRAC, "n_null_draws": N_NULL,
        "overall": summarize(recs),
        "by_class": by(lambda r: r["cls"]),
        "by_kleynen_class": by(lambda r: f"{r['kleynen_class']} ({KLEYNEN_NAMES[r['kleynen_class']]})"),
        "by_prediction": by(lambda r: "pred correct" if r["correct"] else "pred wrong"),
        "by_class_and_prediction": by(lambda r: f"{r['cls']} / {'correct' if r['correct'] else 'wrong'}"),
        "by_tissue_type": by(lambda r: r["tissue"]),
        "per_image": recs,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    os.makedirs(OVERLAY_DIR, exist_ok=True)
    pick = np.random.RandomState(SEED)
    for cls in ("Internal_Defect", "Surface_Defect"):
        for outcome in (True, False):
            idx = [i for i, r in enumerate(recs) if r["cls"] == cls and r["hit"] == outcome]
            for i in pick.permutation(idx)[:3]:
                r = recs[i]
                save_overlay(os.path.join(OVERLAY_DIR, f"{cls}_{'hit' if outcome else 'miss'}_Defect{r['defect_no']:04d}.png"),
                              irs[i], cams[i], masks[i], peaks[i],
                              f"Defect{r['defect_no']:04d} {r['tissue']} ({cls}) pred={r['pred']} "
                              f"hit={r['hit']} IoU={r['iou']:.2f}")

    def row(name, s):
        lo, hi = s["pointing_hit_ci95"]
        return (f"  {name:30s} n={s['n']:3d}  hit {s['pointing_hit_rate']*100:5.1f}% [{lo*100:4.1f},{hi*100:5.1f}] "
                f"(null any {s['null_pointing_hit_rate']*100:4.1f}% / same-cls {s['null_cls_pointing_hit_rate']*100:4.1f}%)  "
                f"IoU {s['iou_mean']:.3f} (null any {s['null_iou_mean']:.3f} / same-cls {s['null_cls_iou_mean']:.3f})  "
                f"mask {s['mask_area_frac_mean']*100:4.1f}%")

    print("\n=== Grad-CAM vs defect mask (out-of-fold, nir_SE_ss) ===")
    print(row("ALL", results["overall"]))
    for sec in ("by_class", "by_kleynen_class", "by_prediction", "by_class_and_prediction", "by_tissue_type"):
        print(f"-- {sec}")
        for k, s in results[sec].items():
            print(row(k, s))
    print(f"\nsaved -> {OUT_JSON}\noverlays -> {OVERLAY_DIR}")


if __name__ == "__main__":
    main()
