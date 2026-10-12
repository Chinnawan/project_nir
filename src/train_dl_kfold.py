"""K-Fold cross-validation for the EfficientNet-B0 variants.

The original train/val/test split (70/15/15) is small enough that results
vary noticeably between runs (see PROJECT_PLAN.md). This script pools all 819
images back together, re-splits them with StratifiedKFold (5 folds), and
retrains each requested variant once per fold — reusing the exact same
training loop (run_training) as train_dl.py so the procedure stays identical
to the original Phase 3/4 runs.

Does not touch train_dl.py's fixed-split outputs (outputs/dl/metrics.json,
*_best.pt, *_history.json) — results go to outputs/dl/kfold_results.json.

Variants:
  visible, nir, full, full_SE   -- original Phase 3/4 sweep (already run)
  nir_SE                        -- NEW: NIR-only (RE,IR) + SE block, never
                                    tried before (SE was only tried on full)
  nir_no_aug                    -- NEW: NIR-only with augmentation disabled,
                                    an ablation to separate "k-fold changed
                                    the numbers" from "augmentation changed
                                    the numbers" (both were introduced at once)

Usage:
  python -m src.train_dl_kfold --variant nir_SE       # just this one (5 folds)
  python -m src.train_dl_kfold --variant nir_no_aug   # just this one (5 folds)
  python -m src.train_dl_kfold                        # every variant missing
                                                        # from kfold_results.json
Results merge into outputs/dl/kfold_results.json (existing variants are kept,
never overwritten) unless --force re-runs a variant that already has all 5
folds recorded.
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import CLASSES, OUTPUT_DIR, PROJECT_ROOT, SEED
from .dataset import compute_stats_for_files, get_dataloaders_from_items, list_files
from .models import build_model
from .train_dl import DEVICE, run_training

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "kfold_results.json")
N_SPLITS = 5
VAL_FRACTION_OF_TRAINVAL = 0.176  # matches the original ~70/15 train/val ratio

SS_DIR = os.path.join(PROJECT_ROOT, "Processed_Dataset_SS")
CKPT_DIR = os.path.join(OUT, "kfold_ckpt")

# name -> (channel_group, use_se, augment_train, source, save_ckpt)
#   source "raw" = Processed_Dataset (RawA, no flat-field correction)
#   source "ss"  = Processed_Dataset_SS (SingleSpectra, flat-field corrected; same apples,
#                  same file names/splits, so folds line up 1:1 with the raw runs)
VARIANT_SPECS = {
    "visible": ("visible", False, True, "raw", False),
    "nir": ("nir", False, True, "raw", False),
    "full": ("full", False, True, "raw", False),
    "full_SE": ("full", True, True, "raw", False),
    "nir_SE": ("nir", True, True, "raw", False),         # NIR-only + SE block
    "nir_no_aug": ("nir", False, False, "raw", False),   # NIR-only, augmentation OFF (ablation)
    "nir_SE_ss": ("nir", True, True, "ss", True),        # nir_SE on SingleSpectra images; ckpts feed gradcam_mask_eval
    # single-band ablation: identical recipe to `nir` (no SE: SE on 1 channel is meaningless)
    "re_only": ("re", False, True, "raw", False),
    "ir_only": ("ir", False, True, "raw", False),
}


def build_pool(source="raw"):
    """All 819 images (train+val+test folders combined) as one (filepath, label) pool.
    For source="ss" the pool has the identical order/labels as the raw pool, so
    StratifiedKFold puts every apple in the same fold in both experiments."""
    root = SS_DIR if source == "ss" else None
    pool = list_files("train", root) + list_files("val", root) + list_files("test", root)
    labels = np.array([label for _, label in pool])
    if source == "ss":
        raw_pool = list_files("train") + list_files("val") + list_files("test")
        assert [os.path.basename(f) for f, _ in pool] == [os.path.basename(f) for f, _ in raw_pool], \
            "Processed_Dataset_SS does not mirror Processed_Dataset (run src.preprocess_ss)"
    return pool, labels


def get_fold_splits(pool, pool_labels):
    """StratifiedKFold splits over the pool, deterministic given SEED — shared
    with ensemble_kfold.py so both scripts see the exact same fold assignment."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(pool)), pool_labels))


def run_fold(fold_idx, trainval_items, trainval_labels, test_items, channel_group, use_se,
             augment_train=True, name=None, ckpt_path=None):
    name = name or f"{channel_group}{'_SE' if use_se else ''}"
    torch.manual_seed(SEED)

    train_items, val_items = train_test_split(
        trainval_items, test_size=VAL_FRACTION_OF_TRAINVAL,
        stratify=trainval_labels, random_state=SEED,
    )

    train_loader, val_loader, test_loader = get_dataloaders_from_items(
        train_items, val_items, test_items, channel_group=channel_group, batch_size=16,
        augment_train=augment_train,
    )
    model = build_model(channel_group, num_classes=len(CLASSES), pretrained=True, use_se=use_se).to(DEVICE)

    result = run_training(
        model, train_loader, val_loader, test_loader,
        name=f"{name}_fold{fold_idx}", epochs=100, patience=10, lr=1e-4, save_dir=None,
        channel_group=channel_group,
    )

    if ckpt_path:  # run_training leaves the best-val-loss weights loaded in `model`
        mean, std = compute_stats_for_files(train_items)  # same stats the train loader used
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mean": mean.tolist(), "std": std.tolist()}, ckpt_path)

    del model, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "fold": fold_idx,
        "test_accuracy": result["test_accuracy"],
        "test_macro_f1": result["test_macro_f1"],
        "epochs_trained": result["epochs_trained"],
        "channel_weights": result["channel_weights"],
        "per_class_f1": [result["test_report"][c]["f1-score"] for c in CLASSES],
        "confusion_matrix": result["confusion_matrix"],
        "test_predictions": result["test_predictions"],  # same order as the fold's test_idx
        "test_labels": result["test_labels"],
    }


def _load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            data = json.load(f)
        return data.get("per_fold", {}), data.get("summary", {})
    return {}, {}


def _save_results(per_fold, summary):
    os.makedirs(OUT, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"per_fold": per_fold, "summary": summary}, f, indent=2)


def _summarize(fold_results):
    f1s = np.array([r["test_macro_f1"] for r in fold_results])
    accs = np.array([r["test_accuracy"] for r in fold_results])
    return {
        "macro_f1_mean": float(f1s.mean()), "macro_f1_std": float(f1s.std()),
        "accuracy_mean": float(accs.mean()), "accuracy_std": float(accs.std()),
        "per_fold_macro_f1": f1s.tolist(),
    }


def run_variant(name, force=False):
    if name not in VARIANT_SPECS:
        raise ValueError(f"unknown variant {name!r}, choose from {list(VARIANT_SPECS)}")
    channel_group, use_se, augment_train, source, save_ckpt = VARIANT_SPECS[name]

    per_fold, summary = _load_results()
    if not force and len(per_fold.get(name, [])) == N_SPLITS:
        print(f"[{name}] already has all {N_SPLITS} folds in {RESULTS_PATH}, skipping "
              f"(pass --force to re-run)")
        return per_fold, summary

    pool, pool_labels = build_pool(source)
    fold_splits = get_fold_splits(pool, pool_labels)

    print(f"\n### Variant {name} (channel_group={channel_group}, use_se={use_se}, "
          f"augment_train={augment_train}, source={source}) -- {N_SPLITS} folds ###")

    fold_results = []
    t0 = time.time()
    for fold_idx, (trainval_idx, test_idx) in enumerate(fold_splits):
        print(f"\n=== {name} | fold {fold_idx + 1}/{N_SPLITS} ===")
        trainval_items = [pool[i] for i in trainval_idx]
        trainval_labels = pool_labels[trainval_idx]
        test_items = [pool[i] for i in test_idx]

        ckpt_path = os.path.join(CKPT_DIR, f"{name}_fold{fold_idx}.pt") if save_ckpt else None
        fold_result = run_fold(fold_idx, trainval_items, trainval_labels, test_items,
                                channel_group, use_se, augment_train=augment_train, name=name,
                                ckpt_path=ckpt_path)
        fold_results.append(fold_result)

        per_fold[name] = fold_results
        summary[name] = _summarize(fold_results)
        _save_results(per_fold, summary)

    elapsed = time.time() - t0
    print(f"\n[{name}] done in {elapsed / 60:.1f} min -- "
          f"macro-F1 = {summary[name]['macro_f1_mean']:.4f} +/- {summary[name]['macro_f1_std']:.4f}  "
          f"(acc = {summary[name]['accuracy_mean']:.4f} +/- {summary[name]['accuracy_std']:.4f})")
    return per_fold, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANT_SPECS), default=None,
                         help="run just this variant (5 folds). Omit to run every "
                              "variant not already fully present in kfold_results.json")
    parser.add_argument("--force", action="store_true",
                         help="re-run even if this variant already has all 5 folds saved")
    args = parser.parse_args()

    pool, pool_labels = build_pool()
    print(f"Pool: {len(pool)} images, class counts {np.bincount(pool_labels)}")

    targets = [args.variant] if args.variant else list(VARIANT_SPECS)
    per_fold, summary = _load_results()
    pending = [v for v in targets if args.force or len(per_fold.get(v, [])) < N_SPLITS]

    if not pending:
        print("Nothing to do -- all requested variants already have 5 folds saved. "
              "Pass --force to re-run.")
    else:
        n_runs = len(pending) * N_SPLITS
        print(f"Running {len(pending)} variant(s) x {N_SPLITS} folds = {n_runs} training runs: {pending}")
        print("Each run trains from scratch (pretrained ImageNet backbone) with early stopping "
              "(patience=10). Based on prior runs on this machine, expect roughly 1.5-2.5 min "
              f"per fold, so ~{n_runs * 2}-{n_runs * 3} min total.")

    for name in pending:
        per_fold, summary = run_variant(name, force=args.force)

    print("\n=== Summary: test macro-F1 mean +/- std across 5 folds ===")
    for name in targets:
        if name in summary:
            s = summary[name]
            print(f"  {name:12s}: macro-F1 = {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f}  "
                  f"(acc = {s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f})  "
                  f"folds={np.round(s['per_fold_macro_f1'], 3).tolist()}")

    if summary:
        best = max(summary, key=lambda n: summary[n]["macro_f1_mean"])
        print(f"\nBest variant overall by mean macro-F1: {best} "
              f"({summary[best]['macro_f1_mean']:.4f} +/- {summary[best]['macro_f1_std']:.4f})")


if __name__ == "__main__":
    main()
