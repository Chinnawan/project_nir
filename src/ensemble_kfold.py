"""Ensemble the 5 per-fold models (soft-voting) for the two variants that
matter most for the thesis (nir, full_SE), evaluated on the original
single-split test set (outputs/dl/kfold_results.json only kept metrics, not
checkpoints, so this retrains+saves those 10 models: 2 variants x 5 folds).

Caveat (see PROJECT_PLAN.md): the 5-fold split pools ALL 819 images (including
the original test set), so there is no data left that is guaranteed unseen by
every fold's model. For each eval image we track, per fold, whether it was
held out as that fold's test set (out-of-fold) or part of that fold's
train/val (in-sample) and report this transparently:
  - "ensemble" = soft-vote average of all 5 fold models (mix of in-sample +
    1 out-of-fold model per image) -- optimistic, reported for reference only.
  - "oof_single_model" = for each image, use only the 1 fold model that held
    it out as test -- the unbiased estimate, directly comparable to the plain
    k-fold numbers already in kfold_results.json.

Usage:
  python -m src.ensemble_kfold train <variant> <fold_idx>   # train+save one fold
  python -m src.ensemble_kfold evaluate                      # ensemble + report
  python -m src.ensemble_kfold                                # train all missing + evaluate
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import CLASSES, OUTPUT_DIR, SEED
from .dataset import ApplesDataset, compute_stats_for_files, get_dataloaders_from_items, list_files
from .models import build_model
from .train_dl import DEVICE, run_training
from .train_dl_kfold import N_SPLITS, VAL_FRACTION_OF_TRAINVAL, build_pool, get_fold_splits

CKPT_DIR = os.path.join(OUTPUT_DIR, "dl", "kfold_ckpt")
OUT = os.path.join(OUTPUT_DIR, "dl")
VARIANTS = ["nir", "full_SE"]


def _variant_config(name):
    use_se = name.endswith("_SE")
    channel_group = name[:-3] if use_se else name
    return channel_group, use_se


def _ckpt_path(variant, fold_idx):
    return os.path.join(CKPT_DIR, f"{variant}_fold{fold_idx}.pt")


def train_fold(variant, fold_idx):
    ckpt_path = _ckpt_path(variant, fold_idx)
    if os.path.exists(ckpt_path):
        print(f"[{variant} fold {fold_idx}] checkpoint already exists at {ckpt_path}, skipping")
        return

    pool, pool_labels = build_pool()
    fold_splits = get_fold_splits(pool, pool_labels)
    trainval_idx, test_idx = fold_splits[fold_idx]
    trainval_items = [pool[i] for i in trainval_idx]
    trainval_labels = pool_labels[trainval_idx]
    test_items = [pool[i] for i in test_idx]

    channel_group, use_se = _variant_config(variant)
    print(f"\n=== Training {variant} fold {fold_idx} (for ensembling) ===")

    torch.manual_seed(SEED)
    train_items, val_items = train_test_split(
        trainval_items, test_size=VAL_FRACTION_OF_TRAINVAL,
        stratify=trainval_labels, random_state=SEED,
    )
    mean, std = compute_stats_for_files(train_items)
    train_loader, val_loader, test_loader = get_dataloaders_from_items(
        train_items, val_items, test_items, channel_group=channel_group, batch_size=16,
    )
    model = build_model(channel_group, num_classes=len(CLASSES), pretrained=True, use_se=use_se).to(DEVICE)

    result = run_training(
        model, train_loader, val_loader, test_loader,
        name=f"{variant}_fold{fold_idx}", epochs=100, patience=10, lr=1e-4, save_dir=None,
        channel_group=channel_group,
    )
    print(f"[{variant} fold {fold_idx}] this fold's own test macro-F1={result['test_macro_f1']:.4f} "
          f"(matches kfold_results.json)")

    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean.tolist(), "std": std.tolist()}, ckpt_path)
    print(f"[{variant} fold {fold_idx}] saved checkpoint -> {ckpt_path}")


def _predict_probs(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(DEVICE))
            all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def evaluate():
    pool, pool_labels = build_pool()
    fold_splits = get_fold_splits(pool, pool_labels)
    pool_index_by_path = {fp: i for i, (fp, _) in enumerate(pool)}

    eval_items = list_files("test")  # original single-split test set (123 images)
    eval_labels = np.array([label for _, label in eval_items])
    n_eval = len(eval_items)

    results = {}
    for variant in VARIANTS:
        channel_group, use_se = _variant_config(variant)
        missing = [f for f in range(N_SPLITS) if not os.path.exists(_ckpt_path(variant, f))]
        if missing:
            print(f"[{variant}] missing checkpoints for folds {missing} -- run `train` for them first")
            continue

        fold_probs = np.zeros((N_SPLITS, n_eval, len(CLASSES)), dtype=np.float64)
        is_oof = np.zeros((N_SPLITS, n_eval), dtype=bool)  # True = this image was this fold's held-out test image

        for fold_idx in range(N_SPLITS):
            ckpt = torch.load(_ckpt_path(variant, fold_idx), map_location=DEVICE)
            model = build_model(channel_group, num_classes=len(CLASSES), pretrained=False, use_se=use_se).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])

            ds = ApplesDataset("test", channel_group, augment=False,
                                mean=ckpt["mean"], std=ckpt["std"], items=eval_items)
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            fold_probs[fold_idx] = _predict_probs(model, loader)

            _, test_idx = fold_splits[fold_idx]
            test_idx_set = set(test_idx.tolist())
            for j, (fp, _) in enumerate(eval_items):
                is_oof[fold_idx, j] = pool_index_by_path[fp] in test_idx_set

        # sanity: every eval image should be the held-out test image for exactly 1 of the 5 folds
        n_oof_per_image = is_oof.sum(axis=0)

        ensemble_probs = fold_probs.mean(axis=0)
        ensemble_preds = ensemble_probs.argmax(axis=1)
        ensemble_f1 = f1_score(eval_labels, ensemble_preds, average="macro")
        ensemble_acc = accuracy_score(eval_labels, ensemble_preds)

        indiv_f1, indiv_acc = [], []
        for fold_idx in range(N_SPLITS):
            preds = fold_probs[fold_idx].argmax(axis=1)
            indiv_f1.append(f1_score(eval_labels, preds, average="macro"))
            indiv_acc.append(accuracy_score(eval_labels, preds))

        # unbiased OOF prediction: for each image, use only the 1 fold that held it out as test
        oof_preds = np.zeros(n_eval, dtype=int)
        for j in range(n_eval):
            fold_for_j = np.where(is_oof[:, j])[0]
            f = fold_for_j[0] if len(fold_for_j) else int(np.argmin(n_oof_per_image))  # fallback, shouldn't happen
            oof_preds[j] = fold_probs[f, j].argmax()
        oof_f1 = f1_score(eval_labels, oof_preds, average="macro")
        oof_acc = accuracy_score(eval_labels, oof_preds)

        print(f"\n=== {variant} ensemble evaluation (on original {n_eval}-image test set) ===")
        print(f"  every eval image was out-of-fold for exactly 1/5 models: "
              f"{'yes' if set(n_oof_per_image.tolist()) == {1} else 'NO -- ' + str(np.bincount(n_oof_per_image))}")
        print(f"  individual folds : macro-F1 = {np.mean(indiv_f1):.4f} +/- {np.std(indiv_f1):.4f}  "
              f"(acc = {np.mean(indiv_acc):.4f} +/- {np.std(indiv_acc):.4f})")
        print(f"  soft-vote ensemble (5 models, ~4/5 in-sample per image) : "
              f"macro-F1 = {ensemble_f1:.4f}  acc = {ensemble_acc:.4f}")
        print(f"  unbiased OOF (1 held-out model per image, no ensembling): "
              f"macro-F1 = {oof_f1:.4f}  acc = {oof_acc:.4f}")

        results[variant] = {
            "n_eval_images": n_eval,
            "individual_folds": {"macro_f1_mean": float(np.mean(indiv_f1)), "macro_f1_std": float(np.std(indiv_f1)),
                                  "accuracy_mean": float(np.mean(indiv_acc)), "accuracy_std": float(np.std(indiv_acc)),
                                  "per_fold_macro_f1": indiv_f1, "per_fold_accuracy": indiv_acc},
            "ensemble_soft_vote": {"macro_f1": float(ensemble_f1), "accuracy": float(ensemble_acc),
                                    "caveat": "mix of ~1 out-of-fold + ~4 in-sample models per image"},
            "oof_single_model": {"macro_f1": float(oof_f1), "accuracy": float(oof_acc),
                                  "note": "unbiased: exactly the 1 fold model that held out each image"},
        }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ensemble_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {os.path.join(OUT, 'ensemble_results.json')}")
    return results


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    p_train = sub.add_parser("train")
    p_train.add_argument("variant", choices=VARIANTS)
    p_train.add_argument("fold_idx", type=int)
    sub.add_parser("evaluate")
    args = parser.parse_args()

    if args.cmd == "train":
        train_fold(args.variant, args.fold_idx)
    elif args.cmd == "evaluate":
        evaluate()
    else:
        for variant in VARIANTS:
            for fold_idx in range(N_SPLITS):
                train_fold(variant, fold_idx)
        evaluate()


if __name__ == "__main__":
    main()
