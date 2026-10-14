"""Round 2 variants: one change at a time from the baseline `nir` (RE+IR, augmentation, no SE,
224px, raw RawA images, 3 classes), trained under the multi-seed protocol (seeded + deterministic,
StratifiedKFold n=5 seed 42, inner train/val split random_state=42 identical to train_dl_kfold).

Same recipe as run_training: AdamW lr 1e-4 wd 1e-4, ReduceLROnPlateau(0.5, patience 3), early
stopping patience 10 on val loss, best-val-loss weights, batch 16, class-balanced
WeightedRandomSampler (weights from the 3-class counts in every variant). Validation / early
stopping always uses the plain (unsmoothed) cross-entropy so the stopping rule is identical across
variants; only the training loss differs for label smoothing.

Every run evaluates the fold's test images twice (plain, and TTA = mean softmax over identity +
h-flip + v-flip + rot90/180/270) and stores the softmax probabilities.

  nir_5cls     5 training classes (Healthy / Internal / Surface / Stem / Calix); test prediction merges
               Healthy+Stem+Calix probabilities back into Healthy
  nir_ls       label smoothing 0.1 (training loss only)
  full_fix / nir_fix / visible_fix   first-conv init = timm recipe (bands get different RGB filters,
               rescaled by 3/C) instead of the repeated mean filter
  nir_ss       same as nir on the SingleSpectra images (baseline for nir_ss_cp)
  nir_ss_cp    nir_ss + Defect Copy-Paste (copy_paste.py) on Healthy-tissue training images, p=0.3
  nir_ss_plus  train on raw AND SingleSpectra images of the same training apples, test on raw
  nir_320      320x320 input (Processed_Dataset_320)
"""
import copy
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .config import CLASSES
from .copy_paste import DefectCopyPaster
from .dataset import ApplesDataset, compute_stats_for_files
from .models import build_model
from .preprocess_ss import MAPPING_PATH
from .repro import make_generator, seed_worker, set_seed
from .train_dl import DEVICE, _run_epoch
from .train_dl_kfold import N_SPLITS, VAL_FRACTION_OF_TRAINVAL, build_pool, get_fold_splits

SEED_SPLIT = 42
PASTE_PROB = 0.3
CLASSES5 = CLASSES + ["Stem", "Calix"]
DEFAULTS = {"channels": "nir", "source": "raw", "n_classes": 3, "label_smoothing": 0.0, "init": "legacy",
            "copy_paste": False, "extra_train_source": None, "batch_size": 16}
VARIANTS = {
    "nir_5cls": {"n_classes": 5},
    "nir_ls": {"label_smoothing": 0.1},
    "full_fix": {"channels": "full", "init": "timm"},
    "nir_fix": {"init": "timm"},
    "visible_fix": {"channels": "visible", "init": "timm"},
    "nir_ss": {"source": "ss"},
    "nir_ss_cp": {"source": "ss", "copy_paste": True},
    "nir_ss_plus": {"extra_train_source": "ss"},
    "nir_320": {"source": "320", "batch_size": 16},
}
# comparison baselines (multiseed_results.json uses the same seeds/folds/protocol) -- for the report
BASELINES = {"nir_5cls": "nir", "nir_ls": "nir", "full_fix": "full", "nir_fix": "nir", "visible_fix": "visible",
             "nir_ss": "nir", "nir_ss_cp": "nir_ss", "nir_ss_plus": "nir", "nir_320": "nir"}


class Context:
    """Pools, labels, tissue info and fold splits shared by all variants (built once)."""

    def __init__(self):
        self._pools = {}
        self.pool_raw, self.labels = build_pool("raw")
        self.splits = get_fold_splits(self.pool_raw, self.labels)
        with open(MAPPING_PATH) as f:
            mapping = json.load(f)
        rawnos = [int(os.path.basename(fp).split("RawA")[1].split(".")[0]) for fp, _ in self.pool_raw]
        self.tissue = [mapping[str(r)]["tissue"] for r in rawnos]
        self.defect_no = [mapping[str(r)]["defect_no"] for r in rawnos]
        self.labels5 = np.array([3 if t == "Stem" else 4 if t == "Calix" else l
                                 for t, l in zip(self.tissue, self.labels)])
        self._pools["raw"] = (self.pool_raw, self.labels)

    def pool(self, source):
        if source not in self._pools:
            self._pools[source] = build_pool(source)
        return self._pools[source][0]


class Round2Dataset(ApplesDataset):
    """ApplesDataset with (a) its own training labels (5-class for nir_5cls) and (b) optional
    on-the-fly Defect Copy-Paste on Healthy-tissue items (training set only)."""

    def __init__(self, items, labels_train, channel_group, augment, mean, std, healthy_tissue=None,
                 paster=None, paste_prob=PASTE_PROB):
        super().__init__("train" if augment else "eval", channel_group, augment=augment, mean=mean, std=std,
                         items=items)
        self.labels_train = [int(l) for l in labels_train]
        self.healthy_tissue = healthy_tissue
        self.paster = paster
        self.paste_prob = paste_prob

    def __getitem__(self, idx):
        fp, _ = self.items[idx]
        label = self.labels_train[idx]
        cube = np.load(fp)
        if self.paster is not None and self.healthy_tissue[idx] and random.random() < self.paste_prob:
            res = self.paster.apply(cube)
            if res is not None:
                cube, label = res
        arr = np.transpose(cube.astype(np.float32) / 255.0, (2, 0, 1))[self.channels]
        if self.augment:
            arr = self._augment(arr)
        arr = (arr - self.mean) / self.std
        return torch.from_numpy(arr.copy()), label


def predict_probs(model, loader):
    """Returns (plain, tta) softmax arrays (N, n_classes). TTA = mean over identity, h-flip, v-flip,
    rot90, rot180, rot270 of the (already normalised) input tensor."""
    model.eval()
    plain, tta = [], []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            views = [xb, xb.flip(3), xb.flip(2), torch.rot90(xb, 1, (2, 3)), torch.rot90(xb, 2, (2, 3)),
                     torch.rot90(xb, 3, (2, 3))]
            ps = torch.stack([torch.softmax(model(v), dim=1) for v in views])
            plain.append(ps[0].cpu().numpy())
            tta.append(ps.mean(0).cpu().numpy())
    return np.concatenate(plain).astype(np.float64), np.concatenate(tta).astype(np.float64)


def merge_to_3(p):
    """5-class probabilities (Healthy, Internal, Surface, Stem, Calix) -> 3-class."""
    return np.stack([p[:, 0] + p[:, 3] + p[:, 4], p[:, 1], p[:, 2]], axis=1)


def build_paster(ctx, pool, train_idx):
    donors = [(pool[i][0], ctx.defect_no[i], int(ctx.labels[i]), ctx.tissue[i])
              for i in train_idx if ctx.defect_no[i] is not None]
    return DefectCopyPaster(donors), donors


def make_fold_data(ctx, variant, fold, spec):
    """Returns the datasets + sampler weights for one (variant, fold). Pure data logic (no training)."""
    tv, te = ctx.splits[fold]
    tr_rel, va_rel = train_test_split(np.arange(len(tv)), test_size=VAL_FRACTION_OF_TRAINVAL,
                                      stratify=ctx.labels[tv], random_state=SEED_SPLIT)
    tr_idx, va_idx = tv[tr_rel], tv[va_rel]
    pool = ctx.pool(spec["source"])
    lab = ctx.labels5 if spec["n_classes"] == 5 else ctx.labels

    train_idx = list(tr_idx)
    train_items = [pool[i] for i in train_idx]
    if spec["extra_train_source"]:
        extra = ctx.pool(spec["extra_train_source"])
        train_items += [extra[i] for i in tr_idx]
        train_idx = train_idx + list(tr_idx)
    train_lab = [lab[i] for i in train_idx]
    mean, std = compute_stats_for_files([pool[i] for i in tr_idx])  # stats from the training apples, test-source images

    paster = donors = None
    if spec["copy_paste"]:
        paster, donors = build_paster(ctx, pool, tr_idx)
    train_ds = Round2Dataset(train_items, train_lab, spec["channels"], True, mean, std,
                             healthy_tissue=[ctx.tissue[i] == "Healthy" for i in train_idx], paster=paster)
    val_ds = Round2Dataset([pool[i] for i in va_idx], [lab[i] for i in va_idx], spec["channels"], False, mean, std)
    test_ds = Round2Dataset([pool[i] for i in te], [ctx.labels[i] for i in te], spec["channels"], False, mean, std)

    w3 = 1.0 / np.bincount(ctx.labels[tr_idx], minlength=3)  # class-balanced on the 3 real classes, as the baseline
    weights = [float(w3[ctx.labels[i]]) for i in train_idx]
    return {"train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds, "weights": weights, "tr_idx": tr_idx,
            "mean": mean, "std": std,
            "va_idx": va_idx, "te_idx": te, "n_epoch_samples": len(tr_idx) if spec["extra_train_source"] else len(train_items),
            "paster": paster, "donors": donors}


def per_class_and_cm(y_true, y_pred):
    pc = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    return pc, confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()


def run_round2_fold(ctx, variant, seed, fold, epochs=100, batch_size=None, variants=None, ckpt_path=None):
    """variants: alternative {name: spec-overrides} table (confirmatory run); ckpt_path: save the best-val-loss
    weights + the mean/std used to normalise this fold's images (needed by demo_app)."""
    spec = {**DEFAULTS, **(variants or VARIANTS)[variant]}
    bs = batch_size or spec["batch_size"]
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    d = make_fold_data(ctx, variant, fold, spec)
    gen = make_generator(seed)
    sampler = WeightedRandomSampler(d["weights"], num_samples=d["n_epoch_samples"], replacement=True, generator=gen)
    train_loader = DataLoader(d["train_ds"], batch_size=bs, sampler=sampler, num_workers=0, generator=gen,
                              worker_init_fn=seed_worker)
    val_loader = DataLoader(d["val_ds"], batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(d["test_ds"], batch_size=bs, shuffle=False, num_workers=0)

    model = build_model(spec["channels"], num_classes=spec["n_classes"], pretrained=True, use_se=False,
                        init=spec["init"]).to(DEVICE)
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=spec["label_smoothing"])
    val_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val, best_state, no_improve, n_ep = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        tr_loss, _, tr_f1, _, _ = _run_epoch(model, train_loader, train_loss_fn, optimizer)
        val_loss, val_acc, val_f1, _, _ = _run_epoch(model, val_loader, val_loss_fn, optimizer=None)
        scheduler.step(val_loss)
        n_ep = epoch
        print(f"  epoch {epoch:3d} | train loss {tr_loss:.4f} f1 {tr_f1:.3f} | val loss {val_loss:.4f} "
              f"acc {val_acc:.3f} f1 {val_f1:.3f}")
        if val_loss < best_val - 1e-4:
            best_val, best_state, no_improve = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= 10:
                print("  early stopping at epoch %d (no val-loss improvement for 10 epochs)" % epoch)
                break
    model.load_state_dict(best_state)
    if ckpt_path:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mean": np.asarray(d["mean"]).tolist(),
                    "std": np.asarray(d["std"]).tolist(), "variant": variant, "seed": seed, "fold": fold,
                    "channels": spec["channels"], "source": spec["source"], "n_classes": spec["n_classes"],
                    "init": spec["init"], "classes": CLASSES}, ckpt_path)

    p_plain, p_tta = predict_probs(model, test_loader)
    raw5 = (p_plain, p_tta) if spec["n_classes"] == 5 else None
    if spec["n_classes"] == 5:
        p_plain, p_tta = merge_to_3(p_plain), merge_to_3(p_tta)
    y = ctx.labels[d["te_idx"]]
    rec = {"variant": variant, "seed": seed, "fold": fold, "kind": "dl", "test_idx": d["te_idx"].tolist(),
           "best_val_loss": float(best_val), "epochs_trained": n_ep, "batch_size": bs, "source": spec["source"],
           "elapsed_sec": round(time.time() - t0, 1),
           "max_vram_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1) if torch.cuda.is_available() else None,
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for tag, p in (("", p_plain), ("tta_", p_tta)):
        pred = p.argmax(1)
        pc, cm = per_class_and_cm(y, pred)
        rec.update({f"{tag}test_macro_f1": float(f1_score(y, pred, average="macro")),
                    f"{tag}test_accuracy": float(accuracy_score(y, pred)), f"{tag}per_class_f1": pc,
                    f"{tag}confusion_matrix": cm, f"{tag}test_predictions": pred.tolist(),
                    f"{tag}test_probs": np.round(p, 5).tolist()})
    if raw5:
        rec["test_probs5"], rec["tta_test_probs5"] = np.round(raw5[0], 5).tolist(), np.round(raw5[1], 5).tolist()
    if d["paster"] is not None:
        rec["copy_paste"] = {"attempts": d["paster"].n_attempts, "pasted": d["paster"].n_success,
                             "n_donors": len(d["donors"])}
    print(f"  [{variant}_s{seed}_f{fold}] TEST macro-F1={rec['test_macro_f1']:.4f} acc={rec['test_accuracy']:.4f} | "
          f"TTA macro-F1={rec['tta_test_macro_f1']:.4f} acc={rec['tta_test_accuracy']:.4f} | "
          f"VRAM peak {rec['max_vram_mb']} MB")
    del model, optimizer, train_loader, val_loader, test_loader, d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec
