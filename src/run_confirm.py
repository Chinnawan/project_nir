"""Confirmatory run with NEW seeds 3, 4, 5 (seeds 0-2 were used to choose the formula).

Locked formula (decided before this run, not to be changed after seeing results):
    nir_ss  = RE+IR bands of the flat-field-corrected SingleSpectra images, augmentation, no SE,
              224px, 3 classes  +  TTA (6 views)  +  soft vote of the 3 seeds inside a fold.
    nir     = baseline (raw RawA images), same seeds/folds -> paired comparison.
    nir_ss_ls (optional, EXPLORATORY): nir_ss + label smoothing 0.1; only an observation, not a result.

Same machinery/protocol as round 2 (seeded + deterministic, StratifiedKFold n=5 seed 42, same inner
val split). Writes outputs/dl/confirm_results.json after every fold (resumable, never touches the
older result files), saves the 15 nir_ss checkpoints (+ their normalisation stats) to
outputs/dl/confirm_ckpt/ and the fold of every image to outputs/dl/fold_assignment.json (demo app).

  python -m src.run_confirm            full run (resumable)
  python -m src.run_confirm --smoke    3 variants x seed 3 x fold 0 x 2 epochs + checks -> separate files
"""
import argparse
import datetime as dt
import json
import os
import time

import numpy as np
import torch

from . import repro  # noqa: F401
from .config import CLASSES, OUTPUT_DIR
from .infer import load_checkpoint, predict_probs
from .round2 import DEFAULTS, Context, run_round2_fold
from .run_multiseed import keep_awake, load_results, log, record_error, run_key, save_results, setup_logging
from .train_dl_kfold import N_SPLITS

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "confirm_results.json")
SMOKE_PATH = os.path.join(OUT, "confirm_results_smoke.json")
CKPT_DIR = os.path.join(OUT, "confirm_ckpt")
SMOKE_CKPT_DIR = os.path.join(OUT, "confirm_ckpt_smoke")
FOLD_ASSIGN_PATH = os.path.join(OUT, "fold_assignment.json")
SEEDS = [3, 4, 5]
VARIANTS = {"nir": {}, "nir_ss": {"source": "ss"}, "nir_ss_ls": {"source": "ss", "label_smoothing": 0.1}}
GROUPS = [["nir_ss", "nir"], ["nir_ss_ls"]]  # core first, exploratory last
CKPT_VARIANT = "nir_ss"


def write_fold_assignment(ctx, path=FOLD_ASSIGN_PATH):
    fold_of = {}
    for fold, (_, te) in enumerate(ctx.splits):
        for i in te:
            name = os.path.basename(ctx.pool_raw[i][0]).replace(".npy", "")
            fold_of[name] = {"fold": fold, "tissue": ctx.tissue[i], "label": CLASSES[ctx.labels[i]],
                             "defect_no": ctx.defect_no[i]}
    assert len(fold_of) == 819 and all(0 <= v["fold"] < N_SPLITS for v in fold_of.values())
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_splits": N_SPLITS, "fold_split": "StratifiedKFold(shuffle=True, random_state=42), "
                   "each image is in the test part of exactly one fold", "images": fold_of}, f)
    return fold_of


def ckpt_path_for(ckpt_dir, variant, seed, fold):
    return os.path.join(ckpt_dir, f"{variant}_seed{seed}_fold{fold}.pt") if variant == CKPT_VARIANT else None


def run_batch(path, ckpt_dir, groups, seeds, folds, epochs):
    assert min(seeds) >= 3, "confirmatory run must not use seeds 0-2 (they were used to choose the formula)"
    data = load_results(path)
    if not data.get("meta"):
        data["meta"] = {"created": dt.datetime.now().isoformat(timespec="seconds"), "seeds": seeds,
                        "locked_formula": "nir_ss + TTA(6 views) + 3-seed soft vote inside a fold; baseline nir (raw)",
                        "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                        "variants": {v: {**DEFAULTS, **s} for v, s in VARIANTS.items()}}
        save_results(path, data)
    ctx = Context()
    write_fold_assignment(ctx)
    variants = [v for g in groups for v in g]
    total = sum(run_key(v, s, f) not in data["runs"] for v in variants for s in seeds for f in folds)
    log(f"confirm: {total} (variant, seed, fold) runs to do ({len(variants)} variants x {len(seeds)} seeds x "
        f"{len(folds)} folds, minus already done)")
    done, times = 0, []
    for group in groups:
        for seed in seeds:
            for variant in group:
                for fold in folds:
                    key = run_key(variant, seed, fold)
                    if key in data["runs"]:
                        continue
                    t0 = time.time()
                    log(f"START {key}   ({done}/{total} done)")
                    try:
                        rec = run_round2_fold(ctx, variant, seed, fold, epochs=epochs, variants=VARIANTS,
                                              ckpt_path=ckpt_path_for(ckpt_dir, variant, seed, fold))
                        data["runs"][key] = rec
                        save_results(path, data)
                    except Exception:
                        record_error(path, data, variant, seed, fold)
                        done += 1
                        continue
                    dur = time.time() - t0
                    times.append(dur)
                    done += 1
                    eta = np.mean(times[-15:]) * (total - done) / 60
                    log(f"END   {key}: F1={rec['test_macro_f1']:.4f} (TTA {rec['tta_test_macro_f1']:.4f}) "
                        f"acc={rec['test_accuracy']:.4f} epochs={rec['epochs_trained']} ({dur:.0f}s) | {done}/{total} done, "
                        f"ETA ~{eta / 60:.1f} h ({eta:.0f} min)")
    log("confirm batch finished")
    if data.get("errors"):
        log(f"WARNING: {len(data['errors'])} failed runs recorded under 'errors' -- re-run to retry them")
    return data


def verify(ctx, data, ckpt_dir):
    """The demo app relies on: (1) checkpoint + saved mean/std reproduce the recorded TTA probabilities,
    (2) fold_assignment.json agrees with the fold the run actually tested on."""
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    log("verify: fold_assignment.json vs the recorded test fold")
    fa = json.load(open(FOLD_ASSIGN_PATH, encoding="utf-8"))["images"]
    for key, rec in data["runs"].items():
        if rec["variant"] != CKPT_VARIANT:
            continue
        names = {os.path.basename(ctx.pool_raw[i][0]).replace(".npy", "") for i in rec["test_idx"]}
        check(f"{key}: {len(names)} test images are exactly the images assigned to fold {rec['fold']}",
              names == {n for n, v in fa.items() if v["fold"] == rec["fold"]})
    log("verify: checkpoint reproduces the recorded TTA probabilities")
    from torch.utils.data import DataLoader

    from .round2 import Round2Dataset, predict_probs as r2_predict
    pool = ctx.pool("ss")
    for key, rec in data["runs"].items():
        if rec["variant"] != CKPT_VARIANT:
            continue
        path = ckpt_path_for(ckpt_dir, rec["variant"], rec["seed"], rec["fold"])
        model, meta = load_checkpoint(path)
        recorded = np.array(rec["tta_test_probs"])
        items = [pool[i] for i in rec["test_idx"]]
        # (a) training-time evaluation path (batches of 16): checkpoint + saved mean/std must give the recorded numbers
        ds = Round2Dataset(items, [0] * len(items), meta["channels"], False, np.array(meta["mean"]), np.array(meta["std"]))
        _, p16 = r2_predict(model, DataLoader(ds, batch_size=16, shuffle=False))
        d16 = float(np.abs(p16 - recorded).max())
        # (b) demo path (one image per forward pass): different cuDNN kernels -> tiny float differences, same decisions
        p1 = np.array([predict_probs(model, meta, np.load(fp), tta=True) for fp, _ in items])
        d1 = float(np.abs(p1 - recorded).max())
        mism = int((p1.argmax(1) != recorded.argmax(1)).sum())
        check(f"{key}: ckpt meta OK; batch-16 path max |prob diff| = {d16:.1e} (limit 1e-4)",
              meta["fold"] == rec["fold"] and meta["seed"] == rec["seed"] and len(meta["mean"]) == 4 and d16 < 1e-4)
        check(f"{key}: single-image (demo) path max |prob diff| = {d1:.1e} (limit 1e-2), {mism}/{len(items)} decisions differ",
              d1 < 1e-2 and mism == 0)
    log("verify " + ("ALL PASSED" if ok else "HAD FAILURES"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    args = ap.parse_args()
    keep_awake()
    if args.smoke:
        print("log ->", setup_logging("confirm_smoke"))
        log("SMOKE TEST: 3 variants x seed 3 x fold 0 x 2 epochs -> " + SMOKE_PATH)
        data = run_batch(SMOKE_PATH, SMOKE_CKPT_DIR, GROUPS, [3], [0], epochs=2)
        raise SystemExit(0 if verify(Context(), data, SMOKE_CKPT_DIR) else 1)
    print("log ->", setup_logging("confirm"))
    log(f"confirmatory run start: seeds={args.seeds} folds={args.folds} -> {RESULTS_PATH}")
    data = run_batch(RESULTS_PATH, CKPT_DIR, GROUPS, args.seeds, args.folds, args.epochs)
    verify(Context(), data, CKPT_DIR)


if __name__ == "__main__":
    main()
