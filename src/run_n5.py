"""N5 (extended): train `visible_ss` (BL+GR) and `full_ss` (all 4 bands) on the geometrically corrected,
band-aligned SingleSpectra images, seeds 3,4,5 x 5 folds -- the same protocol, recipe (legacy first-conv init,
augmentation, no SE, 224 px, 3 classes, TTA) and folds as `nir_ss` in confirm_results.json, so every run pairs with
an nir_ss run of the same (seed, fold).

Why: the raw quadrant images are misregistered by ~15-22 px between bands (N4 + visual check, alignment_visual.py),
so `full` (raw) < `nir` might come from misregistration rather than from the visible bands themselves.

Checkpoints (+ their normalisation stats) go to outputs/dl/confirm_ckpt/{variant}_seed{S}_fold{K}.pt (gitignored),
next to the nir_ss ones, for the demo's model switch (N7). Results -> outputs/dl/n5_results.json (resumable).

  python -m src.run_n5            full run, then checkpoint verification
  python -m src.run_n5 --smoke    1 variant x seed 3 x fold 0 x 2 epochs -> separate files
"""
import argparse
import os
import time

import numpy as np
from torch.utils.data import DataLoader

from . import repro  # noqa: F401
from .config import OUTPUT_DIR
from .infer import load_checkpoint, predict_probs
from .round2 import Context, Round2Dataset, predict_probs as r2_predict, run_round2_fold
from .run_multiseed import keep_awake, load_results, log, record_error, run_key, save_results, setup_logging

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "n5_results.json")
SMOKE_PATH = os.path.join(OUT, "n5_results_smoke.json")
CKPT_DIR = os.path.join(OUT, "confirm_ckpt")
SMOKE_CKPT_DIR = os.path.join(OUT, "confirm_ckpt_smoke")
VARIANTS = {"visible_ss": {"channels": "visible", "source": "ss"}, "full_ss": {"channels": "full", "source": "ss"}}
ORDER = ["full_ss", "visible_ss"]  # full_ss answers the main question -> first
SEEDS = [3, 4, 5]


def ckpt_path(ckpt_dir, variant, seed, fold):
    return os.path.join(ckpt_dir, f"{variant}_seed{seed}_fold{fold}.pt")


def run(path, ckpt_dir, variants, seeds, folds, epochs):
    data = load_results(path)
    if not data.get("meta"):
        data["meta"] = {"purpose": "N5: visible/full on aligned SingleSpectra images, paired with nir_ss (confirm_results.json)",
                        "seeds": seeds, "variants": VARIANTS}
        save_results(path, data)
    ctx = Context()
    total = sum(run_key(v, s, f) not in data["runs"] for v in variants for s in seeds for f in folds)
    log(f"N5: {total} runs to do")
    done, times = 0, []
    for seed in seeds:
        for variant in variants:
            for fold in folds:
                key = run_key(variant, seed, fold)
                if key in data["runs"]:
                    continue
                t0 = time.time()
                log(f"START {key} ({done}/{total} done)")
                try:
                    rec = run_round2_fold(ctx, variant, seed, fold, epochs=epochs, variants=VARIANTS,
                                          ckpt_path=ckpt_path(ckpt_dir, variant, seed, fold))
                    data["runs"][key] = rec
                    save_results(path, data)
                except Exception:
                    record_error(path, data, variant, seed, fold)
                    done += 1
                    continue
                times.append(time.time() - t0)
                done += 1
                eta = np.mean(times[-10:]) * (total - done) / 60
                log(f"END   {key}: F1={rec['test_macro_f1']:.4f} (TTA {rec['tta_test_macro_f1']:.4f}) acc={rec['test_accuracy']:.4f} "
                    f"epochs={rec['epochs_trained']} ({times[-1]:.0f}s) | {done}/{total}, ETA ~{eta:.0f} min")
    log("N5 finished" + (f" with {len(data['errors'])} errors" if data.get("errors") else ""))
    return data


def verify(data, ckpt_dir):
    """Same checks as run_confirm.verify: the checkpoint + saved mean/std reproduce the recorded TTA probabilities
    (batch-16 path < 1e-4) and the demo's one-image path gives the same decisions."""
    ctx = Context()
    pool = ctx.pool("ss")
    ok = True
    for key, rec in sorted(data["runs"].items()):
        model, meta = load_checkpoint(ckpt_path(ckpt_dir, rec["variant"], rec["seed"], rec["fold"]))
        recorded = np.array(rec["tta_test_probs"])
        items = [pool[i] for i in rec["test_idx"]]
        ds = Round2Dataset(items, [0] * len(items), meta["channels"], False, np.array(meta["mean"]), np.array(meta["std"]))
        _, p16 = r2_predict(model, DataLoader(ds, batch_size=16, shuffle=False))
        d16 = float(np.abs(p16 - recorded).max())
        p1 = np.array([predict_probs(model, meta, np.load(fp), tta=True) for fp, _ in items])
        d1 = float(np.abs(p1 - recorded).max())
        flip = p1.argmax(1) != recorded.argmax(1)
        srt = np.sort(recorded, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        # GPU batch-1 vs batch-16 rounding is up to ~1e-2 (same 2e-2 limit as run_confirm.verify): a flipped decision
        # is only a failure when the recorded top-2 margin is larger than that noise; flips on exact ties are listed
        real_flips, tie_flips = int((flip & (margin >= 2e-2)).sum()), int((flip & (margin < 2e-2)).sum())
        good = meta["fold"] == rec["fold"] and meta["seed"] == rec["seed"] and d16 < 1e-4 and d1 < 2e-2 and real_flips == 0
        ok &= good
        log(f"  [{'PASS' if good else 'FAIL'}] {key}: batch-16 max |diff| {d16:.1e}, one-image path max |diff| {d1:.1e}, "
            f"{real_flips} decisions differ" + (f" (+{tie_flips} on a recorded tie, margin < 0.02)" if tie_flips else ""))
    log("verify " + ("ALL PASSED" if ok else "HAD FAILURES"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    if args.verify_only:
        print("log ->", setup_logging("n5_verify"))
        raise SystemExit(0 if verify(load_results(RESULTS_PATH), CKPT_DIR) else 1)
    keep_awake()
    if args.smoke:
        print("log ->", setup_logging("n5_smoke"))
        data = run(SMOKE_PATH, SMOKE_CKPT_DIR, ["full_ss"], [3], [0], epochs=2)
        raise SystemExit(0 if verify(data, SMOKE_CKPT_DIR) else 1)
    print("log ->", setup_logging("n5"))
    data = run(RESULTS_PATH, CKPT_DIR, ORDER, SEEDS, list(range(5)), epochs=100)
    raise SystemExit(0 if verify(data, CKPT_DIR) else 1)


if __name__ == "__main__":
    main()
