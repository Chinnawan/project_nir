"""Round 2 overnight batch (resumable) + smoke test + self-checks.

Same machinery as run_multiseed.py: seeds 0,1,2 x 5 folds, atomic json write after every fold
(outputs/dl/round2_results.json, never touches multiseed_results.json / kfold_results.json),
resume by skipping finished keys, failing folds logged and skipped, log to
outputs/logs/round2_YYYYMMDD.log with ETA per fold.

Order (so that if the machine dies the important ones are done first):
  1. nir_5cls, nir_ls   2. full_fix, nir_fix, visible_fix   3. nir_ss, nir_ss_cp, nir_ss_plus   4. nir_320
(step 5, Grad-CAM 14x14 on the existing nir_SE_ss checkpoints, is run by run_round2.bat afterwards)

  python -m src.run_round2                 full batch (resumable)
  python -m src.run_round2 --smoke         every variant x 1 seed x 1 fold x 2 epochs + self-checks -> separate json
  python -m src.run_round2 --selfcheck     only the data self-checks (split equivalence, copy-paste isolation, ...)
"""
import argparse
import datetime as dt
import os
import random
import time

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from . import repro  # noqa: F401
from .config import OUTPUT_DIR
from .repro import set_seed
from .round2 import DEFAULTS, VARIANTS, Context, build_paster, make_fold_data, run_round2_fold
from .run_multiseed import keep_awake, load_results, log, record_error, run_key, save_results, setup_logging
from .train_dl_kfold import N_SPLITS, VAL_FRACTION_OF_TRAINVAL

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "round2_results.json")
SMOKE_PATH = os.path.join(OUT, "round2_results_smoke.json")
DEMO_DIR = os.path.join(OUTPUT_DIR, "synthetic_demo")
SEEDS = [0, 1, 2]
GROUPS = [["nir_5cls", "nir_ls"], ["full_fix", "nir_fix", "visible_fix"], ["nir_ss", "nir_ss_cp", "nir_ss_plus"],
          ["nir_320"]]


def meta_block():
    return {"created": dt.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "protocol": "multi-seed protocol (seeded, deterministic, StratifiedKFold n=5 seed 42, inner val split rs=42); "
                        "each run stores plain + TTA metrics and softmax probabilities",
            "variants": {v: {**DEFAULTS, **s} for v, s in VARIANTS.items()}, "seeds": SEEDS}


def run_batch(path, groups, seeds, folds, epochs, batch_size=None):
    data = load_results(path)
    if not data.get("meta"):
        data["meta"] = meta_block()
        save_results(path, data)
    ctx = Context()
    variants = [v for g in groups for v in g]
    total = sum(run_key(v, s, f) not in data["runs"] for v in variants for s in seeds for f in folds)
    log(f"round2: {total} (variant, seed, fold) runs to do ({len(variants)} variants x {len(seeds)} seeds x "
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
                        try:
                            rec = run_round2_fold(ctx, variant, seed, fold, epochs=epochs, batch_size=batch_size)
                        except torch.cuda.OutOfMemoryError:
                            if batch_size is not None:
                                raise
                            log(f"CUDA out of memory in {key} at batch 16 -> retrying once with batch size 8 "
                                f"(recorded in the result as batch_size)")
                            torch.cuda.empty_cache()
                            rec = run_round2_fold(ctx, variant, seed, fold, epochs=epochs, batch_size=8)
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
    log("round2 batch finished")
    if data.get("errors"):
        log(f"WARNING: {len(data['errors'])} failed runs recorded under 'errors' -- re-run to retry them")


# ----------------------------------------------------------------------------- self-checks
def selfcheck(ctx):
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")

    log("self-check 1: inner train/val split identical to the baseline run_fold split")
    for fold in (0, 3):
        tv, _ = ctx.splits[fold]
        items = [ctx.pool_raw[i] for i in tv]
        base_tr, base_va = train_test_split(items, test_size=VAL_FRACTION_OF_TRAINVAL, stratify=ctx.labels[tv],
                                            random_state=42)
        d = make_fold_data(ctx, "nir_ls", fold, {**DEFAULTS, **VARIANTS["nir_ls"]})
        check(f"fold {fold}: train items (order included) == baseline", [f for f, _ in d["train_ds"].items] ==
              [f for f, _ in base_tr])
        check(f"fold {fold}: val items == baseline", [f for f, _ in d["val_ds"].items] == [f for f, _ in base_va])

    log("self-check 2: copy-paste uses only the training part of each fold")
    spec = {**DEFAULTS, **VARIANTS["nir_ss_cp"]}
    for fold in range(N_SPLITS):
        d = make_fold_data(ctx, "nir_ss_cp", fold, spec)
        pool = ctx.pool("ss")
        train_fps = {pool[i][0] for i in d["tr_idx"]}
        other_fps = {pool[i][0] for i in d["va_idx"]} | {pool[i][0] for i in d["te_idx"]}
        donor_fps = {fp for fp, *_ in d["donors"]}
        target_fps = {fp for fp, _ in d["train_ds"].items}
        check(f"fold {fold}: {len(donor_fps)} donors all in train fold, none in val/test",
              donor_fps <= train_fps and not (donor_fps & other_fps))
        check(f"fold {fold}: pasting targets (train dataset items) all in train fold", target_fps <= train_fps
              and not (target_fps & other_fps))
        check(f"fold {fold}: val/test datasets have no paster", d["val_ds"].paster is None and d["test_ds"].paster is None)

    log("self-check 3: paste statistics + determinism (fold 0)")
    d = make_fold_data(ctx, "nir_ss_cp", 0, spec)
    ds = d["train_ds"]
    healthy = [i for i, h in enumerate(ds.healthy_tissue) if h]
    set_seed(0)
    first = [ds[i][0].numpy() for i in healthy[:25]]
    set_seed(0)
    second = [ds[i][0].numpy() for i in healthy[:25]]
    check("same seed -> identical synthetic batch", all(np.array_equal(a, b) for a, b in zip(first, second)))
    set_seed(1)
    paster = d["paster"]
    paster.n_attempts = paster.n_success = 0
    n, relabelled, t0 = 400, 0, time.time()
    for k in range(n):
        _, y = ds[healthy[k % len(healthy)]]
        relabelled += int(y != 0)
    dt_ms = (time.time() - t0) / n * 1000
    rate = relabelled / n
    log(f"  {n} Healthy-tissue draws: {relabelled} turned into synthetic defects ({rate * 100:.1f}%, target ~30%), "
        f"paste attempts {paster.n_attempts}, success {paster.n_success}, {dt_ms:.1f} ms/item")
    check("paste rate close to 30%", 0.24 <= rate <= 0.36)
    check("every attempted paste found a valid placement", paster.n_success == paster.n_attempts,
          f"({paster.n_success}/{paster.n_attempts})")

    log("self-check 4: save 12 synthetic samples (IR band) to " + DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)
    set_seed(7)
    rng = random.Random(7)
    pool = ctx.pool("ss")
    tiles = []
    targets = [i for i in d["tr_idx"] if ctx.tissue[i] == "Healthy"]
    rng.shuffle(targets)
    for i in targets:
        cube = np.load(pool[i][0])
        res = paster.apply(cube, rng=rng)
        if res is None:
            continue
        syn, label = res
        info = paster.last_info
        t, s = cv2.cvtColor(cube[..., 3], cv2.COLOR_GRAY2BGR), cv2.cvtColor(syn[..., 3], cv2.COLOR_GRAY2BGR)
        pair = np.hstack([t, s])
        pair = cv2.copyMakeBorder(pair, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(pair, f"Healthy+{info['tissue']} D{info['defect_no']:04d} ->{['Healthy', 'Internal', 'Surface'][label]}",
                    (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        tiles.append(pair)
        if len(tiles) == 12:
            break
    rows = [np.hstack(tiles[r * 3:(r + 1) * 3]) for r in range(4)]
    cv2.imwrite(os.path.join(DEMO_DIR, "round2_cp_samples.png"), np.vstack(rows))
    check("12 synthetic samples written", len(tiles) == 12)
    log("self-checks " + ("ALL PASSED" if ok else "HAD FAILURES"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    ap.add_argument("--variants", nargs="+", default=None, help="subset of variants (keeps the group order)")
    ap.add_argument("--results", default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="override batch size for every variant")
    args = ap.parse_args()

    keep_awake()
    groups = [[v for v in g if args.variants is None or v in args.variants] for g in GROUPS]
    groups = [g for g in groups if g]
    if args.selfcheck:
        print("log ->", setup_logging("round2_selfcheck"))
        raise SystemExit(0 if selfcheck(Context()) else 1)
    if args.smoke:
        print("log ->", setup_logging("round2_smoke"))
        log("SMOKE TEST: every variant x seed 0 x fold 0 x 2 epochs -> " + SMOKE_PATH)
        ok = selfcheck(Context())
        run_batch(args.results or SMOKE_PATH, groups, [0], [0], epochs=2, batch_size=args.batch_size)
        raise SystemExit(0 if ok else 1)
    print("log ->", setup_logging("round2"))
    log(f"round2 batch start: seeds={args.seeds} folds={args.folds} -> {args.results or RESULTS_PATH}")
    run_batch(args.results or RESULTS_PATH, groups, args.seeds, args.folds, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
