"""Overnight multi-seed run: every variant x seeds {0,1,2} x 5 folds under ONE protocol
(seeded + deterministic kernels, fixed StratifiedKFold n=5 seed 42, raw images).

 - classical ML (SVM / RandomForest / LightGBM) on the same 5 folds first: fast and
   deterministic, so it is run once (stored as seed 0 and reused for every seed in the report)
 - then the DL variants in priority order, so that if the machine dies the important ones
   are finished first. Inside a group: seed -> variant -> fold.
 - results go to outputs/dl/multiseed_results.json right after EVERY fold (atomic write);
   kfold_results.json is never touched. Re-running resumes: finished (variant, seed, fold)
   keys are skipped. A failing fold is logged and skipped, the batch keeps going.
 - everything printed is also appended to outputs/logs/overnight_YYYYMMDD.log

Usage (normally via run_overnight.bat):
  python -m src.run_multiseed                     full batch (resumable)
  python -m src.run_multiseed --smoke             2 variants x 1 seed x 1 fold x 2 epochs -> separate json
  python -m src.run_multiseed --repro-test        run nir (and nir_SE) fold 0 seed 0 twice, compare
"""
import argparse
import ctypes
import datetime as dt
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from . import repro  # noqa: F401  (sets CUBLAS_WORKSPACE_CONFIG before any CUDA work)
from .config import CLASSES, OUTPUT_DIR, PROJECT_ROOT
from .features import build_feature_dataset
from .train_classical import build_models
from .train_dl_kfold import CKPT_DIR, N_SPLITS, VARIANT_SPECS, build_pool, get_fold_splits, run_fold

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "multiseed_results.json")
SMOKE_RESULTS_PATH = os.path.join(OUT, "multiseed_results_smoke.json")
REPRO_PATH = os.path.join(OUT, "repro_test.json")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
SEEDS = [0, 1, 2]
CLASSICAL = ["SVM", "RandomForest", "LightGBM"]
DL_GROUPS = [
    ["nir", "nir_SE", "nir_no_aug"],
    ["re_only", "ir_only", "bl_only", "gr_only"],
    ["visible", "full", "full_SE", "gr_nir"],
]
CKPT_VARIANT_SEED = ("nir_SE", 0)  # only checkpoint kept (for demo / Grad-CAM)


# ----------------------------------------------------------------------------- logging
class Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def isatty(self):
        return False


def setup_logging(name):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{name}_{dt.date.today():%Y%m%d}.log")
    fh = open(path, "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, fh), Tee(sys.__stderr__, fh)
    return path


def log(msg):
    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def keep_awake():
    """Ask Windows not to idle-sleep while this process runs (cleared automatically on exit)."""
    if os.name == "nt":
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


# ----------------------------------------------------------------------------- results file
def load_results(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"meta": {}, "runs": {}, "errors": []}


def save_results(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def run_key(variant, seed, fold):
    return f"{variant}|seed{seed}|fold{fold}"


def meta_block():
    return {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "protocol": {"n_splits": N_SPLITS, "fold_split": "StratifiedKFold(shuffle=True, random_state=42) on the raw pool",
                      "inner_val_split": "train_test_split(test_size=0.176, stratify, random_state=42) -- fixed, seed-independent",
                      "deterministic": True, "seeds": SEEDS, "classical": "seed-independent, stored as seed0"},
    }


# ----------------------------------------------------------------------------- runners
def per_class_and_cm(y_true, y_pred):
    pc = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return pc, cm


def load_classical_features(pool, labels):
    Xs, paths, ys = [], [], []
    for split in ("train", "val", "test"):
        X, y, p = build_feature_dataset(split)
        Xs.append(X)
        ys.append(y)
        paths += [str(q) for q in p]
    X, y = np.concatenate(Xs), np.concatenate(ys)
    assert [os.path.normcase(a) for a in paths] == [os.path.normcase(fp) for fp, _ in pool], \
        "cached classical features are not in build_pool() order"
    assert (y == labels).all()
    return X


def run_classical(path, data, pool, labels, splits, folds):
    todo = [(m, f) for f in folds for m in CLASSICAL if run_key(m, 0, f) not in data["runs"]]
    if not todo:
        log("classical: nothing to do (all folds present)")
        return
    X_all = load_classical_features(pool, labels)
    for fold in folds:
        tv, te = splits[fold]
        if all(run_key(m, 0, fold) in data["runs"] for m in CLASSICAL):
            continue
        t0 = time.time()
        scaler = StandardScaler().fit(X_all[tv])
        Xtr, Xte = scaler.transform(X_all[tv]), scaler.transform(X_all[te])
        for name, model in build_models().items():
            key = run_key(name, 0, fold)
            if key in data["runs"]:
                continue
            try:
                if name == "LightGBM":
                    model.set_params(deterministic=True, force_row_wise=True)
                t1 = time.time()
                model.fit(Xtr, labels[tv])
                pred = model.predict(Xte)
                pc, cm = per_class_and_cm(labels[te], pred)
                data["runs"][key] = {
                    "variant": name, "seed": 0, "fold": fold, "kind": "classical",
                    "test_macro_f1": float(f1_score(labels[te], pred, average="macro")),
                    "test_accuracy": float(accuracy_score(labels[te], pred)),
                    "per_class_f1": pc, "confusion_matrix": cm,
                    "test_idx": te.tolist(), "test_predictions": pred.tolist(),
                    "elapsed_sec": round(time.time() - t1, 1),
                    "finished_at": dt.datetime.now().isoformat(timespec="seconds"),
                }
                save_results(path, data)
                log(f"classical {name} fold {fold}: F1={data['runs'][key]['test_macro_f1']:.4f} "
                    f"acc={data['runs'][key]['test_accuracy']:.4f} ({data['runs'][key]['elapsed_sec']}s)")
            except Exception:
                record_error(path, data, name, 0, fold)
        log(f"classical fold {fold} done in {time.time() - t0:.0f}s")


def record_error(path, data, variant, seed, fold):
    tb = traceback.format_exc()
    log(f"ERROR in {variant} seed {seed} fold {fold} -- skipping, batch continues\n{tb}")
    data.setdefault("errors", []).append({"variant": variant, "seed": seed, "fold": fold, "traceback": tb,
                                           "at": dt.datetime.now().isoformat(timespec="seconds")})
    save_results(path, data)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_dl_fold(path, data, variant, seed, fold, pool, labels, splits, epochs, allow_ckpt=True):
    channel_group, use_se, augment, _source, _ = VARIANT_SPECS[variant]
    tv, te = splits[fold]
    ckpt = None
    if allow_ckpt and (variant, seed) == CKPT_VARIANT_SEED:
        ckpt = os.path.join(CKPT_DIR, f"{variant}_seed{seed}_fold{fold}.pt")
    t0 = time.time()
    rec = run_fold(fold, [pool[i] for i in tv], labels[tv], [pool[i] for i in te], channel_group, use_se,
                   augment_train=augment, name=variant, ckpt_path=ckpt, seed=seed, epochs=epochs)
    rec.pop("test_labels", None)
    rec.update({"variant": variant, "seed": seed, "fold": fold, "kind": "dl", "test_idx": te.tolist(),
                "elapsed_sec": round(time.time() - t0, 1), "epochs_cap": epochs,
                "finished_at": dt.datetime.now().isoformat(timespec="seconds")})
    data["runs"][run_key(variant, seed, fold)] = rec
    save_results(path, data)
    return rec


def count_todo(data, variants, seeds, folds):
    return sum(run_key(v, s, f) not in data["runs"] for v in variants for s in seeds for f in folds)


def run_batch(path, variants_groups, seeds, folds, epochs, do_classical=True, allow_ckpt=True):
    data = load_results(path)
    if not data.get("meta"):
        data["meta"] = meta_block()
        save_results(path, data)

    pool, labels = build_pool("raw")
    splits = get_fold_splits(pool, labels)

    if do_classical:
        run_classical(path, data, pool, labels, splits, folds)

    all_variants = [v for g in variants_groups for v in g]
    total = count_todo(data, all_variants, seeds, folds)
    log(f"DL: {total} (variant, seed, fold) runs to do "
        f"({len(all_variants)} variants x {len(seeds)} seeds x {len(folds)} folds, minus already done)")
    done_times, done = [], 0
    for group in variants_groups:
        for seed in seeds:
            for variant in group:
                for fold in folds:
                    key = run_key(variant, seed, fold)
                    if key in data["runs"]:
                        continue
                    t0 = time.time()
                    log(f"START {key}   ({done}/{total} done)")
                    try:
                        rec = run_dl_fold(path, data, variant, seed, fold, pool, labels, splits, epochs, allow_ckpt)
                    except Exception:
                        record_error(path, data, variant, seed, fold)
                        done += 1
                        continue
                    dur = time.time() - t0
                    done_times.append(dur)
                    done += 1
                    recent = np.mean(done_times[-15:])
                    eta_min = recent * (total - done) / 60
                    log(f"END   {key}: F1={rec['test_macro_f1']:.4f} acc={rec['test_accuracy']:.4f} "
                        f"epochs={rec['epochs_trained']} ({dur:.0f}s) | {done}/{total} done, "
                        f"ETA ~{eta_min / 60:.1f} h ({eta_min:.0f} min)")
    log("batch finished")
    n_err = len(data.get("errors", []))
    if n_err:
        log(f"WARNING: {n_err} failed runs recorded in the results json under 'errors' -- re-run to retry them")


# ----------------------------------------------------------------------------- repro test
def repro_one(variant, epochs, seed=0, fold=0):
    pool, labels = build_pool("raw")
    tv, te = get_fold_splits(pool, labels)[fold]
    channel_group, use_se, augment, _s, _c = VARIANT_SPECS[variant]
    rec = run_fold(fold, [pool[i] for i in tv], labels[tv], [pool[i] for i in te], channel_group, use_se,
                   augment_train=augment, name=variant, seed=seed, epochs=epochs, with_history=True)
    print("REPRO_JSON " + json.dumps({
        "variant": variant, "seed": seed, "fold": fold, "test_macro_f1": rec["test_macro_f1"],
        "test_accuracy": rec["test_accuracy"], "best_val_loss": rec["best_val_loss"],
        "history": rec["history"], "test_predictions": rec["test_predictions"]}))


def repro_test(epochs=5, variants=("nir", "nir_SE")):
    env = dict(os.environ, PYTHONHASHSEED="0", CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTHONUNBUFFERED="1")
    report = {"epochs": epochs, "seed": 0, "fold": 0, "variants": {}}
    for variant in variants:
        runs, warns = [], set()
        for i in (1, 2):
            log(f"repro test: {variant} fold 0 seed 0, run {i}/2 ({epochs} epochs) in a fresh process")
            t0 = time.time()
            proc = subprocess.run([sys.executable, "-u", "-m", "src.run_multiseed", "--repro-one", variant,
                                   "--epochs", str(epochs)], capture_output=True, text=True, env=env, cwd=PROJECT_ROOT)
            line = [l for l in proc.stdout.splitlines() if l.startswith("REPRO_JSON ")]
            if not line:
                raise RuntimeError(f"repro run failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
            runs.append(json.loads(line[0][len("REPRO_JSON "):]))
            for l in proc.stderr.splitlines():
                if "Warning" in l or "deterministic" in l.lower():
                    warns.add(l.strip()[:300])
            log(f"  run {i}: test F1={runs[-1]['test_macro_f1']:.6f}, best val loss={runs[-1]['best_val_loss']:.6f} "
                f"({time.time() - t0:.0f}s)")
        a, b = runs
        keys = ("train_loss", "val_loss", "val_f1", "val_acc")
        diffs = {k: float(np.max(np.abs(np.array([h[k] for h in a["history"]]) -
                                         np.array([h[k] for h in b["history"]])))) for k in keys}
        identical = (a["history"] == b["history"] and a["test_macro_f1"] == b["test_macro_f1"]
                     and a["test_predictions"] == b["test_predictions"])
        report["variants"][variant] = {
            "bit_identical": identical, "max_abs_diff_per_metric": diffs,
            "test_macro_f1": [a["test_macro_f1"], b["test_macro_f1"]],
            "best_val_loss": [a["best_val_loss"], b["best_val_loss"]],
            "n_prediction_mismatches": int(sum(x != y for x, y in zip(a["test_predictions"], b["test_predictions"]))),
            "val_loss_per_epoch_run1": [h["val_loss"] for h in a["history"]],
            "val_loss_per_epoch_run2": [h["val_loss"] for h in b["history"]],
            "warnings_from_deterministic_mode": sorted(warns),
        }
        log(f"repro {variant}: bit-identical={identical}, max |diff| {diffs}")
    os.makedirs(OUT, exist_ok=True)
    with open(REPRO_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"saved -> {REPRO_PATH}")
    return report


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 variants x 1 seed x 1 fold x 2 epochs -> separate json")
    ap.add_argument("--repro-test", action="store_true")
    ap.add_argument("--repro-one", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(N_SPLITS)))
    ap.add_argument("--results", default=None)
    ap.add_argument("--no-classical", action="store_true")
    args = ap.parse_args()

    if args.repro_one:
        repro_one(args.repro_one, args.epochs)
        return

    keep_awake()
    if args.repro_test:
        print("log ->", setup_logging("repro_test"))
        repro_test(epochs=args.epochs if args.epochs != 100 else 5)
        return

    if args.smoke:
        print("log ->", setup_logging("smoke"))
        log("SMOKE TEST: classical fold 0 + nir & nir_SE, seed 0, fold 0, 2 epochs -> " + SMOKE_RESULTS_PATH)
        run_batch(args.results or SMOKE_RESULTS_PATH, [["nir", "nir_SE"]], [0], [0], epochs=2,
                  do_classical=True, allow_ckpt=False)
        return

    print("log ->", setup_logging("overnight"))
    log(f"overnight batch start: seeds={args.seeds} folds={args.folds} -> {args.results or RESULTS_PATH}")
    run_batch(args.results or RESULTS_PATH, DL_GROUPS, args.seeds, args.folds, args.epochs,
              do_classical=not args.no_classical)


if __name__ == "__main__":
    main()
