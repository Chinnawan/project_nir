"""K4: train nir_ss and visible_ss directly as 2-class models (Healthy vs Defect = Internal or Surface).

Same protocol as nir_ss (confirm_results.json) / visible_ss (n5_results.json): SingleSpectra images, seeds 3,4,5,
StratifiedKFold(5, seed 42) + the same inner val split, legacy init, augmentation, no SE, 224 px, TTA. Only the
target changes (2 classes instead of 3). The sampler keeps the class-balanced weights of the 3 real classes (as for
nir_ss), so the training data stream is identical and only the head / loss differ.

  python -m src.run_k4            (resumable)  -> outputs/dl/k4_results.json
  python -m src.run_k4 --smoke    1 variant x seed 3 x fold 0 x 2 epochs -> separate file
"""
import argparse
import os
import time

import numpy as np

from . import repro  # noqa: F401
from .config import OUTPUT_DIR
from .round2 import Context, run_round2_fold
from .run_multiseed import keep_awake, load_results, log, record_error, run_key, save_results, setup_logging

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "k4_results.json")
SMOKE_PATH = os.path.join(OUT, "k4_results_smoke.json")
VARIANTS = {"nir_ss_2cls": {"channels": "nir", "source": "ss", "n_classes": 2},
            "visible_ss_2cls": {"channels": "visible", "source": "ss", "n_classes": 2}}
ORDER = ["nir_ss_2cls", "visible_ss_2cls"]
SEEDS = [3, 4, 5]


def run(path, variants, seeds, folds, epochs):
    data = load_results(path)
    if not data.get("meta"):
        data["meta"] = {"purpose": "K4: 2-class (Healthy vs Defect) nir_ss / visible_ss, paired with the 3-class runs",
                        "seeds": seeds, "variants": VARIANTS, "classes": ["Healthy", "Defect"]}
        save_results(path, data)
    ctx = Context()
    total = sum(run_key(v, s, f) not in data["runs"] for v in variants for s in seeds for f in folds)
    log(f"K4: {total} runs to do")
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
                    rec = run_round2_fold(ctx, variant, seed, fold, epochs=epochs, variants=VARIANTS)
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
    log("K4 finished" + (f" with {len(data['errors'])} errors" if data.get("errors") else ""))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    keep_awake()
    if args.smoke:
        print("log ->", setup_logging("k4_smoke"))
        run(SMOKE_PATH, ["nir_ss_2cls"], [3], [0], epochs=2)
        return
    print("log ->", setup_logging("k4"))
    run(RESULTS_PATH, ORDER, SEEDS, list(range(5)), epochs=100)


if __name__ == "__main__":
    main()
