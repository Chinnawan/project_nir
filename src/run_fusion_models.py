"""N4 support run: train `visible` (BL+GR) and `full` (all 4 bands) on the raw images with seeds 3,4,5,
storing softmax probabilities, so they can be paired with `nir` (raw) from confirm_results.json, which was
trained with the same seeds / folds / protocol and already has probabilities.

Needed because multiseed_results.json (seeds 0-2) only stored predictions, and late fusion needs probabilities.
`full` is included so early fusion (4 bands in one network) and late fusion (average of the nir and visible
networks) are compared on the same seeds.

  python -m src.run_fusion_models          (resumable)  -> outputs/dl/fusion_results.json
  python -m src.run_fusion_models --smoke  (1 fold, 1 epoch, separate file)
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
RESULTS_PATH = os.path.join(OUT, "fusion_results.json")
SMOKE_PATH = os.path.join(OUT, "fusion_results_smoke.json")
VARIANTS = {"visible": {"channels": "visible"}, "full": {"channels": "full"}}  # raw images, legacy init = multiseed recipe
ORDER = ["visible", "full"]
SEEDS = [3, 4, 5]


def run(path, seeds, folds, epochs):
    data = load_results(path)
    if not data.get("meta"):
        data["meta"] = {"purpose": "N4 late fusion: probabilities for visible/full, paired with nir in confirm_results.json",
                        "seeds": seeds, "variants": VARIANTS}
        save_results(path, data)
    ctx = Context()
    total = sum(run_key(v, s, f) not in data["runs"] for v in ORDER for s in seeds for f in folds)
    log(f"fusion models: {total} runs to do")
    done, times = 0, []
    for variant in ORDER:
        for seed in seeds:
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
                log(f"END   {key}: F1={rec['test_macro_f1']:.4f} acc={rec['test_accuracy']:.4f} "
                    f"epochs={rec['epochs_trained']} ({times[-1]:.0f}s) | {done}/{total}, ETA ~{eta:.0f} min")
    log("fusion models finished" + (f" with {len(data['errors'])} errors" if data.get("errors") else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    keep_awake()
    if args.smoke:
        print("log ->", setup_logging("fusion_smoke"))
        run(SMOKE_PATH, [3], [0], epochs=1)
        return
    print("log ->", setup_logging("fusion"))
    run(RESULTS_PATH, SEEDS, list(range(5)), epochs=100)


if __name__ == "__main__":
    main()
