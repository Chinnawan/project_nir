"""Report for the single-band ablation (re_only, ir_only) vs nir (RE+IR).

Reads outputs/dl/kfold_results.json (macro-F1/acc per fold, plus stored
predictions for re_only/ir_only). `nir` was trained before predictions were
stored, so its out-of-fold predictions come from the per-fold checkpoints in
outputs/dl/kfold_ckpt. CAUTION: those checkpoints are from a SEPARATE training run
(src.ensemble_kfold) with the same recipe/folds, not the run behind the nir numbers
in kfold_results.json -- per-fold F1 differs by up to ~0.06 (run-to-run noise from
non-deterministic GPU kernels and unseeded Python-`random` augmentation). The
per-class / McNemar rows for nir therefore describe that second run; the
macro-F1 mean+-std table uses kfold_results.json.

Outputs: console tables + outputs/dl/band_ablation.json
Usage:   python -m src.band_ablation_report
"""
import json
import os

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .config import CLASSES, OUTPUT_DIR
from .dataset import ApplesDataset
from .explain import mcnemar_test
from .models import build_model
from .train_dl_kfold import CKPT_DIR, N_SPLITS, RESULTS_PATH, build_pool, get_fold_splits

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_JSON = os.path.join(OUTPUT_DIR, "dl", "band_ablation.json")
VARIANTS = ["re_only", "ir_only", "nir"]
LABELS = {"re_only": "RE only", "ir_only": "IR only", "nir": "RE+IR (nir)"}


def nir_predictions_from_ckpts(pool, splits):
    """Out-of-fold predictions of the 5 `nir` checkpoints, in each fold's test_idx order."""
    preds_by_fold = []
    for f, (_, test_idx) in enumerate(splits):
        ckpt = torch.load(os.path.join(CKPT_DIR, f"nir_fold{f}.pt"), map_location=DEVICE, weights_only=False)
        model = build_model("nir", num_classes=len(CLASSES), pretrained=False, use_se=False).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        ds = ApplesDataset("test", "nir", augment=False, mean=ckpt["mean"], std=ckpt["std"],
                            items=[pool[i] for i in test_idx])
        out = []
        with torch.no_grad():
            for xb, _ in DataLoader(ds, batch_size=32, shuffle=False):
                out.append(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
        preds_by_fold.append(np.concatenate(out).tolist())
    return preds_by_fold


def main():
    with open(RESULTS_PATH) as f:
        per_fold = json.load(f)["per_fold"]
    for v in ("re_only", "ir_only", "nir"):
        assert len(per_fold.get(v, [])) == N_SPLITS, f"{v} missing from kfold_results.json"

    pool, labels = build_pool("raw")
    splits = get_fold_splits(pool, labels)

    # out-of-fold predictions placed back in pool order, per variant
    oof = {}
    fold_preds = {v: [r["test_predictions"] for r in per_fold[v]] for v in ("re_only", "ir_only")}
    fold_preds["nir"] = nir_predictions_from_ckpts(pool, splits)
    for v in VARIANTS:
        arr = np.full(len(pool), -1, dtype=int)
        for f, (_, test_idx) in enumerate(splits):
            assert len(fold_preds[v][f]) == len(test_idx)
            arr[test_idx] = fold_preds[v][f]
        assert (arr >= 0).all()
        oof[v] = arr

    # sanity: stored per-fold labels match the fold split; recomputed nir F1 matches stored F1
    for v in ("re_only", "ir_only"):
        for f, (_, test_idx) in enumerate(splits):
            assert per_fold[v][f]["test_labels"] == labels[test_idx].tolist(), (v, f)
    nir_check = [abs(f1_score(labels[te], fold_preds["nir"][f], average="macro") - per_fold["nir"][f]["test_macro_f1"])
                 for f, (_, te) in enumerate(splits)]
    ckpt_run_f1 = [f1_score(labels[te], fold_preds["nir"][f], average="macro") for f, (_, te) in enumerate(splits)]
    stored_f1 = [per_fold["nir"][f]["test_macro_f1"] for f in range(N_SPLITS)]
    print("NOTE: nir checkpoints are a separate training run of the same recipe (not the kfold_results.json run).")
    print(f"  per-fold F1  kfold_results.json run: {np.round(stored_f1, 3).tolist()}")
    print(f"  per-fold F1  checkpoint (2nd) run  : {np.round(ckpt_run_f1, 3).tolist()}  "
          f"(max |diff| = {max(nir_check):.3f} -> run-to-run noise of one fold)")

    f1s = {v: np.array([r["test_macro_f1"] for r in per_fold[v]]) for v in VARIANTS}
    accs = {v: np.array([r["test_accuracy"] for r in per_fold[v]]) for v in VARIANTS}

    print("\n=== 5-fold test results (raw images, same folds) ===")
    print(f"  {'variant':14s} {'Macro F1':>16s} {'Accuracy':>16s}   per-fold F1")
    for v in VARIANTS:
        print(f"  {LABELS[v]:14s} {f1s[v].mean():.3f} +/- {f1s[v].std():.3f}   "
              f"{accs[v].mean()*100:5.1f}% +/- {accs[v].std()*100:.1f}%   {np.round(f1s[v], 3).tolist()}")

    report = {"nir_two_runs_per_fold_f1": {"kfold_results_run": stored_f1, "checkpoint_run": ckpt_run_f1},
              "nir_two_runs_max_abs_f1_diff": float(max(nir_check)), "variants": {}, "paired": {}, "mcnemar_pooled_oof": {}}
    for v in VARIANTS:
        pooled_pc = f1_score(labels, oof[v], average=None, labels=[0, 1, 2])
        report["variants"][v] = {
            "macro_f1_mean": float(f1s[v].mean()), "macro_f1_std": float(f1s[v].std()),
            "accuracy_mean": float(accs[v].mean()), "accuracy_std": float(accs[v].std()),
            "per_fold_macro_f1": f1s[v].tolist(), "per_fold_accuracy": accs[v].tolist(),
            "pooled_oof_macro_f1": float(f1_score(labels, oof[v], average="macro")),
            "pooled_oof_accuracy": float(accuracy_score(labels, oof[v])),
            "pooled_oof_per_class_f1": dict(zip(CLASSES, map(float, pooled_pc))),
        }

    print("\n=== Per-class F1 (pooled out-of-fold over all 819 apples) ===")
    print(f"  {'variant':14s} " + " ".join(f"{c:>16s}" for c in CLASSES) + f" {'macro':>8s}")
    for v in VARIANTS:
        pc = report["variants"][v]["pooled_oof_per_class_f1"]
        print(f"  {LABELS[v]:14s} " + " ".join(f"{pc[c]:16.3f}" for c in CLASSES) +
              f" {report['variants'][v]['pooled_oof_macro_f1']:8.3f}")

    print("\n=== Paired per-fold differences in macro-F1 (same apples in each fold) ===")
    for a, b in (("re_only", "ir_only"), ("nir", "re_only"), ("nir", "ir_only")):
        d = f1s[a] - f1s[b]
        t = stats.ttest_rel(f1s[a], f1s[b])
        pos = int((d > 0).sum())
        report["paired"][f"{a}_minus_{b}"] = {
            "diffs": d.tolist(), "mean": float(d.mean()), "std": float(d.std(ddof=1)),
            "folds_a_better": pos, "paired_t_p": float(t.pvalue),
        }
        print(f"  {LABELS[a]} - {LABELS[b]}: mean {d.mean():+.3f} (sd {d.std(ddof=1):.3f}), "
              f"{pos}/{N_SPLITS} folds favour {LABELS[a]}, diffs={np.round(d, 3).tolist()}, "
              f"paired-t p={t.pvalue:.3f} (n=5, indicative only)")

    print("\n=== McNemar on pooled out-of-fold predictions (n=819; folds share training data, so approximate) ===")
    for a, b in (("re_only", "ir_only"), ("nir", "re_only"), ("nir", "ir_only")):
        m = mcnemar_test(labels, oof[a], oof[b])
        a_only, b_only = m["table"][0][1], m["table"][1][0]
        report["mcnemar_pooled_oof"][f"{a}_vs_{b}"] = {**m, "a_only_correct": a_only, "b_only_correct": b_only}
        print(f"  {LABELS[a]} vs {LABELS[b]}: only-{LABELS[a]} correct {a_only}, only-{LABELS[b]} correct {b_only}, "
              f"p={m['p_value']:.4f}")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
