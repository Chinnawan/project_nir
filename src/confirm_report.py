"""Report of the confirmatory run (seeds 3,4,5 only; seeds 0-2 are never used here).

Locked formula: nir_ss + TTA + 3-seed soft vote inside a fold. Baseline: nir (raw images) run with
the very same seeds/folds/TTA, so every comparison is paired.

  1. per variant: macro-F1 / accuracy (mean +- std of the 5-fold mean across the 3 seeds), plain and TTA,
     per-class F1 (pooled out-of-fold)
  2. paired comparisons (rule from the multi-seed report: CI excludes 0 AND >= 12/15 same sign, only with
     all 15 pairs) + fold-level check:  nir_ss - nir (plain),  nir_ss[TTA] - nir[TTA],  nir[TTA] - nir,
     nir_ss[TTA] - nir (formula vs the old baseline);  exploratory: nir_ss_ls - nir_ss
  3. the numbers for the thesis: 3-seed soft-vote ensemble (+TTA) per variant, mean over the 5 folds and the
     pooled out-of-fold confusion matrix / precision / recall of the final formula over all 819 apples
  4. error rate per tissue for nir vs nir_ss

Usage: python -m src.confirm_report [--results PATH]  ->  outputs/dl/confirm_summary.json
"""
import argparse
import json
import os
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .config import CLASSES, OUTPUT_DIR
from .multiseed_report import paired, summarize_variant
from .round2 import Context
from .round2_report import ensemble_over_folds, fold_level, rec_table, tissue_errors, verdict
from .run_confirm import RESULTS_PATH

SUMMARY_PATH = os.path.join(OUTPUT_DIR, "dl", "confirm_summary.json")
CORE = ["nir_ss", "nir"]
EXPLORATORY = ["nir_ss_ls"]


def pooled_ensemble(tables, labels):
    """Soft vote of all members per fold, concatenated over folds -> (y_true, y_pred) over every apple."""
    ys, ps = [], []
    for f in sorted({f for t in tables for (_, f) in t}):
        recs = [r for t in tables for (s, ff), r in t.items() if ff == f]
        idx = np.array(recs[0]["test_idx"])
        ys.append(labels[idx])
        ps.append(np.mean([np.array(r["probs"]) for r in recs], axis=0).argmax(1))
    return np.concatenate(ys), np.concatenate(ps)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--summary-out", default=SUMMARY_PATH)
    args = ap.parse_args()

    data = json.load(open(args.results, encoding="utf-8"))
    runs = data["runs"]
    assert all(r["seed"] >= 3 for r in runs.values()), "confirm results must only contain seeds >= 3"
    ctx = Context()
    rng = np.random.RandomState(42)
    names = [v for v in CORE + EXPLORATORY if any(r["variant"] == v for r in runs.values())]
    plain = {v: rec_table(runs, v) for v in names}
    tta = {v: rec_table(runs, v, tta=True) for v in names}
    n_err = len(data.get("errors", []))
    print(f"{len(runs)} confirmatory runs (seeds {sorted({r['seed'] for r in runs.values()})}) in {args.results}"
          + (f"  ({n_err} FAILED runs under 'errors'!)" if n_err else ""))
    summary = {"variants": {}, "paired": {}, "ensembles": {}, "final_formula": {}}
    sd = lambda x: "  n/a" if x is None else f"{x:.3f}"

    print("\n=== 1) Per variant (mean +- std across seeds of the 5-fold mean) ===")
    print(f"  {'variant':11s} {'seeds':>5s} {'F1 plain':>14s} {'acc':>7s} | {'F1 TTA':>14s} {'acc':>7s} | per-class F1 TTA (H / Int / Surf)")
    for v in names:
        s, st = summarize_variant(plain[v]), summarize_variant(tta[v])
        summary["variants"][v] = {"plain": s, "tta": st, "exploratory": v in EXPLORATORY}
        if "f1_mean_of_seed_means" not in s:
            print(f"  {v:11s} incomplete ({s['n_runs']} runs)")
            continue
        pc = st["pooled_oof_per_class_f1"]
        print(f"  {v + ('*' if v in EXPLORATORY else ''):11s} {s['n_complete_seeds']:5d} {s['f1_mean_of_seed_means']:.3f} +- {sd(s['f1_std_of_seed_means'])} "
              f"{s['acc_mean_of_seed_means'] * 100:6.1f}% | {st['f1_mean_of_seed_means']:.3f} +- {sd(st['f1_std_of_seed_means'])} "
              f"{st['acc_mean_of_seed_means'] * 100:6.1f}% | {pc['Healthy']:.3f} / {pc['Internal_Defect']:.3f} / {pc['Surface_Defect']:.3f}")
    if EXPLORATORY[0] in names:
        print("  (* exploratory, not part of the locked formula)")

    print("\n=== 2) Paired comparisons (A - B in macro-F1; paired by seed+fold) ===")
    comps = [("nir_ss - nir", plain.get("nir_ss"), plain.get("nir")),
             ("nir_ss[TTA] - nir[TTA]", tta.get("nir_ss"), tta.get("nir")),
             ("nir[TTA] - nir", tta.get("nir"), plain.get("nir")),
             ("nir_ss[TTA] - nir  (formula vs old baseline)", tta.get("nir_ss"), plain.get("nir")),
             ("nir_ss[TTA] - nir_ss", tta.get("nir_ss"), plain.get("nir_ss")),
             ("[exploratory] nir_ss_ls - nir_ss", plain.get("nir_ss_ls"), plain.get("nir_ss")),
             ("[exploratory] nir_ss_ls[TTA] - nir_ss[TTA]", tta.get("nir_ss_ls"), tta.get("nir_ss"))]
    for name, A, B in comps:
        if not A or not B:
            continue
        r = paired(A, B, rng)
        if r is None:
            continue
        fl = fold_level(A, B)
        summary["paired"][name] = {**r, "fold_level": fl, "verdict": verdict(r)}
        fl_txt = (f"folds {fl['same_sign_folds']}/{fl['n_folds']} same sign, fold-CI [{fl['ci95'][0]:+.3f},{fl['ci95'][1]:+.3f}]"
                  if fl else "")
        print(f"  {name:46s} mean {r['mean_diff']:+.3f}  CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  same sign "
              f"{r['same_sign_pairs']}/{r['n_pairs']}  {fl_txt}  -> {verdict(r)}")

    print("\n=== 3) Numbers for the thesis: 3-seed ensembles inside a fold (members trained on the same training fold) ===")
    print(f"  {'ensemble':38s} {'F1 (mean over folds)':>26s} {'acc':>7s}")
    for v in names:
        for label, tabs, soft in ((f"{v}: 3-seed soft vote", plain[v], True), (f"{v}[TTA]: 3-seed soft vote", tta[v], True),
                                  (f"{v}: 3-seed hard vote", plain[v], False)):
            e = ensemble_over_folds([tabs], ctx.labels, soft)
            if e is None:
                continue
            summary["ensembles"][label] = e
            print(f"  {label:38s} {e['f1_mean']:.3f} +- {e['f1_std']:.3f} (folds){'':3s}{e['acc_mean'] * 100:6.1f}%")

    if "nir_ss" in names and len(tta["nir_ss"]) == 15:
        y, p = pooled_ensemble([tta["nir_ss"]], ctx.labels)
        cm = confusion_matrix(y, p, labels=[0, 1, 2])
        pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        summary["final_formula"] = {
            "definition": "nir_ss + TTA(6 views) + 3-seed soft vote, pooled out-of-fold over all 819 apples",
            "accuracy": float(accuracy_score(y, p)), "macro_f1": float(f1_score(y, p, average="macro")),
            "confusion_matrix": cm.tolist(), "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]),
                                                              "f1": float(f1[i]), "n": int(sup[i])} for i, c in enumerate(CLASSES)}}
        print("\n  FINAL FORMULA, pooled out-of-fold over 819 apples: "
              f"accuracy {accuracy_score(y, p) * 100:.1f}%  macro-F1 {f1_score(y, p, average='macro'):.3f}")
        print(f"  {'':18s} " + " ".join(f"pred {c[:8]:>8s}" for c in CLASSES) + "   precision recall   F1")
        for i, c in enumerate(CLASSES):
            print(f"  true {c:13s} " + " ".join(f"{cm[i, j]:13d}" for j in range(3)) +
                  f"   {pr[i]:.3f}     {rc[i]:.3f}  {f1[i]:.3f}")
        if "nir" in names and len(tta["nir"]) == 15:
            y0, p0 = pooled_ensemble([tta["nir"]], ctx.labels)
            summary["final_formula"]["baseline_nir_tta_ensemble"] = {"accuracy": float(accuracy_score(y0, p0)),
                                                                     "macro_f1": float(f1_score(y0, p0, average="macro"))}
            fa = ensemble_over_folds([tta["nir_ss"]], ctx.labels, True)["fold_f1"]
            fb = ensemble_over_folds([tta["nir"]], ctx.labels, True)["fold_f1"]
            d = np.array(fa) - np.array(fb)
            summary["final_formula"]["ensemble_fold_diff_vs_nir_tta_ensemble"] = d.tolist()
            print(f"  baseline nir[TTA] 3-seed soft vote, pooled: accuracy {accuracy_score(y0, p0) * 100:.1f}%  macro-F1 "
                  f"{f1_score(y0, p0, average='macro'):.3f};  per-fold F1 difference (nir_ss - nir): {np.round(d, 3).tolist()}"
                  f"  ({int((d > 0).sum())}/5 folds favour nir_ss)")

    print("\n=== 4) Error rate per tissue (plain predictions, mean over seeds) ===")
    te = {v: tissue_errors(plain[v], ctx) for v in names}
    summary["tissue_error"] = te
    tissues = ["Rot", "Russet", "Hail_damage_perf", "Other", "Scald", "Frost_damage", "Hail_damage", "Limb_rub", "Bruise",
               "Flesh_damage", "Stem", "Calix", "Healthy"]
    print(f"  {'tissue':17s} {'n':>4s} " + " ".join(f"{v:>10s}" for v in names))
    for t in tissues:
        n = next((te[v][t]["n_apples"] for v in names if t in te[v]), 0)
        print(f"  {t:17s} {n:4d} " + " ".join(f"{te[v][t]['error_rate'] * 100:9.1f}%" if t in te[v] else f"{'-':>10s}" for v in names))

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump({"results_file": args.results, "n_runs": len(runs), "n_failed_runs": n_err, **summary}, f, indent=2)
    print(f"\nsaved -> {args.summary_out}")


if __name__ == "__main__":
    main()
