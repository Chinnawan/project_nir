"""Summary of the overnight multi-seed run (outputs/dl/multiseed_results.json).

Prints tables and writes outputs/dl/multiseed_summary.json:
  1. per variant: Macro-F1 / accuracy as mean +- std of the 5-fold means across seeds, and
     mean +- std over all fold-level runs; per-class F1 from the pooled out-of-fold
     confusion matrix (per seed, then averaged over seeds)
  2. paired comparisons (paired by seed + fold): mean difference, bootstrap 95% CI over the
     pairs, number of pairs with the same sign. Verdict "real difference" only if the CI
     excludes 0 AND >= 12/15 pairs share the sign (for 3 seeds x 5 folds); otherwise the
     difference is "not larger than the training noise"
  3. seed noise: per (variant, fold) std of F1 across seeds

Classical models are deterministic and stored as seed 0 only; they are reused for every seed
when pairing. The 15 pairs share the same 5 folds, so the CI treats fold-sharing pairs as
independent -- it is an approximation (noted in the output).

Usage: python -m src.multiseed_report [--results PATH]
"""
import argparse
import json
import os

import numpy as np

from .config import CLASSES, OUTPUT_DIR

OUT = os.path.join(OUTPUT_DIR, "dl")
RESULTS_PATH = os.path.join(OUT, "multiseed_results.json")
SUMMARY_PATH = os.path.join(OUT, "multiseed_summary.json")
CLASSICAL = ["SVM", "RandomForest", "LightGBM"]
DL_VARIANTS = ["nir", "nir_SE", "nir_no_aug", "re_only", "ir_only", "bl_only", "gr_only", "visible", "full",
               "full_SE", "gr_nir"]
BANDS = ["bl_only", "gr_only", "re_only", "ir_only"]
MIN_SAME_SIGN_FRAC = 12 / 15
MIN_PAIRS_FOR_VERDICT = 15  # 3 seeds x 5 folds; fewer pairs -> no verdict (a CI over 1-14 pairs is meaningless)
N_BOOT = 10000


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def table_for(runs, variant, broadcast=True):
    """{(seed, fold): record} for one variant. With broadcast=True, classical variants
    (seed 0 only, deterministic) are copied to every seed present in the file so that
    pairing by (seed, fold) works; summaries use broadcast=False."""
    recs = {(r["seed"], r["fold"]): r for r in runs.values() if r["variant"] == variant}
    if broadcast and variant in CLASSICAL and recs:
        seeds = sorted({r["seed"] for r in runs.values() if r["kind"] == "dl"}) or [0]
        recs = {(s, f): r for (s0, f), r in recs.items() for s in seeds}
    return recs


def f1_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(axis=0), 1e-12)
    rec = tp / np.maximum(cm.sum(axis=1), 1e-12)
    return 2 * prec * rec / np.maximum(prec + rec, 1e-12)


def summarize_variant(recs, n_folds=5):
    seeds = sorted({s for s, _ in recs})
    per_seed_f1, per_seed_acc, per_seed_pc = [], [], []
    for s in seeds:
        folds = [recs[(s, f)] for f in range(n_folds) if (s, f) in recs]
        if len(folds) < n_folds:
            continue  # incomplete seed: not used for seed-level stats
        per_seed_f1.append(np.mean([r["test_macro_f1"] for r in folds]))
        per_seed_acc.append(np.mean([r["test_accuracy"] for r in folds]))
        cm = np.sum([np.asarray(r["confusion_matrix"]) for r in folds], axis=0)
        per_seed_pc.append(f1_from_cm(cm))
    all_f1 = np.array([r["test_macro_f1"] for r in recs.values()])
    all_acc = np.array([r["test_accuracy"] for r in recs.values()])
    out = {"n_runs": len(recs), "n_complete_seeds": len(per_seed_f1),
           "f1_all_runs_mean": float(all_f1.mean()), "f1_all_runs_std": float(all_f1.std(ddof=1)) if len(all_f1) > 1 else None,
           "acc_all_runs_mean": float(all_acc.mean()), "acc_all_runs_std": float(all_acc.std(ddof=1)) if len(all_acc) > 1 else None}
    if per_seed_f1:
        out.update({
            "f1_seed_means": [float(x) for x in per_seed_f1],
            "f1_mean_of_seed_means": float(np.mean(per_seed_f1)),
            "f1_std_of_seed_means": float(np.std(per_seed_f1, ddof=1)) if len(per_seed_f1) > 1 else None,
            "acc_mean_of_seed_means": float(np.mean(per_seed_acc)),
            "acc_std_of_seed_means": float(np.std(per_seed_acc, ddof=1)) if len(per_seed_acc) > 1 else None,
            "pooled_oof_per_class_f1": dict(zip(CLASSES, map(float, np.mean(per_seed_pc, axis=0)))),
        })
    return out


def paired(recs_a, recs_b, rng):
    keys = sorted(set(recs_a) & set(recs_b))
    if not keys:
        return None
    d = np.array([recs_a[k]["test_macro_f1"] - recs_b[k]["test_macro_f1"] for k in keys])
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean = float(d.mean())
    same = int(np.sum(np.sign(d) == np.sign(mean))) if mean != 0 else 0
    need = int(np.ceil(MIN_SAME_SIGN_FRAC * len(d)))
    complete = len(d) >= MIN_PAIRS_FOR_VERDICT
    real = (bool(lo > 0 or hi < 0) and same >= need) if complete else None
    return {"n_pairs": len(d), "mean_diff": mean, "ci95": [float(lo), float(hi)], "same_sign_pairs": same,
            "same_sign_needed": need, "real_difference": real, "complete": complete, "diffs": d.tolist()}


def best_of(summary, names):
    cand = [(summary[n]["f1_all_runs_mean"], n) for n in names if n in summary]
    return max(cand)[1] if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--summary-out", default=SUMMARY_PATH)
    args = ap.parse_args()

    data = load(args.results)
    runs = data["runs"]
    n_err = len(data.get("errors", []))
    variants = [v for v in CLASSICAL + DL_VARIANTS if any(r["variant"] == v for r in runs.values())]
    tables = {v: table_for(runs, v) for v in variants}
    summary = {v: summarize_variant(table_for(runs, v, broadcast=False)) for v in variants}
    rng = np.random.RandomState(42)

    print(f"{len(runs)} runs in {args.results}" + (f"  ({n_err} failed runs recorded under 'errors'!)" if n_err else ""))
    print("\n=== 1) Per variant (mean +- std across seeds of the 5-fold mean | std over all fold-level runs) ===")
    print(f"  {'variant':13s} {'seeds':>5s} {'Macro F1 (seed means)':>24s} {'Accuracy (seed means)':>24s} "
          f"{'F1 all runs':>14s}   per-class F1 pooled OOF (Healthy / Internal / Surface)")
    for v in variants:
        s = summary[v]
        if "f1_mean_of_seed_means" not in s:
            print(f"  {v:13s} incomplete ({s['n_runs']} runs)")
            continue
        sd = lambda x: "n/a" if x is None else f"{x:.3f}"
        note = "  (deterministic, 1 run)" if v in CLASSICAL else ""
        pc = s["pooled_oof_per_class_f1"]
        print(f"  {v:13s} {s['n_complete_seeds']:5d} {s['f1_mean_of_seed_means']:.3f} +- {sd(s['f1_std_of_seed_means']):>5s}"
              f"{'':9s}{s['acc_mean_of_seed_means'] * 100:5.1f}% +- {sd(s['acc_std_of_seed_means'] and s['acc_std_of_seed_means'] * 100):>5s}"
              f"{'':7s}{s['f1_all_runs_mean']:.3f} +- {sd(s['f1_all_runs_std'])}   "
              f"{pc['Healthy']:.3f} / {pc['Internal_Defect']:.3f} / {pc['Surface_Defect']:.3f}{note}")

    best_dl = best_of(summary, [v for v in DL_VARIANTS if v in summary])
    best_cl = best_of(summary, [v for v in CLASSICAL if v in summary])
    pairs = [("nir_SE", "nir"), ("nir", "nir_no_aug"), ("nir", "full"), ("nir", "visible"), ("re_only", "ir_only"),
             ("gr_nir", "nir")]
    pairs += [(a, b) for i, a in enumerate(BANDS) for b in BANDS[i + 1:] if (a, b) not in pairs]
    if best_dl and best_cl:
        pairs.append((best_dl, best_cl))

    print("\n=== 2) Paired comparisons (paired by seed+fold; A - B in macro-F1) ===")
    print("  'real difference' = 95% bootstrap CI excludes 0 AND >= 12/15 pairs share the sign; otherwise:")
    print("  'not larger than training noise'.  (Pairs share folds -> CI is approximate.)")
    comp = {}
    for a, b in pairs:
        if a not in tables or b not in tables:
            continue
        r = paired(tables[a], tables[b], rng)
        if r is None:
            continue
        comp[f"{a}_minus_{b}"] = r
        verdict = ("INCOMPLETE (only %d of %d pairs, no verdict)" % (r["n_pairs"], MIN_PAIRS_FOR_VERDICT)
                   if not r["complete"] else
                   "REAL difference" if r["real_difference"] else "not larger than training noise")
        print(f"  {a:>11s} - {b:<12s} mean {r['mean_diff']:+.3f}  CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  "
              f"same sign {r['same_sign_pairs']}/{r['n_pairs']} (need {r['same_sign_needed']})  -> {verdict}")
    if best_dl and best_cl:
        print(f"  (best DL = {best_dl}, best classical = {best_cl}, chosen by mean F1 over all runs)")

    print("\n=== 3) Seed noise after locking the seed: std of F1 across seeds, per fold ===")
    noise = {}
    for v in variants:
        if v in CLASSICAL:
            continue
        per_fold = []
        for f in range(5):
            vals = [tables[v][(s, f)]["test_macro_f1"] for s in sorted({s for s, _ in tables[v]}) if (s, f) in tables[v]]
            if len(vals) >= 2:
                per_fold.append(float(np.std(vals, ddof=1)))
        if per_fold:
            noise[v] = {"per_fold_std": per_fold, "mean_std": float(np.mean(per_fold)), "max_std": float(np.max(per_fold))}
            print(f"  {v:13s} mean {np.mean(per_fold):.3f}  max {np.max(per_fold):.3f}   per fold {np.round(per_fold, 3).tolist()}")

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump({"results_file": args.results, "n_runs": len(runs), "n_failed_runs": n_err, "variants": summary,
                   "paired": comp, "seed_noise": noise, "best_dl": best_dl, "best_classical": best_cl}, f, indent=2)
    print(f"\nsaved -> {args.summary_out}")


if __name__ == "__main__":
    main()
