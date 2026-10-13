"""Summary of the round-2 batch (outputs/dl/round2_results.json) -> outputs/dl/round2_summary.json
and a printed report (the .bat saves it to outputs/logs/round2_report.txt).

  1. per variant: Macro-F1 / accuracy (mean +- std of the 5-fold mean across seeds), plain and TTA,
     per-class F1 (pooled out-of-fold, plain)
  2. paired comparisons (paired by seed + fold) against each variant's own baseline:
       plain - baseline, TTA - baseline(plain), TTA - plain (same variant)
     same rule as before: "real difference" = 95% bootstrap CI excludes 0 AND >= 12/15 pairs share
     the sign, only judged with all 15 pairs; plus the fold-level check (mean over seeds per fold,
     then count the 5 folds) because the 15 pairs share 5 folds
  3. error rate per tissue type (Rot, Russet, Stem, ...), plain predictions, mean over seeds
  4. ensembles WITHIN a fold (all members trained on the same training fold, so no leakage):
     3-seed soft vote (mean probability) and hard vote per variant, and group soft votes.
     Group choices are made after seeing results -> exploratory, must be confirmed with new seeds.
Baselines come from multiseed_results.json (same protocol); they have predictions but no
probabilities, so for them only hard votes are possible.

Usage: python -m src.round2_report [--results PATH]
"""
import argparse
import json
import os
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import OUTPUT_DIR
from .multiseed_report import MIN_PAIRS_FOR_VERDICT, paired, summarize_variant, table_for
from .round2 import BASELINES, VARIANTS, Context

OUT = os.path.join(OUTPUT_DIR, "dl")
R2_PATH = os.path.join(OUT, "round2_results.json")
MS_PATH = os.path.join(OUT, "multiseed_results.json")
SUMMARY_PATH = os.path.join(OUT, "round2_summary.json")
SS_VARIANTS = ["nir_ss", "nir_ss_cp"]
RAW_NIR_GROUP = ["nir_5cls", "nir_ls", "nir_fix", "nir_320", "nir_ss_plus"]
TISSUES = ["Rot", "Russet", "Hail_damage_perf", "Other", "Scald", "Frost_damage", "Hail_damage", "Limb_rub",
           "Bruise", "Flesh_damage", "Stem", "Calix", "Healthy"]
T975_DF4 = 2.776


def rec_table(runs, variant, tta=False):
    """{(seed, fold): record with the generic keys the multiseed helpers expect}."""
    out = {}
    for r in runs.values():
        if r["variant"] != variant:
            continue
        p = "tta_" if tta else ""
        out[(r["seed"], r["fold"])] = {
            "test_macro_f1": r[p + "test_macro_f1"], "test_accuracy": r[p + "test_accuracy"],
            "confusion_matrix": r[p + "confusion_matrix"], "test_idx": r["test_idx"],
            "preds": r[p + "test_predictions"], "probs": r.get(p + "test_probs"), "seed": r["seed"], "fold": r["fold"]}
    return out


def ms_table(ms_runs, variant):
    return {k: {**r, "preds": r["test_predictions"], "probs": None}
            for k, r in table_for(ms_runs, variant, broadcast=False).items()}


def fold_level(a, b):
    keys = sorted(set(a) & set(b))
    folds = sorted({f for _, f in keys})
    pf = np.array([np.mean([a[(s, f)]["test_macro_f1"] - b[(s, f)]["test_macro_f1"] for (s, ff) in keys if ff == f])
                   for f in folds])
    if len(pf) < 3:
        return None
    se = pf.std(ddof=1) / np.sqrt(len(pf))
    return {"per_fold": pf.tolist(), "mean": float(pf.mean()), "ci95": [float(pf.mean() - T975_DF4 * se),
            float(pf.mean() + T975_DF4 * se)], "same_sign_folds": int((np.sign(pf) == np.sign(pf.mean())).sum()),
            "n_folds": len(pf)}


def verdict(r):
    if not r["complete"]:
        return f"INCOMPLETE ({r['n_pairs']}/{MIN_PAIRS_FOR_VERDICT} pairs, no verdict)"
    return "REAL difference" if r["real_difference"] else "not larger than training noise"


def vote(preds_list, soft_probs_list=None):
    if soft_probs_list is not None:
        return np.mean(soft_probs_list, axis=0).argmax(1)
    P = np.stack(preds_list)
    out = np.zeros(P.shape[1], dtype=int)
    for j in range(P.shape[1]):
        c = np.bincount(P[:, j], minlength=3)
        cand = np.where(c == c.max())[0]
        out[j] = P[0, j] if P[0, j] in cand else cand[0]
    return out


def ensemble_over_folds(members, labels, soft):
    """members: list of {(seed,fold): rec}; one vote per fold over all members. Returns fold F1s/accs."""
    folds = sorted({f for t in members for (_, f) in t})
    f1s, accs = [], []
    for f in folds:
        recs = [r for t in members for (s, ff), r in t.items() if ff == f]
        if not recs:
            continue
        idx = np.array(recs[0]["test_idx"])
        y = labels[idx]
        if soft:
            if any(r["probs"] is None for r in recs):
                return None
            pred = vote(None, [np.array(r["probs"]) for r in recs])
        else:
            pred = vote([np.array(r["preds"]) for r in recs])
        f1s.append(f1_score(y, pred, average="macro"))
        accs.append(accuracy_score(y, pred))
    return {"fold_f1": [float(x) for x in f1s], "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else float("nan"),
            "acc_mean": float(np.mean(accs)), "n_members": len(recs)} if f1s else None


def tissue_errors(table, ctx):
    per_seed = {}
    for (s, f), r in table.items():
        idx = np.array(r["test_idx"])
        pred = np.array(r["preds"])
        for t, p, l in zip(idx, pred, ctx.labels[idx]):
            e = per_seed.setdefault(s, {}).setdefault(ctx.tissue[t], [0, 0])
            e[0] += int(p != l)
            e[1] += 1
    out = {}
    for t in TISSUES:
        rates = [per_seed[s][t][0] / per_seed[s][t][1] for s in per_seed if t in per_seed[s]]
        cnt = [per_seed[s][t][1] for s in per_seed if t in per_seed[s]]
        if rates:
            out[t] = {"error_rate": float(np.mean(rates)), "n_apples": int(cnt[0])}
    return out


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # std of 1 fold on partial data
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=R2_PATH)
    ap.add_argument("--summary-out", default=SUMMARY_PATH)
    args = ap.parse_args()

    r2 = json.load(open(args.results, encoding="utf-8"))
    ms = json.load(open(MS_PATH, encoding="utf-8"))["runs"]
    runs = r2["runs"]
    ctx = Context()
    rng = np.random.RandomState(42)
    present = [v for v in VARIANTS if any(r["variant"] == v for r in runs.values())]
    base_names = sorted({BASELINES[v] for v in present if BASELINES[v] in ("nir", "full", "visible")} | {"nir"})
    tabs = {v: rec_table(runs, v) for v in present}
    tabs_tta = {v: rec_table(runs, v, tta=True) for v in present}
    for b in base_names:
        tabs[b] = ms_table(ms, b)
    n_err = len(r2.get("errors", []))
    print(f"{len(runs)} round-2 runs in {args.results}" + (f"  ({n_err} FAILED runs under 'errors'!)" if n_err else ""))

    summary = {"variants": {}, "paired": {}, "tissue_error": {}, "ensembles": {}}
    sd = lambda x: "  n/a" if x is None else f"{x:.3f}"
    print("\n=== 1) Per variant (mean +- std across seeds of the 5-fold mean; plain | TTA) ===")
    print(f"  {'variant':12s} {'seeds':>5s} {'F1 plain':>14s} {'acc':>7s} | {'F1 TTA':>14s} {'acc':>7s} | per-class F1 plain (H / Int / Surf)")
    for v in base_names + present:
        s = summarize_variant(tabs[v])
        st = summarize_variant(tabs_tta[v]) if v in tabs_tta else None
        summary["variants"][v] = {"plain": s, "tta": st}
        if "f1_mean_of_seed_means" not in s:
            print(f"  {v:12s} incomplete ({s['n_runs']} runs)")
            continue
        pc = s["pooled_oof_per_class_f1"]
        tta_txt = (f"{st['f1_mean_of_seed_means']:.3f} +- {sd(st['f1_std_of_seed_means'])} {st['acc_mean_of_seed_means'] * 100:6.1f}%"
                   if st and "f1_mean_of_seed_means" in st else f"{'(baseline: no TTA)':>22s}")
        print(f"  {v:12s} {s['n_complete_seeds']:5d} {s['f1_mean_of_seed_means']:.3f} +- {sd(s['f1_std_of_seed_means'])} "
              f"{s['acc_mean_of_seed_means'] * 100:6.1f}% | {tta_txt} | "
              f"{pc['Healthy']:.3f} / {pc['Internal_Defect']:.3f} / {pc['Surface_Defect']:.3f}")

    print("\n=== 2) Paired comparisons vs the variant's own baseline (A - B in macro-F1; paired by seed+fold) ===")
    print("  'real' = CI excludes 0 AND >=12/15 same sign (needs all 15 pairs); fold-level = mean over seeds per fold, 5 folds")
    passed = []
    for v in present:
        b = BASELINES[v]
        if b not in tabs:
            print(f"  {v}: baseline {b} has no results yet, skipped")
            continue
        rows = [(f"{v} - {b}", tabs[v], tabs[b]), (f"{v}[TTA] - {b}", tabs_tta[v], tabs[b]),
                (f"{v}[TTA] - {v}", tabs_tta[v], tabs[v])]
        for name, A, B in rows:
            r = paired(A, B, rng)
            if r is None:
                continue
            fl = fold_level(A, B)
            summary["paired"][name] = {**r, "fold_level": fl, "verdict": verdict(r)}
            fl_txt = (f"folds {fl['same_sign_folds']}/{fl['n_folds']} same sign, fold-CI [{fl['ci95'][0]:+.3f},{fl['ci95'][1]:+.3f}]"
                      if fl else "")
            print(f"  {name:28s} mean {r['mean_diff']:+.3f}  CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  "
                  f"same sign {r['same_sign_pairs']}/{r['n_pairs']}  {fl_txt}  -> {verdict(r)}")
            if name == f"{v} - {b}" and r["complete"] and r["real_difference"] and r["mean_diff"] > 0:
                passed.append(v)
    summary["passed_real_improvement"] = passed

    print("\n=== 3) Error rate per tissue (plain predictions, mean over seeds; lower is better) ===")
    names = base_names + present
    for v in names:
        summary["tissue_error"][v] = tissue_errors(tabs[v], ctx)
    print(f"  {'tissue':17s} {'n':>4s} " + " ".join(f"{v[:11]:>11s}" for v in names))
    for t in TISSUES:
        n = next((summary["tissue_error"][v][t]["n_apples"] for v in names if t in summary["tissue_error"][v]), 0)
        print(f"  {t:17s} {n:4d} " + " ".join(
            f"{summary['tissue_error'][v][t]['error_rate'] * 100:10.1f}%" if t in summary["tissue_error"][v] else f"{'-':>11s}"
            for v in names))

    print("\n=== 4) Ensembles within a fold (EXPLORATORY: groups chosen after seeing results; confirm with new seeds) ===")
    labels = ctx.labels

    def report_ens(name, members, soft):
        e = ensemble_over_folds(members, labels, soft)
        if e is None:
            return
        summary["ensembles"][name] = e
        print(f"  {name:46s} F1 {e['f1_mean']:.3f} +- {e['f1_std']:.3f} (over folds)  acc {e['acc_mean'] * 100:.1f}%  "
              f"[{e['n_members']} members/fold]")

    for v in present + ["nir", "full", "visible"]:
        if v not in tabs:
            continue
        report_ens(f"{v}: 3-seed hard vote", [tabs[v]], soft=False)
        if v in present:
            report_ens(f"{v}: 3-seed soft vote", [tabs[v]], soft=True)
            report_ens(f"{v}[TTA]: 3-seed soft vote", [tabs_tta[v]], soft=True)
    grp_raw = [v for v in RAW_NIR_GROUP if v in tabs]
    if len(grp_raw) >= 2:
        report_ens("group raw-NIR " + "+".join(grp_raw) + ": soft", [tabs[v] for v in grp_raw], soft=True)
        report_ens("group raw-NIR + baseline nir: hard vote", [tabs[v] for v in grp_raw] + [tabs["nir"]], soft=False)
        report_ens("group raw-NIR [TTA]: soft", [tabs_tta[v] for v in grp_raw], soft=True)
    grp_ss = [v for v in SS_VARIANTS if v in tabs]
    if len(grp_ss) >= 2:
        report_ens("group SingleSpectra " + "+".join(grp_ss) + ": soft", [tabs[v] for v in grp_ss], soft=True)
    grp_pass = [v for v in passed if v in tabs]
    if len(grp_pass) >= 2:
        report_ens("group passed-real " + "+".join(grp_pass) + ": soft", [tabs[v] for v in grp_pass], soft=True)
    elif grp_pass:
        print(f"  (only one variant passed the 'real improvement' rule: {grp_pass}; no group of passed variants)")
    else:
        print("  (no variant passed the 'real improvement' rule vs its baseline)")

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump({"results_file": args.results, "n_runs": len(runs), "n_failed_runs": n_err, **summary}, f, indent=2)
    print(f"\nsaved -> {args.summary_out}")


if __name__ == "__main__":
    main()
