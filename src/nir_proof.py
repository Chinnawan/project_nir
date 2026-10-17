"""N2-N4 of "prove NIR > Visible" (see PROJECT_PLAN.md).

  python -m src.nir_proof n2   per-class / per-tissue nir vs visible (+ McNemar nir vs visible / full)
                               from multiseed_results.json (seeds 0-2, same pipeline for every variant)
  python -m src.nir_proof n3   classical ML on visible-only vs NIR-only hand-crafted features, same folds
  python -m src.nir_proof n4   late fusion nir+visible vs nir vs early fusion (full), seeds 3-5
                               (nir: confirm_results.json, visible/full: fusion_results.json) + band alignment check
Each writes outputs/dl/nir_proof_n{2,3,4}.json and prints a table.
"""
import json
import os
import sys

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from statsmodels.stats.contingency_tables import mcnemar

from .config import CLASSES, OUTPUT_DIR, PROJECT_ROOT
from .multiseed_report import paired
from .round2 import Context

DL = os.path.join(OUTPUT_DIR, "dl")
T975_DF4 = 2.776
N_BOOT = 10000
TISSUES = ["Bruise", "Flesh_damage", "Rot", "Russet", "Scald", "Hail_damage", "Hail_damage_perf", "Frost_damage",
           "Limb_rub", "Other", "Stem", "Calix", "Healthy"]


def boot_ci(x, seed=42, n=N_BOOT):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    m = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n)])
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def fold_check(diffs_by_key):
    """diffs_by_key: {(seed, fold): diff}. Mean over seeds per fold -> 5 values, t-CI and sign count."""
    folds = sorted({f for _, f in diffs_by_key})
    pf = np.array([np.mean([d for (s, ff), d in diffs_by_key.items() if ff == f]) for f in folds])
    se = pf.std(ddof=1) / np.sqrt(len(pf))
    return {"per_fold": pf.tolist(), "ci95": [float(pf.mean() - T975_DF4 * se), float(pf.mean() + T975_DF4 * se)],
            "same_sign_folds": int((np.sign(pf) == np.sign(pf.mean())).sum())}


def mcnemar_exact(y, a, b):
    ca, cb = a == y, b == y
    tab = [[int(np.sum(ca & cb)), int(np.sum(ca & ~cb))], [int(np.sum(~ca & cb)), int(np.sum(~ca & ~cb))]]
    r = mcnemar(tab, exact=(tab[0][1] + tab[1][0]) < 25)
    return {"table": tab, "only_a_correct": tab[0][1], "only_b_correct": tab[1][0], "p_value": float(r.pvalue)}


def verdict(r):
    return "REAL" if r["real_difference"] else "not larger than training noise"


def pooled_preds(runs, variant, seed, n):
    pred = np.full(n, -1)
    for r in runs.values():
        if r["variant"] == variant and r["seed"] == seed:
            pred[np.array(r["test_idx"])] = r["test_predictions"]
    assert (pred >= 0).all(), (variant, seed)
    return pred


# ----------------------------------------------------------------------------------------------- N2
def n2():
    runs = json.load(open(os.path.join(DL, "multiseed_results.json"), encoding="utf-8"))["runs"]
    ctx = Context()
    y = ctx.labels
    n = len(y)
    rng = np.random.RandomState(42)
    tab = lambda v: {(r["seed"], r["fold"]): r for r in runs.values() if r["variant"] == v}
    T = {v: tab(v) for v in ("nir", "visible", "full")}
    seeds = sorted({s for s, _ in T["nir"]})
    out = {"seeds": seeds, "overall": {}, "per_class": {}, "per_tissue": {}, "mcnemar": {}}

    print("N2  nir vs visible (and full), multiseed seeds 0-2, 15 paired (seed, fold) runs")
    for v in ("nir", "visible", "full"):
        f1 = np.mean([r["test_macro_f1"] for r in T[v].values()])
        acc = np.mean([r["test_accuracy"] for r in T[v].values()])
        out["overall"][v] = {"macro_f1": float(f1), "accuracy": float(acc)}
        print(f"  {v:8s} macro-F1 {f1:.3f}  accuracy {acc * 100:.1f}%")

    print("\n  per-class F1, paired by seed+fold (A - B); 'REAL' = CI excludes 0 and >= 12/15 same sign")
    for a, b in (("nir", "visible"), ("nir", "full")):
        for ci, c in enumerate(CLASSES):
            A = {k: {"test_macro_f1": r["per_class_f1"][ci]} for k, r in T[a].items()}
            B = {k: {"test_macro_f1": r["per_class_f1"][ci]} for k, r in T[b].items()}
            r = paired(A, B, rng)
            fc = fold_check({k: A[k]["test_macro_f1"] - B[k]["test_macro_f1"] for k in A})
            out["per_class"][f"{a}-{b}|{c}"] = {**r, "fold_level": fc, "mean_a": float(np.mean([x["test_macro_f1"] for x in A.values()])),
                                                "mean_b": float(np.mean([x["test_macro_f1"] for x in B.values()]))}
            print(f"  {a}-{b:8s} {c:16s} F1 {out['per_class'][f'{a}-{b}|{c}']['mean_a']:.3f} vs "
                  f"{out['per_class'][f'{a}-{b}|{c}']['mean_b']:.3f}  diff {r['mean_diff']:+.3f} "
                  f"CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {r['same_sign_pairs']}/15, folds {fc['same_sign_folds']}/5 -> {verdict(r)}")

    print("\n  recall per tissue (fraction of apples of that tissue classified into the correct class, mean over 3 seeds);"
          "\n  CI = bootstrap over apples of the paired per-apple difference (nir - visible)")
    correct = {v: np.mean([pooled_preds(runs, v, s, n) == y for s in seeds], axis=0) for v in ("nir", "visible", "full")}
    for t in TISSUES:
        idx = np.array([i for i in range(n) if ctx.tissue[i] == t])
        d = correct["nir"][idx] - correct["visible"][idx]
        m, ci = boot_ci(d)
        out["per_tissue"][t] = {"n": int(len(idx)), "recall_nir": float(correct["nir"][idx].mean()),
                                "recall_visible": float(correct["visible"][idx].mean()),
                                "recall_full": float(correct["full"][idx].mean()), "nir_minus_visible": m, "ci95": ci}
        flag = "  NIR better" if ci[0] > 0 else ("  visible better" if ci[1] < 0 else "")
        print(f"  {t:17s} n={len(idx):3d}  nir {correct['nir'][idx].mean() * 100:5.1f}%  visible {correct['visible'][idx].mean() * 100:5.1f}%"
              f"  full {correct['full'][idx].mean() * 100:5.1f}%   nir-vis {m * 100:+6.1f} pts [{ci[0] * 100:+.1f},{ci[1] * 100:+.1f}]{flag}")

    print("\n  McNemar on pooled out-of-fold predictions (819 apples), per seed")
    for a, b in (("nir", "visible"), ("nir", "full")):
        for s in seeds:
            r = mcnemar_exact(y, pooled_preds(runs, a, s, n), pooled_preds(runs, b, s, n))
            out["mcnemar"][f"{a}-{b}|seed{s}"] = r
            print(f"  {a} vs {b:8s} seed {s}: only {a} correct {r['only_a_correct']:3d}, only {b} correct {r['only_b_correct']:3d}"
                  f"  p = {r['p_value']:.2e}")
    json.dump(out, open(os.path.join(DL, "nir_proof_n2.json"), "w", encoding="utf-8"), indent=1)
    print("saved -> outputs/dl/nir_proof_n2.json")


# ----------------------------------------------------------------------------------------------- N3
def n3():
    from .features import build_feature_dataset, feature_names
    from .train_classical import build_models
    from sklearn.preprocessing import StandardScaler

    ctx = Context()
    y = ctx.labels
    Xs, paths = [], []
    for split in ("train", "val", "test"):
        X, _, p = build_feature_dataset(split)
        Xs.append(X)
        paths += [str(q) for q in p]
    X = np.concatenate(Xs)
    assert [os.path.normcase(a) for a in paths] == [os.path.normcase(fp) for fp, _ in ctx.pool_raw]
    names = feature_names()
    sets = {"visible": [i for i, nm in enumerate(names) if nm.split("_")[0] in ("BL", "GR") or nm == "NDI_BL_GR"],
            "nir": [i for i, nm in enumerate(names) if nm.split("_")[0] in ("RE", "IR") or nm == "NDI_RE_IR"],
            "all": list(range(len(names)))}
    out = {"feature_sets": {k: [names[i] for i in v] for k, v in sets.items()}, "models": {}}
    print(f"N3  classical ML, same 5 folds; features: visible {len(sets['visible'])}, nir {len(sets['nir'])}, all {len(sets['all'])}"
          " (cross-band ratios such as NDI_BL_RE only in 'all')")
    preds = {}
    for fs, cols in sets.items():
        for fold, (tv, te) in enumerate(ctx.splits):
            sc = StandardScaler().fit(X[tv][:, cols])
            Xtr, Xte = sc.transform(X[tv][:, cols]), sc.transform(X[te][:, cols])
            for name, model in build_models().items():
                if name == "LightGBM":
                    model.set_params(deterministic=True, force_row_wise=True)
                model.fit(Xtr, y[tv])
                p = model.predict(Xte)
                preds.setdefault((name, fs), np.full(len(y), -1))[te] = p
                out["models"].setdefault(name, {}).setdefault(fs, {"fold_f1": [], "fold_acc": []})
                out["models"][name][fs]["fold_f1"].append(float(f1_score(y[te], p, average="macro")))
                out["models"][name][fs]["fold_acc"].append(float(accuracy_score(y[te], p)))
    print(f"  {'model':13s} {'features':8s} {'macro-F1 (5 folds)':>20s} {'accuracy':>10s}")
    for name, d in out["models"].items():
        for fs in ("visible", "nir", "all"):
            f, a = np.array(d[fs]["fold_f1"]), np.array(d[fs]["fold_acc"])
            d[fs].update({"f1_mean": float(f.mean()), "f1_std": float(f.std(ddof=1)), "acc_mean": float(a.mean())})
            print(f"  {name:13s} {fs:8s} {f.mean():.3f} +- {f.std(ddof=1):.3f}      {a.mean() * 100:6.1f}%")
        df = np.array(d["nir"]["fold_f1"]) - np.array(d["visible"]["fold_f1"])
        se = df.std(ddof=1) / np.sqrt(5)
        mc = mcnemar_exact(y, preds[(name, "nir")], preds[(name, "visible")])
        d["nir_minus_visible"] = {"fold_diffs": df.tolist(), "mean": float(df.mean()),
                                  "ci95": [float(df.mean() - T975_DF4 * se), float(df.mean() + T975_DF4 * se)],
                                  "folds_nir_better": int((df > 0).sum()), "mcnemar": mc}
        print(f"  {name:13s} nir - visible: {df.mean():+.3f} (fold t-CI [{df.mean() - T975_DF4 * se:+.3f},{df.mean() + T975_DF4 * se:+.3f}],"
              f" {int((df > 0).sum())}/5 folds), McNemar only-nir {mc['only_a_correct']} vs only-visible {mc['only_b_correct']}, p = {mc['p_value']:.2e}")
    json.dump(out, open(os.path.join(DL, "nir_proof_n3.json"), "w", encoding="utf-8"), indent=1)
    print("saved -> outputs/dl/nir_proof_n3.json")


# ----------------------------------------------------------------------------------------------- N4
def band_alignment(n_images=50, seed=42):
    """Phase correlation of gradient-magnitude images between bands of the same apple (raw RawA quadrants, the
    images the early-fusion `full` model is built from; and the geometrically corrected SingleSpectra images)."""
    from .preprocess_320 import crop_and_resize
    from .preprocess_ss import read_bands
    mapping = json.load(open(os.path.join(OUTPUT_DIR, "ss_mapping.json")))
    rng = np.random.RandomState(seed)
    ids = sorted(rng.choice(range(1, 820), n_images, replace=False))
    win = None

    def grad(img):
        g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2)
        return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

    res = {"raw": {}, "ss": {}}
    pairs = [("BL", "RE"), ("BL", "IR"), ("GR", "RE"), ("GR", "IR"), ("BL", "GR"), ("RE", "IR")]
    for rawno in ids:
        img = cv2.imread(os.path.join(PROJECT_ROOT, "images", "RawImages", f"RawA{rawno:04d}.tif"), cv2.IMREAD_GRAYSCALE)
        raw = {"IR": img[0:512, 0:640], "GR": img[0:512, 640:1280], "BL": img[512:1024, 0:640], "RE": img[512:1024, 640:1280]}
        raw = {k: crop_and_resize(v, 512) for k, v in raw.items()}  # 512x512, scale factor to 224: 224/512
        ss = dict(zip(["BL", "GR", "RE", "IR"], read_bands(mapping[str(rawno)]["ss_name"])))
        for src, bands, scale in (("raw", raw, 224 / 512), ("ss", ss, 224 / 560)):
            h, w = bands["BL"].shape
            win = cv2.createHanningWindow((w, h), cv2.CV_32F)
            for a, b in pairs:
                (dx, dy), resp = cv2.phaseCorrelate(grad(bands[a]), grad(bands[b]), win)
                res[src].setdefault(f"{a}-{b}", []).append([dx * scale, dy * scale, resp])
    summary = {}
    for src in res:
        for pr, v in res[src].items():
            v = np.array(v)
            mag = np.hypot(v[:, 0], v[:, 1])
            summary[f"{src}|{pr}"] = {"mean_shift_px224": float(mag.mean()), "median_shift_px224": float(np.median(mag)),
                                     "max_shift_px224": float(mag.max()), "mean_dx": float(v[:, 0].mean()),
                                     "mean_dy": float(v[:, 1].mean()), "mean_response": float(v[:, 2].mean())}
    return ids, summary


def n4():
    rng = np.random.RandomState(42)
    ctx = Context()
    y = ctx.labels
    conf = json.load(open(os.path.join(DL, "confirm_results.json"), encoding="utf-8"))["runs"]
    fus_path = os.path.join(DL, "fusion_results.json")
    fus = json.load(open(fus_path, encoding="utf-8"))["runs"]
    get = lambda runs, v: {(r["seed"], r["fold"]): r for r in runs.values() if r["variant"] == v}
    T = {"nir": get(conf, "nir"), "visible": get(fus, "visible"), "full": get(fus, "full")}
    keys = sorted(set(T["nir"]) & set(T["visible"]) & set(T["full"]))
    print(f"N4  late vs early fusion, seeds 3-5: {len(keys)} paired (seed, fold) runs with nir + visible + full")
    out = {"n_pairs": len(keys), "weights": {}, "paired": {}, "complementarity": {}}

    def rec(variant_probs, k):
        idx = np.array(T["nir"][k]["test_idx"])
        p = variant_probs
        pred = p.argmax(1)
        return {"test_macro_f1": float(f1_score(y[idx], pred, average="macro")), "test_accuracy": float(accuracy_score(y[idx], pred)),
                "pred": pred, "idx": idx}

    tables = {}
    for k in keys:
        for v in ("nir", "visible", "full"):
            assert T[v][k]["test_idx"] == T["nir"][k]["test_idx"]
    for tag in ("", "tta_"):
        for v in ("nir", "visible", "full"):
            tables[f"{v}{'[TTA]' if tag else ''}"] = {k: rec(np.array(T[v][k][tag + "test_probs"]), k) for k in keys}
        for w in (0.1, 0.2, 0.3, 0.4, 0.5):
            tables[f"late w={w}{'[TTA]' if tag else ''}"] = {
                k: rec((1 - w) * np.array(T["nir"][k][tag + "test_probs"]) + w * np.array(T["visible"][k][tag + "test_probs"]), k)
                for k in keys}
    print(f"  {'model':18s} {'macro-F1':>9s} {'accuracy':>9s}   (mean over {len(keys)} runs)")
    for name, t in tables.items():
        f1 = np.mean([r["test_macro_f1"] for r in t.values()])
        acc = np.mean([r["test_accuracy"] for r in t.values()])
        out["weights"][name] = {"macro_f1": float(f1), "accuracy": float(acc)}
        if "[TTA]" not in name or name.startswith("late w=0.5") or name in ("nir[TTA]", "visible[TTA]", "full[TTA]"):
            print(f"  {name:18s} {f1:9.3f} {acc * 100:8.1f}%")
    print("\n  paired comparisons (primary: equal-weight late fusion w=0.5, plain probabilities)")
    for a, b in (("late w=0.5", "nir"), ("late w=0.5", "full"), ("nir", "full"), ("nir", "visible"),
                 ("late w=0.5[TTA]", "nir[TTA]"), ("full", "visible")):
        r = paired(tables[a], tables[b], rng)
        fc = fold_check({k: tables[a][k]["test_macro_f1"] - tables[b][k]["test_macro_f1"] for k in keys})
        out["paired"][f"{a} - {b}"] = {**r, "fold_level": fc, "verdict": verdict(r)}
        print(f"  {a:16s} - {b:9s} {r['mean_diff']:+.3f} CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] "
              f"{r['same_sign_pairs']}/{r['n_pairs']}, folds {fc['same_sign_folds']}/5 -> {verdict(r)}")
    nir_ok = np.concatenate([tables["nir"][k]["pred"] == y[tables["nir"][k]["idx"]] for k in keys])
    vis_ok = np.concatenate([tables["visible"][k]["pred"] == y[tables["visible"][k]["idx"]] for k in keys])
    out["complementarity"] = {"nir_wrong_visible_right": float(np.mean(~nir_ok & vis_ok)),
                              "nir_right_visible_wrong": float(np.mean(nir_ok & ~vis_ok)),
                              "both_wrong": float(np.mean(~nir_ok & ~vis_ok)), "n_predictions": int(len(nir_ok))}
    c = out["complementarity"]
    print(f"\n  complementarity over {c['n_predictions']} predictions: nir wrong & visible right {c['nir_wrong_visible_right'] * 100:.1f}%,"
          f" nir right & visible wrong {c['nir_right_visible_wrong'] * 100:.1f}%, both wrong {c['both_wrong'] * 100:.1f}%")
    print("\n  band alignment (phase correlation of gradient images, 50 random apples, shift in pixels at 224x224):")
    ids, align = band_alignment()
    out["alignment"] = {"images": [int(i) for i in ids], "summary": align}
    for k, v in align.items():
        print(f"  {k:12s} mean {v['mean_shift_px224']:.2f} px, median {v['median_shift_px224']:.2f}, max {v['max_shift_px224']:.2f}"
              f"  (mean dx {v['mean_dx']:+.2f}, dy {v['mean_dy']:+.2f}; response {v['mean_response']:.2f})")
    json.dump(out, open(os.path.join(DL, "nir_proof_n4.json"), "w", encoding="utf-8"), indent=1)
    print("saved -> outputs/dl/nir_proof_n4.json")


if __name__ == "__main__":
    {"n2": n2, "n3": n3, "n4": n4}[sys.argv[1]]()
