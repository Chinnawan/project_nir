"""K: replicate Kleynen et al. (2005) and compare in 2 groups (sound vs defective).

The 3-class nir_ss result stays the main result of the thesis; everything here is an additional 2-group view.

  python -m src.kleynen k1   severity labels (segmentations/KleynenDefectClasses_Simplified.txt)
                             -> outputs/severity_labels.csv (local only) + outputs/dl/k1_severity.json
  python -m src.kleynen k2   CNN with merged classes: Internal + Surface -> Defect, from the recorded out-of-fold
                             probabilities (3-seed vote + TTA, defective if P(I) + P(S) > 0.5)  -> outputs/dl/k2_merged.json
  python -m src.kleynen k3   Kleynen 2005 on the SingleSpectra images (CPU): background by thresholding 800 nm + fill holes
                             + closing, Bayes pixel classification on multi-dimensional histograms (equal priors), per-fruit
                             features, LDA sound/defective; 3 band sets; same 5 folds  -> outputs/dl/k3_kleynen.json
                             (+ 20 example masks in outputs/kleynen_masks/, gitignored: dataset pixels)
  python -m src.kleynen k3 15 sensitivity run with a 15 px erosion of the fruit border -> outputs/dl/k3_kleynen_erode15.json
  python -m src.kleynen k3_features_only  -> outputs/dl/k3_kleynen_features_only.json
                             Paper-described fruit features only; other approximations remain unchanged.
  python -m src.kleynen k5   summary table (paper | replication | CNN merged | CNN 2-class) + McNemar
                             -> outputs/dl/k5_summary.json   (K4 = src/run_k4.py -> outputs/dl/k4_results.json)

Kleynen 2005 (J. Food Eng. 69:41-49), the parts replicated here:
  * fruit/background: threshold of the 800 nm band, then erosion to drop the dark fruit border
  * pixel classes: P(sound|x) = P(x|sound) / (P(x|sound) + P(x|defect)) with P(sound) = P(defect) = 0.5, the two
    likelihoods are relative frequency distributions (histograms) of 450/750/800 nm; sound pixels from sound fruits,
    defect pixels from manually selected defect tissue (here: the SBW_Defect masks); pixel is sound if P > 0.5
  * fruit features: mean, median, std of each band on the sound and on the defective tissue + defect area ratio -> LDA
  * paper: 280 sound + 246 defective fruits (stem-end/calyx handled by a separate template matcher, not replicated)
    sound 94.3 %, defective 84.6 % (Table 2: rejected 100, recent bruises 98.2, serious 94.5, slight 55)
"""
import csv
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import OUTPUT_DIR, PROJECT_ROOT
from .consent import require_consent

DL = os.path.join(OUTPUT_DIR, "dl")
SEVERITY_TXT = os.path.join(PROJECT_ROOT, "segmentations", "KleynenDefectClasses_Simplified.txt")
SEVERITY_CSV = os.path.join(OUTPUT_DIR, "severity_labels.csv")
MASK_DIR = os.path.join(PROJECT_ROOT, "segmentations", "Segmented_defects", "jona", "im", "segmentedBW")
EXAMPLE_DIR = os.path.join(OUTPUT_DIR, "kleynen_masks")
SEV_NAMES = {1: "reject", 2: "bruise", 3: "serious", 4: "slight"}
SEV_ORDER = ["reject", "bruise", "serious", "slight"]
AGARLA_COUNTS = [55, 56, 76, 59]  # Agarla et al. 2023 (Sensors 23:7893), as given in the request
PAPER = {"sound": 94.3, "defective": 84.6, "reject": 100.0, "bruise": 98.2, "serious": 94.5, "slight": 55.0,
         "defective_450_750_only": 80.1, "n_sound": 280, "n_defective": 246, "source": "Kleynen et al. 2005, Sect. 3 + Table 2"}
TISSUES = ["Bruise", "Flesh_damage", "Rot", "Russet", "Scald", "Hail_damage", "Hail_damage_perf", "Frost_damage",
           "Limb_rub", "Other", "Stem", "Calix", "Healthy"]
BANDS = ["BL", "GR", "RE", "IR"]  # 450, 500, 750, 800 nm
BAND_SETS = {"450/750/800 (paper)": [0, 2, 3], "750/800 (NIR)": [2, 3], "450/500 (Visible)": [0, 1]}
LEVELS = 64           # histogram bins per band (256 grey levels / 4): 64^3 = 262k cells for 3 bands
ERODE_PX = 5          # erosion of the fruit mask before using its pixels (paper: erosion, size not given)
CLOSE_PX = 15         # closing of the thresholded 800 nm mask (fills dark defects / stem-calyx at the border)
MIN_REGION_PX = 10    # connected defect regions smaller than this are not counted as regions


# ----------------------------------------------------------------------------------------------- shared
def context():
    from .round2 import Context
    ctx = Context()
    names = [os.path.basename(fp).replace(".npy", "") for fp, _ in ctx.pool_raw]
    fold = np.zeros(len(names), int)
    for f, (_, te) in enumerate(ctx.splits):
        fold[te] = f
    return ctx, names, fold


def load_severity():
    require_consent()
    rows = [l.split() for l in open(SEVERITY_TXT, encoding="latin-1") if l.startswith("Defect0")]
    return {int(r[0][len("Defect"):]): SEV_NAMES[int(r[1])] for r in rows}


def severity_per_image(ctx):
    sev = load_severity()
    return np.array([sev[d] if d is not None else "" for d in ctx.defect_no])


def group_report(pred_def, ctx, sev, mask=None):
    """pred_def: (819,) 1 = predicted defective. Correct rate per group / tissue / severity + 2x2 confusion.
    mask: optional subset of apples to include (e.g. the paper's 280 sound + 246 defective)."""
    y3 = np.asarray(ctx.labels)
    tissue = np.asarray(ctx.tissue)
    keep = np.ones(len(y3), bool) if mask is None else mask
    yb = (y3 != 0).astype(int)
    ok = (pred_def == yb)

    def rate(m):
        m = m & keep
        return {"n": int(m.sum()), "correct": int(ok[m].sum()), "pct": float(ok[m].mean() * 100) if m.any() else None}
    sound = tissue == "Healthy"
    sc = np.isin(tissue, ["Stem", "Calix"])
    defect = yb == 1
    out = {"groups": {"healthy (sound fruit)": rate(sound), "stem+calyx": rate(sc), "stem": rate(tissue == "Stem"),
                      "calyx": rate(tissue == "Calix"), "defective": rate(defect)},
           "per_tissue": {t: rate(tissue == t) for t in TISSUES if (tissue == t).any()},
           "per_severity": {s: rate(sev == s) for s in SEV_ORDER}}
    k = keep
    out["confusion_2x2"] = {"rows": "true sound(all healthy-class), true defective", "cols": "pred sound, pred defective",
                            "matrix": [[int(np.sum(k & (yb == 0) & (pred_def == 0))), int(np.sum(k & (yb == 0) & (pred_def == 1)))],
                                       [int(np.sum(k & (yb == 1) & (pred_def == 0))), int(np.sum(k & (yb == 1) & (pred_def == 1)))]]}
    out["accuracy"] = float(ok[k].mean())
    out["f1_defective"] = float(f1_score(yb[k], pred_def[k]))
    paper = (sound | defect) & keep
    out["paper_subset_280_246"] = {"accuracy": float(ok[paper].mean()), "sound_pct": rate(sound)["pct"],
                                   "defective_pct": rate(defect)["pct"],
                                   **{s + "_pct": rate(sev == s)["pct"] for s in SEV_ORDER}}
    return out


def print_report(name, r):
    g, p = r["groups"], r["paper_subset_280_246"]
    cm = r["confusion_2x2"]["matrix"]
    print(f"  {name:34s} healthy {g['healthy (sound fruit)']['pct']:5.1f}%  stem+calyx {g['stem+calyx']['pct'] if g['stem+calyx']['pct'] is not None else float('nan'):5.1f}%"
          f"  defective {g['defective']['pct']:5.1f}% | reject {p['reject_pct']:5.1f} bruise {p['bruise_pct']:5.1f} serious {p['serious_pct']:5.1f}"
          f" slight {p['slight_pct']:5.1f} | acc(280+246) {p['accuracy'] * 100:5.1f}%  acc(all) {r['accuracy'] * 100:5.1f}%  2x2 {cm}")


def vote_probs(runs, n, key="tta_test_probs"):
    p, cnt = None, np.zeros(n)
    for r in runs.values():
        q = np.array(r[key])
        if p is None:
            p = np.zeros((n, q.shape[1]))
        idx = np.array(r["test_idx"])
        p[idx] += q
        cnt[idx] += 1
    assert (cnt == 3).all(), "expected 3 seeds per apple"
    return p / 3


def load_runs(fname, variant):
    p = os.path.join(DL, fname)
    if not os.path.exists(p):
        return None
    runs = {(r["seed"], r["fold"]): r for r in json.load(open(p, encoding="utf-8"))["runs"].values() if r["variant"] == variant}
    return runs if len(runs) == 15 else None


def merged_defect_pred(runs, n):
    """3-class model -> 2 groups: defective if P(Internal) + P(Surface) > 0.5 (3-seed vote + TTA)."""
    return (vote_probs(runs, n)[:, 1:].sum(1) > 0.5).astype(int)


# ----------------------------------------------------------------------------------------------- K1
def k1():
    ctx, names, _ = context()
    sev = load_severity()
    counts = Counter(sev.values())
    assert len(sev) == 246 and set(sev) == {d for d in ctx.defect_no if d is not None}
    rows = [(names[i], ctx.defect_no[i], ctx.tissue[i], ctx.labels[i], sev[ctx.defect_no[i]])
            for i in range(len(names)) if ctx.defect_no[i] is not None]
    with open(SEVERITY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "defect_no", "tissue", "class3", "severity"])
        w.writerows(sorted(rows))
    cross = {t: {s: sum(1 for r in rows if r[2] == t and r[4] == s) for s in SEV_ORDER}
             for t in TISSUES if any(r[2] == t for r in rows)}
    ours = [counts["serious"], counts["bruise"], counts["slight"], counts["reject"]]
    out = {"found": True, "source": "segmentations/KleynenDefectClasses_Simplified.txt (Class 1 reject, 2 bruises, 3 serious, 4 slight)",
           "counts": {s: counts[s] for s in SEV_ORDER}, "n": len(sev),
           "agarla_2023_counts_as_given": AGARLA_COUNTS,
           "ours_in_order_serious_bruise_slight_reject": ours, "matches_agarla": sorted(ours) == sorted(AGARLA_COUNTS),
           "matches_in_given_order_if_order_is_serious_bruise_slight_reject": ours == AGARLA_COUNTS,
           "kleynen_2005_table2_check": {"bruise 98.2% = 55/56": round(55 / 56 * 100, 1), "serious 94.5% = 52/55": round(52 / 55 * 100, 1),
                                         "slight 55% = 42/76": round(42 / 76 * 100, 1), "global 84.6% = 208/246": round(208 / 246 * 100, 1)},
           "tissue_x_severity": cross, "csv": "outputs/severity_labels.csv (local only, not committed)"}
    json.dump(out, open(os.path.join(DL, "k1_severity.json"), "w", encoding="utf-8"), indent=1)
    print("K1 severity labels found:", out["counts"], "| ours (serious, bruise, slight, reject) =", ours,
          "| Agarla 2023 =", AGARLA_COUNTS, "| match:", out["matches_agarla"])
    print("  Kleynen 2005 Table 2 is consistent with these counts:", out["kleynen_2005_table2_check"])
    print(f"  {'tissue':17s} " + " ".join(f"{s:>8s}" for s in SEV_ORDER))
    for t, c in cross.items():
        print(f"  {t:17s} " + " ".join(f"{c[s]:8d}" for s in SEV_ORDER))
    print("saved -> outputs/dl/k1_severity.json, outputs/severity_labels.csv")


# ----------------------------------------------------------------------------------------------- K2
def k2():
    ctx, names, _ = context()
    sev = severity_per_image(ctx)
    n = len(names)
    models = {"nir_ss (NIR, SS)": ("confirm_results.json", "nir_ss"), "visible_ss (Visible, SS)": ("n5_results.json", "visible_ss"),
              "nir (NIR, raw)": ("confirm_results.json", "nir"), "visible (Visible, raw)": ("fusion_results.json", "visible")}
    out = {"rule": "3-class model, 3-seed soft vote + TTA (seeds 3-5), pooled out-of-fold; defective if P(Internal) + P(Surface) > 0.5",
           "models": {}, "predictions": {}}
    print("K2  CNN with merged classes (Internal + Surface -> Defect), out-of-fold, 819 apples")
    for name, (f, v) in models.items():
        runs = load_runs(f, v)
        pred = merged_defect_pred(runs, n)
        out["models"][name] = group_report(pred, ctx, sev)
        out["predictions"][name] = pred.tolist()
        print_report(name, out["models"][name])
    json.dump(out, open(os.path.join(DL, "k2_merged.json"), "w", encoding="utf-8"), indent=1)
    print("saved -> outputs/dl/k2_merged.json")


# ----------------------------------------------------------------------------------------------- K3
def fruit_mask(ir):
    """Fruit/background: Otsu threshold of the 800 nm band, fill holes, closing, largest component."""
    _, m = cv2.threshold(ir, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = binary_fill_holes(m).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_PX, CLOSE_PX)))
    m = binary_fill_holes(m).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        m = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m


def load_cache(ctx, names):
    require_consent()
    from .preprocess_ss import read_bands
    mapping = json.load(open(os.path.join(OUTPUT_DIR, "ss_mapping.json")))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    cache = []
    for i, name in enumerate(names):
        bands = np.stack(read_bands(mapping[str(int(name[4:]))]["ss_name"]), axis=-1)  # 430x560x4 uint8
        fm = fruit_mask(bands[..., 3])
        er = cv2.erode(fm, ker)
        idx = np.flatnonzero(er)
        rec = {"idx": idx, "vals": bands.reshape(-1, 4)[idx], "shape": fm.shape, "fruit_area": int(fm.sum()), "defect": None}
        if ctx.defect_no[i] is not None:
            m = cv2.imread(os.path.join(MASK_DIR, f"SBW_Defect{ctx.defect_no[i]:04d}.bmp"), cv2.IMREAD_GRAYSCALE) > 127
            assert m.shape == fm.shape
            rec["defect"] = m.reshape(-1)[idx]
            rec["mask_px_outside_eroded_fruit"] = int(m.sum() - rec["defect"].sum())
        cache.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1}/{len(names)}", flush=True)
    return cache


def codes(vals, bands):
    c = np.zeros(len(vals), np.int64)
    for b in bands:
        c = c * LEVELS + (vals[:, b].astype(np.int64) * LEVELS // 256)
    return c


def per_image_hist(cache, ctx, bands):
    """sparse (codes, counts) per image: sound pixels of sound fruits (tissue Healthy), defect pixels of defect fruits"""
    sound, defect = {}, {}
    for i, rec in enumerate(cache):
        c = codes(rec["vals"], bands)
        if ctx.tissue[i] == "Healthy":
            sound[i] = np.unique(c, return_counts=True)
        elif rec["defect"] is not None and rec["defect"].any():
            defect[i] = np.unique(c[rec["defect"]], return_counts=True)
    return sound, defect


def lut_from(images, sound, defect, n_cells):
    hs, hd = np.zeros(n_cells), np.zeros(n_cells)
    for i in images:
        if i in sound:
            hs[sound[i][0]] += sound[i][1]
        if i in defect:
            hd[defect[i][0]] += defect[i][1]
    ps, pd = hs / hs.sum(), hd / hd.sum()
    den = ps + pd
    # equal priors: P(sound|x) = P(x|sound) / (P(x|sound) + P(x|defect)); colours never seen in training -> 0 -> defect
    # (the paper's rule: sound only if P(sound|x) > 0.5)
    return np.where(den > 0, ps / np.where(den > 0, den, 1), 0.0)


def fruit_features(rec, bands, lut, include_region_features=True):
    c = codes(rec["vals"], bands)
    d = lut[c] <= 0.5
    feats = []
    sound_sel = rec["vals"][~d] if (~d).any() else rec["vals"]  # whole fruit segmented as defect: sound stats = fruit stats
    defect_sel = rec["vals"][d] if d.any() else sound_sel        # no defect pixels: defect stats = sound stats
    for sel in (sound_sel, defect_sel):
        for b in bands:
            v = sel[:, b].astype(np.float64)
            feats += [v.mean(), np.median(v), v.std()]
    ratio = d.mean()
    feats.append(ratio)
    if include_region_features:
        full = np.zeros(rec["shape"][0] * rec["shape"][1], np.uint8)
        full[rec["idx"][d]] = 1
        n, _, stats, _ = cv2.connectedComponentsWithStats(full.reshape(rec["shape"]), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
        big = areas[areas >= MIN_REGION_PX]
        feats += [len(big), (areas.max() if len(areas) else 0) / len(rec["idx"])]
    return np.array(feats), d


def k3(erode=None, features_only=False):
    """erode: optional erosion size for a sensitivity run (primary run = ERODE_PX, fixed before seeing any result);
    writes k3_kleynen_erode{N}.json and no example images"""
    global ERODE_PX
    tag = "_features_only" if features_only else ""
    if erode is not None:
        if features_only:
            raise ValueError("features_only and erosion sensitivity must be separate experiments")
        ERODE_PX, tag = int(erode), f"_erode{int(erode)}"
    ctx, names, fold = context()
    sev = severity_per_image(ctx)
    y3 = np.asarray(ctx.labels)
    yb = (y3 != 0).astype(int)
    tissue = np.asarray(ctx.tissue)
    n = len(names)
    print("K3  Kleynen 2005 replication on SingleSpectra images (CPU) -- loading 819 images ...")
    cache = load_cache(ctx, names)
    out_px = [r["mask_px_outside_eroded_fruit"] for r in cache if r["defect"] is not None]
    out = {"method": {"fruit_mask": f"Otsu threshold of 800 nm + fill holes + closing {CLOSE_PX}px + largest component; erosion {ERODE_PX}px",
                      "pixel_classifier": f"Bayes, equal priors, histograms {LEVELS} bins/band; sound pixels = eroded fruit of Healthy-tissue "
                                          "fruits, defect pixels = SBW_Defect masks inside the eroded fruit; train folds only",
                      "features": ("mean/median/std of each band on sound and on defect tissue + defect ratio"
                                   if features_only else "mean/median/std of each band on sound and on defect tissue + defect ratio "
                                   f"+ n regions (>= {MIN_REGION_PX}px) + largest region / fruit area"),
                      "feature_variant": "paper_described_only" if features_only else "enriched_original",
                      "paper_faithful": False,
                      "remaining_differences": "64-bin histogram, Otsu threshold, 5px erosion, whole SBW masks, "
                                               "cross-fitted evaluation; stem/calyx template matcher not replicated",
                      "classifier": "StandardScaler + LDA (priors = training class frequencies)",
                      "cross_fitting": "LDA training features of a train-fold fruit come from histograms that exclude the outer test "
                                       "fold AND the fruit's own fold (out-of-sample, like the test fruits)",
                      "folds": "StratifiedKFold(5, shuffle, seed 42) = fold_assignment.json; deterministic, no seed dependence",
                      "settings": {"A_paper": "LDA trained on sound (Healthy) + defective fruits only (paper: 280 + 246; stem/calyx "
                                              "recognised by a separate template matcher that is NOT replicated); applied to all test fruits",
                                   "B_all": "LDA trained on all fruits, stem/calyx counted as sound (same task as the CNN, no stem/calyx detector)"}},
           "mask_pixels_outside_eroded_fruit": {"total": int(np.sum(out_px)), "images_with_any": int(np.sum(np.array(out_px) > 0))},
           "band_sets": {}, "predictions": {}}
    rng = np.random.RandomState(0)
    ex = sorted(rng.choice(np.where(tissue == "Healthy")[0], 6, replace=False).tolist() +
                rng.choice(np.where(tissue == "Stem")[0], 4, replace=False).tolist() +
                rng.choice(np.where(tissue == "Calix")[0], 4, replace=False).tolist() +
                rng.choice(np.where(yb == 1)[0], 6, replace=False).tolist())
    ex_seg = {}
    for bs_name, bands in BAND_SETS.items():
        n_cells = LEVELS ** len(bands)
        sound, defect = per_image_hist(cache, ctx, bands)
        feats_test = np.zeros((n, 3 * 2 * len(bands) + (1 if features_only else 3)))
        pred = {"A_paper": np.full(n, -1), "B_all": np.full(n, -1)}
        post = {"A_paper": np.zeros(n), "B_all": np.zeros(n)}
        px = {"defect_px_recall": [], "sound_px_flagged_healthy_fruit": [], "sound_px_flagged_stem_calyx": []}
        for f in range(5):
            tr = np.where(fold != f)[0]
            te = np.where(fold == f)[0]
            lut_te = lut_from(tr, sound, defect, n_cells)
            luts_in = {g: lut_from(tr[fold[tr] != g], sound, defect, n_cells) for g in range(5) if g != f}
            Xtr = np.array([fruit_features(cache[i], bands, luts_in[fold[i]],
                                           include_region_features=not features_only)[0] for i in tr])
            for i in te:
                feats_test[i], d = fruit_features(cache[i], bands, lut_te,
                                                 include_region_features=not features_only)
                if cache[i]["defect"] is not None:
                    px["defect_px_recall"].append(float(d[cache[i]["defect"]].mean()) if cache[i]["defect"].any() else np.nan)
                elif tissue[i] == "Healthy":
                    px["sound_px_flagged_healthy_fruit"].append(float(d.mean()))
                else:
                    px["sound_px_flagged_stem_calyx"].append(float(d.mean()))
                if i in ex and bs_name.startswith("450/750/800"):
                    ex_seg[i] = d
            for setting, use in (("A_paper", ~np.isin(tissue[tr], ["Stem", "Calix"])), ("B_all", np.ones(len(tr), bool))):
                lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
                lda.fit(Xtr[use], yb[tr][use])
                pred[setting][te] = lda.predict(feats_test[te])
                post[setting][te] = lda.predict_proba(feats_test[te])[:, 1]
        res = {"pixel_level_test": {k: float(np.nanmean(v)) for k, v in px.items()}}
        print(f"\n  bands {bs_name}: pixel level (test folds) defect-pixel recall {res['pixel_level_test']['defect_px_recall'] * 100:.1f}%, "
              f"sound pixels flagged: healthy fruit {res['pixel_level_test']['sound_px_flagged_healthy_fruit'] * 100:.1f}%, "
              f"stem/calyx fruit {res['pixel_level_test']['sound_px_flagged_stem_calyx'] * 100:.1f}%")
        for setting in pred:
            assert (pred[setting] >= 0).all()
            res[setting] = group_report(pred[setting], ctx, sev)
            if setting == "A_paper":
                res[setting]["paper_subset_only"] = group_report(pred[setting], ctx, sev, mask=~np.isin(tissue, ["Stem", "Calix"]))
            out["predictions"][f"{bs_name}|{setting}"] = pred[setting].tolist()
            out["predictions"][f"{bs_name}|{setting}|p_defect"] = np.round(post[setting], 5).tolist()
            print_report(f"{setting}", res[setting])
        out["band_sets"][bs_name] = res
    if not tag:
        save_examples(cache, names, ctx, ex, ex_seg)
    out["erode_px"] = ERODE_PX
    json.dump(out, open(os.path.join(DL, f"k3_kleynen{tag}.json"), "w", encoding="utf-8"), indent=1)
    print(f"saved -> outputs/dl/k3_kleynen{tag}.json")


def save_examples(cache, names, ctx, ex, ex_seg):
    """20 example fruits: binary fruit mask PNG each + one contact sheet (800 nm band, fruit outline green,
    eroded outline yellow, segmented defect pixels red, manual defect mask outline cyan). Dataset pixels -> gitignored."""
    from .preprocess_ss import read_bands
    mapping = json.load(open(os.path.join(OUTPUT_DIR, "ss_mapping.json")))
    os.makedirs(EXAMPLE_DIR, exist_ok=True)
    tiles = []
    for i in ex:
        name = names[i]
        bands = np.stack(read_bands(mapping[str(int(name[4:]))]["ss_name"]), axis=-1)
        fm = fruit_mask(bands[..., 3])
        cv2.imwrite(os.path.join(EXAMPLE_DIR, f"{name}_{ctx.tissue[i]}_fruitmask.png"), fm * 255)
        vis = cv2.cvtColor(bands[..., 3], cv2.COLOR_GRAY2BGR)
        rec = cache[i]
        seg = np.zeros(fm.size, np.uint8)
        seg[rec["idx"][ex_seg[i]]] = 1
        vis[seg.reshape(fm.shape) == 1] = (0, 0, 255)
        for m, col in ((fm, (0, 255, 0)), (cv2.erode(fm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1,) * 2)), (0, 255, 255))):
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, col, 1)
        if ctx.defect_no[i] is not None:
            dm = (cv2.imread(os.path.join(MASK_DIR, f"SBW_Defect{ctx.defect_no[i]:04d}.bmp"), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            cnts, _ = cv2.findContours(dm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, (255, 255, 0), 1)
        vis = cv2.resize(vis, (280, 215))
        cv2.putText(vis, f"{name} {ctx.tissue[i]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(vis)
    rows = [np.hstack(tiles[r * 5:(r + 1) * 5]) for r in range(4)]
    cv2.imwrite(os.path.join(EXAMPLE_DIR, "contact_sheet.png"), np.vstack(rows))
    print(f"  20 example masks -> {EXAMPLE_DIR} (gitignored)")


# ----------------------------------------------------------------------------------------------- K5
def k5():
    from .multiseed_report import paired
    from .nir_proof import mcnemar_exact, verdict
    ctx, names, fold = context()
    sev = severity_per_image(ctx)
    n = len(names)
    tissue = np.asarray(ctx.tissue)
    yb = (np.asarray(ctx.labels) != 0).astype(int)
    paper_set = ~np.isin(tissue, ["Stem", "Calix"])
    k2r = json.load(open(os.path.join(DL, "k2_merged.json"), encoding="utf-8"))
    k3r = json.load(open(os.path.join(DL, "k3_kleynen.json"), encoding="utf-8"))
    preds = {f"Kleynen replicated {bs} [A: paper protocol]": np.array(k3r["predictions"][f"{bs}|A_paper"]) for bs in BAND_SETS}
    preds |= {f"Kleynen replicated {bs} [B: stem/calyx = sound]": np.array(k3r["predictions"][f"{bs}|B_all"]) for bs in BAND_SETS}
    sens = os.path.join(DL, "k3_kleynen_erode15.json")
    if os.path.exists(sens):  # sensitivity run (erosion 15 px), exploratory
        k3s = json.load(open(sens, encoding="utf-8"))
        preds["Kleynen replicated 450/750/800 (paper) [A, erosion 15px: sensitivity]"] = np.array(k3s["predictions"]["450/750/800 (paper)|A_paper"])
    preds |= {"CNN merged NIR (nir_ss)": np.array(k2r["predictions"]["nir_ss (NIR, SS)"]),
              "CNN merged Visible (visible_ss)": np.array(k2r["predictions"]["visible_ss (Visible, SS)"])}
    k4 = {v: load_runs("k4_results.json", v) for v in ("nir_ss_2cls", "visible_ss_2cls")}
    if all(k4.values()):
        preds["CNN 2-class NIR (nir_ss_2cls)"] = (vote_probs(k4["nir_ss_2cls"], n)[:, 1] > 0.5).astype(int)
        preds["CNN 2-class Visible (visible_ss_2cls)"] = (vote_probs(k4["visible_ss_2cls"], n)[:, 1] > 0.5).astype(int)
    out = {"paper": PAPER, "rows": {}, "mcnemar": {}, "paired_runs": {}}
    print("K5  summary: sound / defective correct rates (paper format: 280 sound + 246 defective) and on all 819 apples")
    print(f"  {'method':58s} {'sound':>6s} {'defect':>7s} {'reject':>7s} {'bruise':>7s} {'serious':>8s} {'slight':>7s}"
          f" {'acc 526':>8s} {'stem/cal':>9s} {'acc 819':>8s}")
    print(f"  {'Kleynen 2005 (paper numbers)':58s} {PAPER['sound']:6.1f} {PAPER['defective']:7.1f} {PAPER['reject']:7.1f} {PAPER['bruise']:7.1f}"
          f" {PAPER['serious']:8.1f} {PAPER['slight']:7.1f} {(280 * PAPER['sound'] + 246 * PAPER['defective']) / 526:8.1f}"
          f" {'(detector)':>9s} {'-':>8s}")
    for name, p in preds.items():
        r = group_report(p, ctx, sev)
        out["rows"][name] = r
        q = r["paper_subset_280_246"]
        print(f"  {name:58s} {q['sound_pct']:6.1f} {q['defective_pct']:7.1f} {q['reject_pct']:7.1f} {q['bruise_pct']:7.1f}"
              f" {q['serious_pct']:8.1f} {q['slight_pct']:7.1f} {q['accuracy'] * 100:8.1f} {r['groups']['stem+calyx']['pct']:9.1f}"
              f" {r['accuracy'] * 100:8.1f}")
    print("\n  McNemar (exact when discordant < 25)")
    comps = [("Kleynen replicated 450/750/800 (paper) [A: paper protocol]", "CNN merged NIR (nir_ss)", "paper_526", paper_set),
             ("Kleynen replicated 450/750/800 (paper) [B: stem/calyx = sound]", "CNN merged NIR (nir_ss)", "all_819", np.ones(n, bool)),
             ("Kleynen replicated 450/750/800 (paper) [A: paper protocol]", "Kleynen replicated 750/800 (NIR) [A: paper protocol]", "paper_526", paper_set),
             ("Kleynen replicated 750/800 (NIR) [A: paper protocol]", "Kleynen replicated 450/500 (Visible) [A: paper protocol]", "paper_526", paper_set),
             ("CNN merged NIR (nir_ss)", "CNN merged Visible (visible_ss)", "all_819", np.ones(n, bool))]
    if os.path.exists(sens):
        comps.append(("Kleynen replicated 450/750/800 (paper) [A, erosion 15px: sensitivity]", "CNN merged NIR (nir_ss)", "paper_526", paper_set))
    if "CNN 2-class NIR (nir_ss_2cls)" in preds:
        comps += [("Kleynen replicated 450/750/800 (paper) [A: paper protocol]", "CNN 2-class NIR (nir_ss_2cls)", "paper_526", paper_set),
                  ("CNN 2-class NIR (nir_ss_2cls)", "CNN merged NIR (nir_ss)", "all_819", np.ones(n, bool)),
                  ("CNN 2-class NIR (nir_ss_2cls)", "CNN 2-class Visible (visible_ss_2cls)", "all_819", np.ones(n, bool))]
    for a, b, tag, m in comps:
        r = mcnemar_exact(yb[m], preds[a][m], preds[b][m])
        acc_a, acc_b = float((preds[a][m] == yb[m]).mean()), float((preds[b][m] == yb[m]).mean())
        out["mcnemar"][f"{a} vs {b} [{tag}]"] = {**r, "acc_a": acc_a, "acc_b": acc_b, "n": int(m.sum())}
        print(f"  [{tag}] {a}\n      vs {b}: acc {acc_a * 100:.1f}% vs {acc_b * 100:.1f}%, only-first correct {r['only_a_correct']}, "
              f"only-second correct {r['only_b_correct']}, p = {r['p_value']:.1e}")
    if all(k4.values()):
        print("\n  paired over the 15 (seed, fold) runs, single model + TTA: 2-class training vs merged 3-class (2-group accuracy / F1)")
        rng = np.random.RandomState(42)
        three = {"nir": load_runs("confirm_results.json", "nir_ss"), "visible": load_runs("n5_results.json", "visible_ss")}
        for band, v2 in (("nir", "nir_ss_2cls"), ("visible", "visible_ss_2cls")):
            for metric in ("accuracy", "f1"):
                def per_run(runs, two):
                    recs = {}
                    for k, r in runs.items():
                        idx = np.array(r["test_idx"])
                        p = np.array(r["tta_test_probs"])
                        pb = (p[:, 1] > 0.5) if two else (p[:, 1:].sum(1) > 0.5)
                        val = accuracy_score(yb[idx], pb) if metric == "accuracy" else f1_score(yb[idx], pb)
                        recs[k] = {"test_macro_f1": float(val)}
                    return recs
                r = paired(per_run(k4[v2], True), per_run(three[band], False), rng)
                out["paired_runs"][f"{band}: 2-class - merged 3-class|{metric}"] = {**r, "verdict": verdict(r)}
                print(f"  {band:7s} 2-class - merged  {metric:8s} {r['mean_diff']:+.3f} CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] "
                      f"{r['same_sign_pairs']}/{r['n_pairs']} -> {verdict(r)}")
    json.dump(out, open(os.path.join(DL, "k5_summary.json"), "w", encoding="utf-8"), indent=1)
    print("saved -> outputs/dl/k5_summary.json")


if __name__ == "__main__":
    commands = {"k1": k1, "k2": k2, "k3": k3, "k3_features_only": lambda: k3(features_only=True), "k5": k5}
    commands[sys.argv[1]](*sys.argv[2:])
