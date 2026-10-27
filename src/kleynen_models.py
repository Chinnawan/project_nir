"""G1: the K3 Kleynen 2005 pipeline (erosion 5 px, the primary K3 setting) as saved models for the GUI.

  python -m src.kleynen_models      -> models/kleynen_{450_750_800,750_800}_{fold0..fold4,all}.pkl  (gitignored)

FOR DISPLAY IN THE GUI ONLY. The evaluation numbers stay those of K3 (cross-fitting, outputs/dl/k3_kleynen.json).
  * fold{f}: exactly the K3 model of outer fold f, setting A (paper protocol): histograms (LUT) of the training folds,
    LDA on the cross-fitted features of the training sound + defective fruits. The GUI uses it for the dataset images
    of fold f only (fold_assignment.json) -> the model never saw the image, like the CNN. fit() asserts that these
    models reproduce the K3 decisions (and posteriors) of all 819 images.
  * all: LUT from all images, LDA on all 280 sound + 246 defective fruits (features of each fruit from the LUT that
    excludes its own fold, as in K3). Used only for images that are not in the dataset (upload).
"""
import json
import os
import pickle
import time

import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import kleynen as K
from .config import OUTPUT_DIR, PROJECT_ROOT

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
GUI_SETS = {"450/750/800 (paper)": "450_750_800", "750/800 (NIR)": "750_800"}
DEFAULT_SET = "450/750/800 (paper)"


def model_path(band_set, fold=None):
    return os.path.join(MODEL_DIR, f"kleynen_{GUI_SETS[band_set]}_{'all' if fold is None else f'fold{fold}'}.pkl")


def all_present():
    return all(os.path.exists(model_path(bs, f)) for bs in GUI_SETS for f in [None, *range(5)])


def load_model(band_set, fold=None):
    with open(model_path(band_set, fold), "rb") as f:
        return pickle.load(f)


def load_ss_bands(rawid, mapping=None):
    """full-resolution SingleSpectra bands of a dataset image: (430, 560, 4) uint8 BL/GR/RE/IR"""
    from .preprocess_ss import read_bands
    mapping = mapping or json.load(open(os.path.join(OUTPUT_DIR, "ss_mapping.json")))
    return np.stack(read_bands(mapping[str(int(rawid[4:]))]["ss_name"]), axis=-1)


def erosion_kernel(px):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def analyze(bands4, model):
    """Every step of the pipeline on one image (H, W, 4) uint8 -> dict; "ms" = processing time from the loaded bands
    to the decision (fruit mask, erosion, pixel classification, features, LDA)."""
    t0 = time.perf_counter()
    fm = K.fruit_mask(bands4[..., 3])
    er = cv2.erode(fm, erosion_kernel(model["erode_px"]))
    idx = np.flatnonzero(er)
    if len(idx) == 0:
        raise ValueError("ไม่พบผลแอปเปิลในภาพ (threshold แบนด์ 800 nm ไม่ได้พื้นที่ผล)")
    rec = {"idx": idx, "vals": bands4.reshape(-1, 4)[idx], "shape": fm.shape}
    feats, d = K.fruit_features(rec, model["bands"], model["lut"])
    p_def = float(model["lda"].predict_proba(feats[None])[0, 1])
    pred = int(model["lda"].predict(feats[None])[0])
    ms = (time.perf_counter() - t0) * 1000
    dmap = np.zeros(fm.size, np.uint8)
    dmap[idx[d]] = 1
    nb = len(model["bands"])
    return {"fruit_mask": fm, "eroded": er, "defect_map": dmap.reshape(fm.shape), "features": feats,
            "defect_ratio": float(feats[6 * nb]), "n_regions": int(feats[6 * nb + 1]), "largest_region": float(feats[6 * nb + 2]),
            "p_defect": p_def, "pred": pred, "ms": ms}


def fit():
    assert K.ERODE_PX == 5, "GUI models use the primary K3 setting (erosion 5 px)"
    ctx, names, fold = K.context()
    tissue = np.asarray(ctx.tissue)
    yb = (np.asarray(ctx.labels) != 0).astype(int)
    paper = ~np.isin(tissue, ["Stem", "Calix"])
    k3 = json.load(open(os.path.join(K.DL, "k3_kleynen.json"), encoding="utf-8"))
    assert f"erosion {K.ERODE_PX}px" in k3["method"]["fruit_mask"]
    print("G1  loading 819 SingleSpectra images ...")
    cache = K.load_cache(ctx, names)
    os.makedirs(MODEL_DIR, exist_ok=True)
    report = {}
    for bs, tag in GUI_SETS.items():
        bands = K.BAND_SETS[bs]
        n_cells = K.LEVELS ** len(bands)
        sound, defect = K.per_image_hist(cache, ctx, bands)
        base = {"band_set": bs, "bands": bands, "band_names": [K.BANDS[b] for b in bands], "levels": K.LEVELS,
                "erode_px": K.ERODE_PX, "close_px": K.CLOSE_PX, "min_region_px": K.MIN_REGION_PX, "setting": "A_paper",
                "purpose": "GUI display only; evaluation numbers = K3 cross-fitting (outputs/dl/k3_kleynen.json)"}
        pred_all, post_all = np.full(len(names), -1), np.zeros(len(names))
        for f in range(5):  # the K3 model of outer fold f, reproduced
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            lut = K.lut_from(tr, sound, defect, n_cells)
            luts_in = {g: K.lut_from(tr[fold[tr] != g], sound, defect, n_cells) for g in range(5) if g != f}
            Xtr = np.array([K.fruit_features(cache[i], bands, luts_in[fold[i]])[0] for i in tr])
            use = paper[tr]
            lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(Xtr[use], yb[tr][use])
            Xte = np.array([K.fruit_features(cache[i], bands, lut)[0] for i in te])
            pred_all[te] = lda.predict(Xte)
            post_all[te] = lda.predict_proba(Xte)[:, 1]
            m = {**base, "fold": f, "lut": lut, "lda": lda, "test_images": [names[i] for i in te],
                 "n_train_sound": int(np.sum(yb[tr][use] == 0)), "n_train_defective": int(np.sum(yb[tr][use] == 1))}
            with open(model_path(bs, f), "wb") as fh:
                pickle.dump(m, fh)
        same = np.array_equal(pred_all, np.array(k3["predictions"][f"{bs}|A_paper"]))
        dpost = float(np.max(np.abs(np.round(post_all, 5) - np.array(k3["predictions"][f"{bs}|A_paper|p_defect"]))))
        assert same and dpost < 1e-5, f"{bs}: fold models do not reproduce K3 (same decisions {same}, max posterior diff {dpost})"
        # all-data model (uploads only)
        lut = K.lut_from(np.arange(len(names)), sound, defect, n_cells)
        luts_out = {g: K.lut_from(np.where(fold != g)[0], sound, defect, n_cells) for g in range(5)}
        X = np.array([K.fruit_features(cache[i], bands, luts_out[fold[i]])[0] for i in np.where(paper)[0]])
        lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()).fit(X, yb[paper])
        with open(model_path(bs), "wb") as fh:
            pickle.dump({**base, "fold": None, "lut": lut, "lda": lda, "n_train_sound": int(np.sum(yb[paper] == 0)),
                         "n_train_defective": int(np.sum(yb[paper] == 1))}, fh)
        acc526 = float((pred_all[paper] == yb[paper]).mean())
        report[bs] = {"fold_models_reproduce_k3_decisions": bool(same), "max_posterior_diff_vs_k3": dpost,
                      "acc_526_from_fold_models": acc526}
        print(f"  {bs}: 5 fold models reproduce K3 for all 819 images (decisions identical, max posterior diff {dpost:.1e}); "
              f"acc 526 = {acc526 * 100:.1f}%; all-data model trained on {int(paper.sum())} fruits -> {MODEL_DIR}")
    out = {"purpose": "G1: Kleynen GUI models (display only; evaluation numbers = K3 cross-fitting)", "erode_px": K.ERODE_PX,
           "files": sorted(os.path.basename(model_path(bs, f)) for bs in GUI_SETS for f in [None, *range(5)]),
           "band_sets": report}
    json.dump(out, open(os.path.join(K.DL, "g1_kleynen_models.json"), "w", encoding="utf-8"), indent=1)
    print("saved -> outputs/dl/g1_kleynen_models.json")
    return report


if __name__ == "__main__":
    fit()
