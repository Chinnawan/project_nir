"""N1 -- physical evidence without any CNN: how visible is each defect in each band?

For every one of the 246 Defect images (flat-field-corrected SingleSpectra, full resolution 560x430, the
geometry in which the pixel-level masks were drawn) and every band (BL 450, GR 500, RE 750, IR 800 nm):
  defect  = mask pixels inside the apple
  healthy = a ring of apple surface around the defect (mask dilated by 15 px minus a 3 px gap), other
            defect pixels and the background excluded
  contrast = (healthy - defect) / (healthy + defect)   (signed: > 0 means the defect is darker)
  Cohen's d = (healthy - defect) / pooled std
Detectability uses the ABSOLUTE values (a defect can be darker or brighter than the skin; both are visible).

Null baseline (needed because skin is not uniform: Jonagold is bi-coloured in the visible bands, so ANY region
differs from its surroundings): the same mask, at the same position, is laid on 5 random Healthy apples where it
fits inside the fruit, and measured the same way. Per band:
  excess = |contrast| of the defect - mean |contrast| of the same mask on healthy apples
  AUC    = P(|contrast| defect > |contrast| null)  (Mann-Whitney), 0.5 = indistinguishable from normal skin
Per image: NIR = mean over RE and IR, visible = mean over BL and GR; paired NIR - visible differences are
summarised with a bootstrap 95% CI, separately for Internal and Surface defects and per tissue type.

Usage: python -m src.band_contrast  ->  outputs/dl/nir_proof_n1.json, outputs/figures/band_contrast.png
"""
import json
import os

import cv2
import numpy as np

from .config import OUTPUT_DIR
from .copy_paste import MASK_DIR, apple_region
from .preprocess_ss import MAPPING_PATH, read_bands

BANDS = ["BL", "GR", "RE", "IR"]
OUT_JSON = os.path.join(OUTPUT_DIR, "dl", "nir_proof_n1.json")
FIG = os.path.join(OUTPUT_DIR, "figures", "band_contrast.png")
RING_OUTER, RING_GAP = 31, 7  # ellipse kernel sizes (px) at 560x430
MIN_DEFECT_PX, MIN_RING_PX = 20, 50
N_BOOT = 10000
N_NULL = 5


def boot_ci(x, seed=42):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    m = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(N_BOOT)])
    return float(x.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def measure(bands, mask):
    cube = np.stack(bands, axis=-1)
    apple = apple_region(cube).astype(bool)
    defect = mask & apple
    m8 = mask.astype(np.uint8)
    ring = cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_OUTER, RING_OUTER))).astype(bool)
    ring &= ~cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_GAP, RING_GAP))).astype(bool)
    ring &= apple & ~mask
    fallback = False
    if ring.sum() < MIN_RING_PX:
        ring = apple & ~cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (RING_GAP, RING_GAP))).astype(bool)
        fallback = True
    out = {"defect_px": int(defect.sum()), "ring_px": int(ring.sum()), "ring_fallback": fallback}
    if defect.sum() < MIN_DEFECT_PX or ring.sum() < MIN_RING_PX:
        return None, out
    per_band = {}
    for name, img in zip(BANDS, bands):
        d, h = img[defect].astype(float), img[ring].astype(float)
        md, mh = d.mean(), h.mean()
        pooled = np.sqrt((d.var(ddof=1) + h.var(ddof=1)) / 2) + 1e-9
        per_band[name] = {"defect_mean": md, "healthy_mean": mh, "contrast": (mh - md) / (mh + md + 1e-9),
                          "cohen_d": (mh - md) / pooled}
    return per_band, out


_healthy_cache = {}


def healthy_bands(name):
    if name not in _healthy_cache:
        _healthy_cache[name] = read_bands(name)
    return _healthy_cache[name]


def null_contrasts(mask, healthy_names, rng):
    """|contrast| / |d| of the same mask on N_NULL healthy apples where it lies inside the fruit."""
    out = []
    for _ in range(40):
        name = healthy_names[rng.randint(len(healthy_names))]
        bands = healthy_bands(name)
        apple = apple_region(np.stack(bands, axis=-1)).astype(bool)
        if (mask & ~apple).sum() > 0.02 * mask.sum():
            continue
        per_band, _ = measure(bands, mask)
        if per_band is None:
            continue
        out.append({b: (abs(per_band[b]["contrast"]), abs(per_band[b]["cohen_d"])) for b in BANDS})
        if len(out) == N_NULL:
            break
    return out


def auc_fast(pos, neg):
    """Mann-Whitney AUC via ranks (ties averaged)."""
    from scipy.stats import rankdata
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)[:len(pos)]
    return (r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_diff_ci(rs, n_boot=2000, seed=42):
    """AUC(NIR) - AUC(visible), AUC(NIR) = mean of RE and IR AUCs vs their null, same for BL/GR.
    Bootstrap over defect images (each image carries its own null measurements)."""
    rng = np.random.RandomState(seed)
    pos = {b: np.array([abs(r["bands"][b]["contrast"]) for r in rs]) for b in BANDS}
    neg = {b: [np.array(r["bands"][b]["null_abs_contrast_all"]) for r in rs] for b in BANDS}

    def stat(idx):
        a = {b: auc_fast(pos[b][idx], np.concatenate([neg[b][i] for i in idx])) for b in BANDS}
        return (a["RE"] + a["IR"]) / 2 - (a["BL"] + a["GR"]) / 2

    full = stat(np.arange(len(rs)))
    boots = [stat(rng.randint(0, len(rs), len(rs))) for _ in range(n_boot)]
    return float(full), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def auc(pos, neg):
    from scipy.stats import mannwhitneyu
    return float(mannwhitneyu(pos, neg, alternative="two-sided").statistic / (len(pos) * len(neg)))


def main():
    with open(MAPPING_PATH) as f:
        mapping = json.load(f)
    defects = sorted((v for v in mapping.values() if v["defect_no"] is not None), key=lambda v: v["defect_no"])
    healthy_names = sorted(v["ss_name"] for v in mapping.values() if v["tissue"] == "Healthy")
    rng = np.random.RandomState(42)
    recs, skipped, shape_mismatch = [], [], []
    for v in defects:
        bands = read_bands(v["ss_name"])
        m = cv2.imread(os.path.join(MASK_DIR, f"SBW_Defect{v['defect_no']:04d}.bmp"), cv2.IMREAD_UNCHANGED)
        m = (m[..., 0] if m.ndim == 3 else m) > 127
        if m.shape != bands[0].shape:
            shape_mismatch.append(v["defect_no"])
            continue
        per_band, info = measure(bands, m)
        if per_band is None:
            skipped.append({"defect_no": v["defect_no"], **info})
            continue
        nulls = null_contrasts(m, healthy_names, rng)
        if len(nulls) < 2:
            skipped.append({"defect_no": v["defect_no"], "reason": "mask does not fit inside healthy apples", **info})
            continue
        for b in BANDS:
            per_band[b]["null_abs_contrast"] = float(np.mean([n[b][0] for n in nulls]))
            per_band[b]["null_abs_d"] = float(np.mean([n[b][1] for n in nulls]))
            per_band[b]["null_abs_contrast_all"] = [n[b][0] for n in nulls]
            per_band[b]["excess_abs_contrast"] = abs(per_band[b]["contrast"]) - per_band[b]["null_abs_contrast"]
            per_band[b]["excess_abs_d"] = abs(per_band[b]["cohen_d"]) - per_band[b]["null_abs_d"]
        exc_nir = np.mean([per_band[b]["excess_abs_contrast"] for b in ("RE", "IR")])
        exc_vis = np.mean([per_band[b]["excess_abs_contrast"] for b in ("BL", "GR")])
        excd_nir = np.mean([per_band[b]["excess_abs_d"] for b in ("RE", "IR")])
        excd_vis = np.mean([per_band[b]["excess_abs_d"] for b in ("BL", "GR")])
        nir_abs = np.mean([abs(per_band[b]["contrast"]) for b in ("RE", "IR")])
        vis_abs = np.mean([abs(per_band[b]["contrast"]) for b in ("BL", "GR")])
        nir_d = np.mean([abs(per_band[b]["cohen_d"]) for b in ("RE", "IR")])
        vis_d = np.mean([abs(per_band[b]["cohen_d"]) for b in ("BL", "GR")])
        recs.append({"defect_no": v["defect_no"], "tissue": v["tissue"], "cls": v["cls"], **info,
                     "bands": per_band, "nir_abs_contrast": nir_abs, "vis_abs_contrast": vis_abs,
                     "diff_abs_contrast": nir_abs - vis_abs, "nir_abs_d": nir_d, "vis_abs_d": vis_d,
                     "diff_abs_d": nir_d - vis_d, "n_null": len(nulls),
                     "diff_excess_abs_contrast": exc_nir - exc_vis, "diff_excess_abs_d": excd_nir - excd_vis})

    def summarise(rs):
        s = {"n": len(rs)}
        for b in BANDS:
            s[b] = {"abs_contrast": boot_ci([abs(r["bands"][b]["contrast"]) for r in rs]),
                    "signed_contrast_mean": float(np.mean([r["bands"][b]["contrast"] for r in rs])),
                    "abs_cohen_d": boot_ci([abs(r["bands"][b]["cohen_d"]) for r in rs])}
        for b in BANDS:
            s[b]["null_abs_contrast"] = float(np.mean([r["bands"][b]["null_abs_contrast"] for r in rs]))
            s[b]["excess_abs_contrast"] = boot_ci([r["bands"][b]["excess_abs_contrast"] for r in rs])
            s[b]["excess_abs_d"] = boot_ci([r["bands"][b]["excess_abs_d"] for r in rs])
            s[b]["auc_vs_null"] = auc([abs(r["bands"][b]["contrast"]) for r in rs],
                                      [x for r in rs for x in r["bands"][b]["null_abs_contrast_all"]])
        s["auc_nir_minus_visible"] = auc_diff_ci(rs)
        s["nir_minus_visible_excess_abs_contrast"] = boot_ci([r["diff_excess_abs_contrast"] for r in rs])
        s["nir_minus_visible_excess_abs_d"] = boot_ci([r["diff_excess_abs_d"] for r in rs])
        s["frac_images_nir_higher_excess"] = float(np.mean([r["diff_excess_abs_contrast"] > 0 for r in rs]))
        s["nir_minus_visible_abs_contrast"] = boot_ci([r["diff_abs_contrast"] for r in rs])
        s["nir_minus_visible_abs_d"] = boot_ci([r["diff_abs_d"] for r in rs])
        s["frac_images_nir_higher_contrast"] = float(np.mean([r["diff_abs_contrast"] > 0 for r in rs]))
        return s

    groups = {"ALL defects": recs}
    for c in ("Internal_Defect", "Surface_Defect"):
        groups[c] = [r for r in recs if r["cls"] == c]
    tissues = sorted({r["tissue"] for r in recs}, key=lambda t: (next(r["cls"] for r in recs if r["tissue"] == t), t))
    for t in tissues:
        groups[t] = [r for r in recs if r["tissue"] == t]
    summary = {k: summarise(v) for k, v in groups.items()}

    print(f"N1 band contrast: {len(recs)} defect images measured, {len(skipped)} skipped (too small), "
          f"{len(shape_mismatch)} mask/image size mismatches, ring fallback used in {sum(r['ring_fallback'] for r in recs)}")
    print(f"\n  {'group':18s} {'n':>4s} " + " ".join(f"|c| {b:>3s}" for b in BANDS) +
          "   NIR-vis |c| [95% CI]      NIR higher   NIR-vis |d| [95% CI]")
    for k, s in summary.items():
        dc, dd = s["nir_minus_visible_abs_contrast"], s["nir_minus_visible_abs_d"]
        print(f"  {k:18s} {s['n']:4d} " + " ".join(f"{s[b]['abs_contrast'][0]:7.3f}" for b in BANDS) +
              f"   {dc[0]:+.3f} [{dc[1][0]:+.3f},{dc[1][1]:+.3f}]   {s['frac_images_nir_higher_contrast'] * 100:5.1f}%"
              f"   {dd[0]:+.2f} [{dd[1][0]:+.2f},{dd[1][1]:+.2f}]")
    print("\n  WITH NULL BASELINE (same mask on healthy apples): excess |c| = defect - null; AUC defect vs null")
    print(f"  {'group':18s} {'n':>4s} " + " ".join(f"null {b:>3s}" for b in BANDS) + "  " +
          " ".join(f"exc {b:>3s}" for b in BANDS) + "  " + " ".join(f"AUC {b:>3s}" for b in BANDS) +
          "   AUC NIR-vis [95% CI]    NIR-vis excess |c| [95% CI]   NIR-vis excess |d| [95% CI]")
    for k, s in summary.items():
        e, ed = s["nir_minus_visible_excess_abs_contrast"], s["nir_minus_visible_excess_abs_d"]
        print(f"  {k:18s} {s['n']:4d} " + " ".join(f"{s[b]['null_abs_contrast']:8.3f}" for b in BANDS) + "  " +
              " ".join(f"{s[b]['excess_abs_contrast'][0]:+7.3f}" for b in BANDS) + "  " +
              " ".join(f"{s[b]['auc_vs_null']:7.2f}" for b in BANDS) +
              f"   {s['auc_nir_minus_visible'][0]:+.3f} [{s['auc_nir_minus_visible'][1][0]:+.3f},{s['auc_nir_minus_visible'][1][1]:+.3f}]"
              f"   {e[0]:+.3f} [{e[1][0]:+.3f},{e[1][1]:+.3f}]   {ed[0]:+.2f} [{ed[1][0]:+.2f},{ed[1][1]:+.2f}]")
    print("\n  signed contrast (> 0: defect darker than the surrounding skin):")
    for k in ["Internal_Defect", "Surface_Defect"] + tissues:
        s = summary[k]
        print(f"  {k:18s} " + " ".join(f"{b} {s[b]['signed_contrast_mean']:+.3f}" for b in BANDS))

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["Internal_Defect", "Surface_Defect"] + tissues
    colors = {"BL": "#3b6fd4", "GR": "#2e9e4f", "RE": "#c0392b", "IR": "#7d3c98"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    x = np.arange(len(names))
    for i, b in enumerate(BANDS):
        lab = f"{b} ({ {'BL': 450, 'GR': 500, 'RE': 750, 'IR': 800}[b]} nm)"
        ex = [summary[n][b]["excess_abs_contrast"] for n in names]
        err = [[e[0] - e[1][0] for e in ex], [e[1][1] - e[0] for e in ex]]
        axes[0].bar(x + (i - 1.5) * 0.2, [e[0] for e in ex], 0.2, yerr=err, capsize=2, color=colors[b], label=lab)
        axes[1].bar(x + (i - 1.5) * 0.2, [summary[n][b]["auc_vs_null"] for n in names], 0.2, color=colors[b], label=lab)
    axes[0].set_ylabel("excess |contrast|\n(defect - same mask on healthy apples)")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("AUC defect vs normal skin\n(0.5 = indistinguishable)")
    axes[1].axhline(0.5, color="k", lw=0.8, ls="--")
    axes[1].set_ylim(0.3, 1.0)
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=4, fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{n}\n(n={summary[n]['n']})" for n in names], rotation=45, ha="right", fontsize=8)
    fig.suptitle("N1: defect visibility per band (flat-field corrected images, 95% bootstrap CI)")
    plt.tight_layout()
    plt.savefig(FIG, dpi=140)
    plt.close()

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"n_measured": len(recs), "skipped": skipped, "mask_shape_mismatch": shape_mismatch,
                   "method": {"images": "SingleSpectra 560x430 (flat-field corrected)", "ring_outer_px": RING_OUTER,
                              "ring_gap_px": RING_GAP, "detectability": "absolute contrast / absolute Cohen's d"},
                   "summary": summary, "per_image": recs}, f, indent=1)
    print(f"\nsaved -> {OUT_JSON}\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
