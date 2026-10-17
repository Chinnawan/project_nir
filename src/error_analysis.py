"""N8 -- error analysis of the final formula + 2-group (healthy vs defective) results.

Final formula = nir_ss + TTA (6 views) + soft vote of seeds 3,4,5 inside a fold, pooled out-of-fold over all 819
apples (confirm_results.json; must reproduce confirm_summary.json -> asserted).

  1. error types: (a) defective apple called healthy, (b) healthy apple called defective,
                  (c) defect found but put in the wrong defect group (Internal <-> Surface)
  2. per-tissue table: how each tissue type was predicted (H / I / S) + mean confidence when wrong
  3. 2 groups: P(defective) = P(Internal) + P(Surface) > 0.5. nir_ss vs visible_ss vs full_ss (N5, same seeds,
     same SingleSpectra preprocessing -> fair) [+ raw nir/visible/full for context]: accuracy, F1 (defective = positive),
     defect recall, healthy recall; paired over the 15 (seed, fold) runs + McNemar on the pooled 3-seed vote
  4. 3 example apples of type (c) where the Grad-CAM peak lies on the real defect mask
     -> outputs/figures/private/n8_wrong_group_examples.png (gitignored: dataset pixels)
Rules (PROJECT_PLAN.md): no tuning to fix these apples; Rot stays Internal_Defect.

  python -m src.error_analysis  ->  outputs/dl/n8_error_analysis.json
"""
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import CLASSES, OUTPUT_DIR
from .multiseed_report import paired
from .nir_proof import mcnemar_exact, verdict
from .round2 import Context

DL = os.path.join(OUTPUT_DIR, "dl")
OUT_JSON = os.path.join(DL, "n8_error_analysis.json")
FIG = os.path.join(OUTPUT_DIR, "figures", "private", "n8_wrong_group_examples.png")
H, I, S = 0, 1, 2
TISSUE_ORDER = ["Bruise", "Flesh_damage", "Rot", "Russet", "Scald", "Hail_damage", "Hail_damage_perf", "Frost_damage",
                "Limb_rub", "Other", "Stem", "Calix", "Healthy"]


def load_runs(fname, variant):
    p = os.path.join(DL, fname)
    if not os.path.exists(p):
        return None
    runs = {(r["seed"], r["fold"]): r for r in json.load(open(p, encoding="utf-8"))["runs"].values() if r["variant"] == variant}
    return runs if len(runs) == 15 else None


def vote_probs(runs, n):
    """pooled out-of-fold 3-seed soft vote with TTA"""
    p, cnt = np.zeros((n, 3)), np.zeros(n)
    for r in runs.values():
        idx = np.array(r["test_idx"])
        p[idx] += np.array(r["tta_test_probs"])
        cnt[idx] += 1
    assert (cnt == 3).all()
    return p / 3


def binary_metrics(yb, pb):
    tp, tn = int(np.sum((pb == 1) & (yb == 1))), int(np.sum((pb == 0) & (yb == 0)))
    return {"accuracy": float(accuracy_score(yb, pb)), "f1_defective": float(f1_score(yb, pb)),
            "defect_recall": tp / int(yb.sum()), "healthy_recall": tn / int((1 - yb).sum()),
            "defects_missed": int(np.sum((pb == 0) & (yb == 1))), "healthy_rejected": int(np.sum((pb == 1) & (yb == 0)))}


def main():
    ctx = Context()
    y, tissue = np.array(ctx.labels), np.array(ctx.tissue)
    names = [os.path.basename(p[0]).replace(".npy", "") for p in ctx.pool_raw]
    n = len(y)
    out = {"definition": "nir_ss + TTA + 3-seed soft vote (seeds 3-5), pooled out-of-fold, 819 apples"}

    final = load_runs("confirm_results.json", "nir_ss")
    P = vote_probs(final, n)
    pred = P.argmax(1)
    ref = json.load(open(os.path.join(DL, "confirm_summary.json"), encoding="utf-8"))["final_formula"]
    assert abs(accuracy_score(y, pred) - ref["accuracy"]) < 1e-9, "does not reproduce confirm_summary.json"
    conf = P.max(1)

    # 1. error types
    a = (y != H) & (pred == H)
    b = (y == H) & (pred != H)
    c = (y != H) & (pred != H) & (pred != y)
    out["error_types"] = {
        "a_defect_called_healthy": {"n": int(a.sum()), "of": int((y != H).sum()), "mean_conf": float(conf[a].mean())},
        "b_healthy_called_defect": {"n": int(b.sum()), "of": int((y == H).sum()), "mean_conf": float(conf[b].mean())},
        "c_wrong_defect_group": {"n": int(c.sum()), "of": int((y != H).sum()), "mean_conf": float(conf[c].mean()),
                                 "internal_called_surface": int(np.sum(c & (y == I))), "surface_called_internal": int(np.sum(c & (y == S)))},
        "total_errors": int((pred != y).sum()), "accuracy": float(accuracy_score(y, pred))}
    e = out["error_types"]
    print(f"N8  error analysis of the final formula ({out['definition']})")
    print(f"  accuracy {e['accuracy'] * 100:.1f}%, {e['total_errors']} errors of {n}")
    print(f"  (a) defective called healthy      {e['a_defect_called_healthy']['n']:3d} / {e['a_defect_called_healthy']['of']} defective"
          f"   mean confidence {e['a_defect_called_healthy']['mean_conf']:.2f}")
    print(f"  (b) healthy called defective      {e['b_healthy_called_defect']['n']:3d} / {e['b_healthy_called_defect']['of']} healthy"
          f"    mean confidence {e['b_healthy_called_defect']['mean_conf']:.2f}")
    print(f"  (c) defect found, wrong group     {e['c_wrong_defect_group']['n']:3d}  (Internal->Surface {e['c_wrong_defect_group']['internal_called_surface']},"
          f" Surface->Internal {e['c_wrong_defect_group']['surface_called_internal']})   mean confidence {e['c_wrong_defect_group']['mean_conf']:.2f}")
    print(f"  mean confidence when correct: {conf[pred == y].mean():.2f}")
    out["mean_conf_correct"] = float(conf[pred == y].mean())

    # 2. per tissue
    out["per_tissue"] = {}
    print(f"\n  {'tissue':17s} {'class':16s} {'n':>4s}  pred H   I   S   correct  mean conf when wrong")
    for t in TISSUE_ORDER:
        m = tissue == t
        if not m.any():
            continue
        cls = int(y[m][0])
        cnt = [int(np.sum(pred[m] == k)) for k in range(3)]
        wrong = m & (pred != y)
        r = {"class": CLASSES[cls], "n": int(m.sum()), "pred_H_I_S": cnt, "recall": cnt[cls] / int(m.sum()),
             "mean_conf_wrong": float(conf[wrong].mean()) if wrong.any() else None}
        out["per_tissue"][t] = r
        mc = f"{r['mean_conf_wrong']:.2f}" if wrong.any() else "  -"
        print(f"  {t:17s} {CLASSES[cls]:16s} {r['n']:4d}  {cnt[0]:6d} {cnt[1]:3d} {cnt[2]:3d}   {r['recall'] * 100:5.1f}%   {mc}")

    # 3. two groups
    yb = (y != H).astype(int)
    models = {"nir_ss": final, "visible_ss": load_runs("n5_results.json", "visible_ss"), "full_ss": load_runs("n5_results.json", "full_ss"),
              "nir (raw)": load_runs("confirm_results.json", "nir"), "visible (raw)": load_runs("fusion_results.json", "visible"),
              "full (raw)": load_runs("fusion_results.json", "full")}
    models = {k: v for k, v in models.items() if v is not None}
    out["binary"] = {"rule": "defective if P(Internal) + P(Surface) > 0.5", "vote": {}, "per_run_mean": {}, "paired": {}, "mcnemar_vote": {}}
    bpred = {k: ((vote_probs(v, n)[:, 1:].sum(1)) > 0.5).astype(int) for k, v in models.items()}
    out["binary"]["argmax_collapsed_nir_ss_accuracy"] = float(accuracy_score(yb, (pred != H).astype(int)))
    print(f"\n  2 groups (healthy vs defective; {out['binary']['rule']}), pooled 3-seed vote + TTA, 819 apples")
    print(f"  {'model':14s} {'accuracy':>9s} {'F1 defect':>10s} {'defect recall':>14s} {'healthy recall':>15s}  missed defects / rejected healthy")
    for k, pb in bpred.items():
        mt = binary_metrics(yb, pb)
        out["binary"]["vote"][k] = mt
        print(f"  {k:14s} {mt['accuracy'] * 100:8.1f}% {mt['f1_defective']:10.3f} {mt['defect_recall'] * 100:13.1f}% {mt['healthy_recall'] * 100:14.1f}%"
              f"   {mt['defects_missed']:3d} / {mt['healthy_rejected']:3d}")
    print(f"  (nir_ss with argmax of 3 classes collapsed to 2 groups: accuracy {out['binary']['argmax_collapsed_nir_ss_accuracy'] * 100:.1f}%)")

    def per_run(runs, metric):
        recs = {}
        for k, r in runs.items():
            idx = np.array(r["test_idx"])
            pb = (np.array(r["tta_test_probs"])[:, 1:].sum(1) > 0.5).astype(int)
            yy = yb[idx]
            recs[k] = {"test_macro_f1": float(f1_score(yy, pb)) if metric == "f1" else float(accuracy_score(yy, pb))}
        return recs
    rng = np.random.RandomState(42)
    pairs = [(x, z) for x, z in (("nir_ss", "visible_ss"), ("nir_ss", "full_ss"), ("full_ss", "full (raw)")) if x in models and z in models]
    if pairs:
        print("\n  paired over the 15 (seed, fold) runs, single model + TTA (2-group F1 / accuracy)")
    for x, z in pairs:
        for metric in ("f1", "accuracy"):
            r = paired(per_run(models[x], metric), per_run(models[z], metric), rng)
            out["binary"]["paired"][f"{x} - {z}|{metric}"] = {**r, "verdict": verdict(r)}
            print(f"  {x:10s} - {z:11s} {metric:8s} {r['mean_diff']:+.3f} CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] "
                  f"{r['same_sign_pairs']}/{r['n_pairs']} -> {verdict(r)}")
        m = mcnemar_exact(yb, bpred[x], bpred[z])
        out["binary"]["mcnemar_vote"][f"{x} - {z}"] = m
        print(f"  {x:10s} vs {z:11s} McNemar (vote): only-{x} correct {m['only_a_correct']}, only-{z} correct {m['only_b_correct']}, p = {m['p_value']:.1e}")
    for k in out["binary"]["vote"]:
        out["binary"]["per_run_mean"][k] = {m: float(np.mean([v["test_macro_f1"] for v in per_run(models[k], m).values()])) for m in ("f1", "accuracy")}

    # 4. examples of type (c) with the Grad-CAM peak on the defect mask
    from .copy_paste import load_defect_mask
    from .demo_app import Demo
    demo = Demo()
    cands = []
    for i in np.where(c)[0]:
        rawid = names[i]
        info = demo.images[rawid]
        cube = np.load(demo.paths[rawid])
        cam = demo.cam(cube, demo.keys_for_known(rawid), int(pred[i]))
        mask = load_defect_mask(info["defect_no"])
        cam224 = cv2.resize(cam, mask.shape[::-1])
        peak = np.unravel_index(np.argmax(cam224), cam224.shape)
        hit = bool(cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))[peak])
        top = cam224 >= np.quantile(cam224, 0.9)
        cands.append({"id": rawid, "tissue": info["tissue"], "true": CLASSES[y[i]], "pred": CLASSES[pred[i]],
                      "probs": P[i].round(3).tolist(), "peak_on_mask": hit,
                      "top10pct_cam_on_mask": float((top & mask).sum() / top.sum()), "_cube": cube, "_cam": cam, "_mask": mask})
    cands.sort(key=lambda r: (not r["peak_on_mask"], -r["top10pct_cam_on_mask"]))
    out["wrong_group_cases"] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in cands]
    chosen = [r for r in cands if r["peak_on_mask"]][:3]
    out["wrong_group_examples"] = [r["id"] for r in chosen]
    print(f"\n  type (c) cases: {len(cands)}, Grad-CAM peak on the defect mask in {sum(r['peak_on_mask'] for r in cands)}")
    for r in out["wrong_group_cases"]:
        print(f"  {r['id']}  {r['tissue']:16s} true {r['true']:16s} pred {r['pred']:16s} P(H,I,S) {r['probs']}  "
              f"peak on mask {'yes' if r['peak_on_mask'] else 'no '}  top-10% CAM on mask {r['top10pct_cam_on_mask'] * 100:.0f}%")
    if chosen:
        from .demo_app import overlay
        fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 5.4))
        for ax, r in zip(np.atleast_1d(axes), chosen):
            ax.imshow(overlay(r["_cube"][..., 3], r["_cam"], r["_mask"]))
            ax.set_title(f"{r['id']} {r['tissue']}\ntrue {r['true']} -> predicted {r['pred']}", fontsize=10)
            ax.axis("off")
        fig.suptitle("Defect found (Grad-CAM peak on the real defect, green outline) but put in the wrong group", fontsize=11)
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(FIG, dpi=100)
        print("  figure (dataset pixels, gitignored) ->", FIG)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
    print("saved ->", OUT_JSON)


if __name__ == "__main__":
    main()
