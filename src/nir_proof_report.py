"""N6 -- one report for "is NIR better than Visible, and why?" built from the N1-N4 result files.

  python -m src.nir_proof_report  ->  outputs/logs/nir_proof_report.txt, outputs/dl/nir_proof_summary.json
Inputs: outputs/dl/nir_proof_n1.json (band contrast), nir_proof_n2.json (per class/tissue + McNemar),
        nir_proof_n3.json (classical ML), nir_proof_n4.json (late vs early fusion + band alignment),
        multiseed_summary.json (single-band results for the headline table).
"""
import io
import json
import os
from contextlib import redirect_stdout

from .config import OUTPUT_DIR

DL = os.path.join(OUTPUT_DIR, "dl")
REPORT = os.path.join(OUTPUT_DIR, "logs", "nir_proof_report.txt")
SUMMARY = os.path.join(DL, "nir_proof_summary.json")


def load(name):
    p = os.path.join(DL, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def pct(x):
    return f"{x * 100:.1f}%"


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def main():
    n1, n2, n3, n4, ms = (load(f) for f in ("nir_proof_n1.json", "nir_proof_n2.json", "nir_proof_n3.json",
                                              "nir_proof_n4.json", "multiseed_summary.json"))
    summary = {}
    buf = io.StringIO()
    with redirect_stdout(buf):
        print("PROVING 'NIR (RE 750 + IR 800 nm) > VISIBLE (BL 450 + GR 500 nm)' -- evidence report")
        print("Visible here = blue + green only (CAPA has no red band); claims are about 450/500 vs 750/800 nm, not RGB cameras.")

        if ms:
            section("0) Headline numbers (multi-seed: 3 seeds x 5 folds, same pipeline for every variant)")
            v = ms["variants"]
            print(f"  {'input':24s} {'macro-F1':>16s} {'accuracy':>9s}")
            for name, label in (("nir", "NIR  RE+IR"), ("re_only", "NIR  RE only"), ("ir_only", "NIR  IR only"),
                                ("visible", "VIS  BL+GR"), ("bl_only", "VIS  BL only"), ("gr_only", "VIS  GR only"),
                                ("full", "ALL  4 bands"), ("gr_nir", "GR + NIR")):
                s = v[name]
                print(f"  {label:24s} {s['f1_mean_of_seed_means']:.3f} +- {s['f1_std_of_seed_means']:.3f}   "
                      f"{pct(s['acc_mean_of_seed_means']):>8s}")
            p = ms["paired"]
            # (stored key, label, sign): single-band pairs are stored as visible - NIR, shown as NIR - visible
            for k, label, sgn in (("nir_minus_visible", "RE+IR - BL+GR", 1), ("bl_only_minus_re_only", "RE - BL", -1),
                                  ("gr_only_minus_re_only", "RE - GR", -1), ("bl_only_minus_ir_only", "IR - BL", -1),
                                  ("gr_only_minus_ir_only", "IR - GR", -1), ("nir_minus_full", "RE+IR - all 4", 1)):
                if k in p:
                    lo, hi = sorted((sgn * p[k]["ci95"][0], sgn * p[k]["ci95"][1]))
                    print(f"  {label:24s} {sgn * p[k]['mean_diff']:+.3f} CI [{lo:+.3f},{hi:+.3f}] "
                          f"same sign {p[k]['same_sign_pairs']}/15 -> {'REAL' if p[k]['real_difference'] else 'not larger than noise'}")
            summary["headline"] = {k: v[k] for k in ("nir", "visible", "full", "re_only", "ir_only", "bl_only", "gr_only")}

        if n1:
            section("N1) Physics without any CNN: defect visibility per band (244 defect images, pixel masks)")
            s = n1["summary"]
            print("  |c| = |contrast| defect vs surrounding skin; null = same mask laid on healthy apples (natural skin variation);")
            print("  AUC = how well the defect contrast separates from normal-skin contrast (0.5 = indistinguishable)")
            print(f"  {'group':18s} {'n':>4s}  {'|c| BL  GR  RE  IR':>24s}   {'null BL  GR  RE  IR':>26s}   {'AUC BL  GR  RE  IR':>22s}"
                  "   AUC NIR-vis [95% CI]")
            for k in ["ALL defects", "Internal_Defect", "Surface_Defect", "Bruise", "Flesh_damage", "Rot", "Russet", "Scald",
                      "Hail_damage", "Hail_damage_perf", "Frost_damage", "Limb_rub", "Other"]:
                g = s[k]
                a = g["auc_nir_minus_visible"]
                print(f"  {k:18s} {g['n']:4d}  " + " ".join(f"{g[b]['abs_contrast'][0]:.3f}" for b in "BL GR RE IR".split()) +
                      "   " + " ".join(f"{g[b]['null_abs_contrast']:.3f}" for b in "BL GR RE IR".split()) +
                      "   " + " ".join(f"{g[b]['auc_vs_null']:.2f}" for b in "BL GR RE IR".split()) +
                      f"   {a[0]:+.3f} [{a[1][0]:+.3f},{a[1][1]:+.3f}]")
            summary["n1"] = {k: {"auc_nir_minus_visible": s[k]["auc_nir_minus_visible"],
                                 "auc": {b: s[k][b]["auc_vs_null"] for b in "BL GR RE IR".split()},
                                 "abs_contrast": {b: s[k][b]["abs_contrast"][0] for b in "BL GR RE IR".split()},
                                 "null_abs_contrast": {b: s[k][b]["null_abs_contrast"] for b in "BL GR RE IR".split()}}
                             for k in s}

        if n2:
            section("N2) Which classes / defect types does NIR win? (multi-seed, nir vs visible and nir vs full)")
            for k, r in n2["per_class"].items():
                print(f"  {k:32s} F1 {r['mean_a']:.3f} vs {r['mean_b']:.3f}  diff {r['mean_diff']:+.3f} CI [{r['ci95'][0]:+.3f},"
                      f"{r['ci95'][1]:+.3f}] {r['same_sign_pairs']}/15, folds {r['fold_level']['same_sign_folds']}/5 -> "
                      f"{'REAL' if r['real_difference'] else 'not larger than noise'}")
            print("\n  recall per tissue (correct class), mean over 3 seeds; CI bootstrap over apples")
            for t, r in n2["per_tissue"].items():
                flag = "NIR better" if r["ci95"][0] > 0 else ("visible better" if r["ci95"][1] < 0 else "")
                print(f"  {t:17s} n={r['n']:3d}  NIR {pct(r['recall_nir']):>6s}  visible {pct(r['recall_visible']):>6s}  "
                      f"all-4 {pct(r['recall_full']):>6s}   NIR-vis {r['nir_minus_visible'] * 100:+6.1f} pts "
                      f"[{r['ci95'][0] * 100:+.1f},{r['ci95'][1] * 100:+.1f}] {flag}")
            print("\n  McNemar (pooled out-of-fold predictions, 819 apples, each seed)")
            for k, r in n2["mcnemar"].items():
                print(f"  {k:22s} only-first correct {r['only_a_correct']:3d}  only-second correct {r['only_b_correct']:3d}  p = {r['p_value']:.1e}")
            summary["n2"] = n2

        if n3:
            section("N3) Not a CNN artefact? Classical ML on visible-only vs NIR-only hand-crafted features (same folds)")
            for name, d in n3["models"].items():
                for fs in ("visible", "nir", "all"):
                    print(f"  {name:13s} {fs:8s} macro-F1 {d[fs]['f1_mean']:.3f} +- {d[fs]['f1_std']:.3f}   accuracy {pct(d[fs]['acc_mean'])}")
                q = d["nir_minus_visible"]
                print(f"  {name:13s} NIR - visible {q['mean']:+.3f} (fold t-CI [{q['ci95'][0]:+.3f},{q['ci95'][1]:+.3f}], "
                      f"{q['folds_nir_better']}/5 folds), McNemar p = {q['mcnemar']['p_value']:.1e}")
            summary["n3"] = {name: {fs: {"f1": d[fs]["f1_mean"], "acc": d[fs]["acc_mean"]} for fs in ("visible", "nir", "all")}
                             | {"nir_minus_visible": d["nir_minus_visible"]} for name, d in n3["models"].items()}

        if n4:
            section("N4) Does visible add information? Late fusion (average of separate NIR and visible networks) vs early fusion")
            print(f"  seeds 3-5, {n4['n_pairs']} paired runs (nir from confirm_results.json, visible/full from fusion_results.json)")
            for k, r in n4["weights"].items():
                if "[TTA]" not in k or k.startswith("late w=0.5") or k in ("nir[TTA]", "visible[TTA]", "full[TTA]"):
                    print(f"  {k:18s} macro-F1 {r['macro_f1']:.3f}   accuracy {pct(r['accuracy'])}")
            print()
            for k, r in n4["paired"].items():
                print(f"  {k:30s} {r['mean_diff']:+.3f} CI [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}] {r['same_sign_pairs']}/{r['n_pairs']}, "
                      f"folds {r['fold_level']['same_sign_folds']}/5 -> {r['verdict']}")
            c = n4["complementarity"]
            print(f"\n  complementarity: NIR wrong & visible right {pct(c['nir_wrong_visible_right'])}, NIR right & visible wrong "
                  f"{pct(c['nir_right_visible_wrong'])}, both wrong {pct(c['both_wrong'])}")
            print("\n  band alignment, 50 apples (phase correlation of gradient images; shift in px at 224x224):")
            for k, v in n4["alignment"]["summary"].items():
                print(f"  {k:12s} median {v['median_shift_px224']:5.2f} px  mean {v['mean_shift_px224']:5.2f}  max {v['max_shift_px224']:5.2f}"
                      f"  (mean dx {v['mean_dx']:+.2f}, dy {v['mean_dy']:+.2f})")
            summary["n4"] = n4
    text = buf.getvalue()
    print(text)
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write(text)
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"), indent=1)
    print(f"saved -> {REPORT}\nsaved -> {SUMMARY}")


if __name__ == "__main__":
    main()
