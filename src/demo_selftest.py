"""Tests of the demo app against the REAL confirmatory checkpoints (run after run_confirm finishes).

  python -m src.demo_selftest            GPU if available
  set CUDA_VISIBLE_DEVICES=-1 & python -m src.demo_selftest --cpu-only-quick    CPU check (one image)

Checks:
  1. mode 1 on sample images (Healthy / Bruise / Russet): only the 3 seeds of the image's own test fold
     are loaded, the fold equals fold_assignment.json, the decision equals the one recorded in
     confirm_results.json (mean of the 3 seeds' TTA probabilities)
  2. belt mode (50 images): same fold rule + decisions agree with the recorded predictions
  3. mode 2 upload (npy of a CAPA cube and 4 tif bands) runs with all 15 models
  4. the real Gradio server starts on 127.0.0.1 only and its API answers (known / belt / compare_all)
  5. model switch (N7): for every model with 15 checkpoints (nir_ss, visible_ss, full_ss), 30 random images:
     each seed's decision == tta_test_predictions recorded in the result json, and the 3-seed vote == recorded vote;
     "compare all models" agrees with those votes; presentation examples match their group definition
  6. G1/G2 Kleynen vs CNN step by step: the 12 Kleynen GUI models (5 folds + all, 2 band sets) exist and each fold model
     was tested on exactly the images of that fold; presentation examples cover sound/stem/calyx/bruise/russet/rot and
     match their rule; on every example both methods use only the image's own fold, Kleynen decision/posterior == K3,
     CNN 2-group decision == K2; band-set switch; upload path (all-data Kleynen model + 15 CNN models)
  7. G3 batch: "all" reproduces K5 (526 apples: Kleynen 82.3 % / CNN 93.2 %; 750/800 85.2 %), every image's decisions
     == K3 / K2, subsets (fold, severity) consistent, CSV export and disagreement gallery
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

from . import demo_app as d
from .config import CLASSES, OUTPUT_DIR

RESULTS = os.path.join(OUTPUT_DIR, "dl", "confirm_results.json")
ok = True


def check(name, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)


def recorded(runs, pool_names, rawid, variant="nir_ss"):
    """Recorded by the training run for this image: {seed: TTA decision}, and the decision of the 3-seed vote
    (argmax of the mean of the 3 seeds' TTA probs)."""
    probs, per_seed = [], {}
    for key, r in runs.items():
        if r["variant"] != variant:
            continue
        names = [pool_names[i] for i in r["test_idx"]]
        if rawid in names:
            j = names.index(rawid)
            probs.append(np.array(r["tta_test_probs"])[j])
            per_seed[r["seed"]] = int(r["tta_test_predictions"][j])
    assert len(probs) == 3, f"expected 3 recorded runs for {rawid}, found {len(probs)}"
    return per_seed, int(np.mean(probs, axis=0).argmax())


def recorded_decision(runs, pool_names, rawid, variant="nir_ss"):
    return recorded(runs, pool_names, rawid, variant)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-only-quick", action="store_true")
    args = ap.parse_args()

    demo = d.Demo()
    print(f"device: {d.DEVICE} | checkpoints found: {len(demo.available())}/15")
    check("all 15 checkpoints present", len(demo.available()) == 15)
    runs = json.load(open(RESULTS, encoding="utf-8"))["runs"]
    from .round2 import Context
    ctx = Context()
    pool_names = [os.path.basename(fp).replace(".npy", "") for fp, _ in ctx.pool_raw]

    samples = {}
    for want in ("Healthy", "Bruise", "Russet"):
        samples[want] = next(k for k in sorted(demo.images) if demo.images[k]["tissue"] == want)
    if args.cpu_only_quick:
        samples = {"Bruise": samples["Bruise"]}

    print("1) mode 1 on sample images")
    for want, rawid in samples.items():
        demo._cache.clear()
        info = demo.images[rawid]
        out = d.classify_known(demo, rawid)
        folds_loaded = {f for (_, _, f) in demo._cache}
        seeds_loaded = sorted(s for (_, s, _) in demo._cache)
        _, rows, p, pred = demo.predict_known(rawid)
        check(f"{rawid} ({want}, true {info['label']}): loaded only fold {info['fold']} seeds {seeds_loaded}",
              folds_loaded == {info["fold"]} and seeds_loaded == [3, 4, 5])
        check(f"{rawid}: all checkpoints' meta fold == fold_assignment fold", all(demo.get(s, f, v)[1]["fold"] == info["fold"]
                                                                                for (v, s, f) in demo._cache))
        rec = recorded_decision(runs, pool_names, rawid)
        check(f"{rawid}: demo decision {CLASSES[pred]} == recorded decision {CLASSES[rec]}", pred == rec,
              f"| probs {np.round(p, 3).tolist()} | truth {info['label']} -> {'correct' if CLASSES[pred] == info['label'] else 'WRONG'}")
        check(f"{rawid}: UI outputs well-formed", out[0].shape == (448, 448, 3) and out[5].shape == (448, 448, 3)
              and len(out[4]) == 4 and abs(sum(out[3].values()) - 1) < 1e-6)
        cv2.imwrite(os.path.join(d._CACHE, f"selftest_{rawid}.png"), cv2.cvtColor(out[5], cv2.COLOR_RGB2BGR))
    if args.cpu_only_quick:
        print("CPU quick check", "OK" if ok else "FAILED")
        sys.exit(0 if ok else 1)

    print("2) belt mode (50 images, seed 0)")
    demo._cache.clear()
    summary, cm_rows, wrong = d.run_belt(demo, 50, 0)
    chosen = d.random.Random(0).sample(sorted(demo.images), 50)
    mism = 0
    for rawid in chosen:
        _, _, p, pred = demo.predict_known(rawid)
        mism += int(pred != recorded_decision(runs, pool_names, rawid))
    cm = np.array([[r[1], r[2], r[3]] for r in cm_rows])
    check("confusion matrix sums to 50", cm.sum() == 50, f"| accuracy {np.trace(cm) / 50 * 100:.1f}%")
    check("belt decisions agree with the recorded decisions", mism == 0, f"({mism} of 50 differ)")
    check("belt used every fold's own models only", all(f in range(5) for (_, _, f) in demo._cache))
    print("   misclassified:", [(w[0], w[1], w[2], '->', w[3]) for w in wrong])

    print("3) mode 2 (upload)")
    rawid = samples["Bruise"]
    npy = os.path.join(d._CACHE, "selftest_upload.npy")
    np.save(npy, np.load(demo.paths[rawid]))
    out = d.classify_upload(demo, [npy])
    check("npy upload uses all 15 models", len(out[4]) == 16, f"(table rows {len(out[4])} = 15 models + vote)")
    tifs = []
    cube = np.load(demo.paths[rawid])
    for i, b in enumerate(("BL", "GR", "RE", "IR")):
        fp = os.path.join(d._CACHE, f"selftest_{b}.tif")
        cv2.imwrite(fp, cube[..., i])
        tifs.append(fp)
    out2 = d.classify_upload(demo, tifs)
    check("4 tif bands give the same vote as the npy", out2[3] == out[3])

    print("4) real Gradio server on 127.0.0.1")
    ui = d.build_ui(demo)
    app, url, _ = ui.queue().launch(server_name="127.0.0.1", server_port=7861, share=False, prevent_thread_lock=True,
                                    inbrowser=False, allowed_paths=[d._CACHE])
    try:
        check("server bound to 127.0.0.1 only", url.startswith("http://127.0.0.1"), url)
        from gradio_client import Client
        client = Client(url, verbose=False)
        r = client.predict(samples["Russet"], api_name="/predict_known")
        check("API predict_known answers", r is not None and len(r) == 7)
        r = client.predict(20, 1, api_name="/belt")
        check("API belt answers", r is not None and len(r) == 3)
        r = client.predict(samples["Bruise"], api_name="/compare_all")
        check("API compare_all answers", r is not None and len(r) == len(d.MODELS) + 2)
        r = client.predict(samples["Bruise"], "450/750/800 (paper)", api_name="/kleynen_vs_cnn")
        check("API kleynen_vs_cnn (G2) answers", r is not None and len(r) == 10)
        r = client.predict("tissue:Rot", "450/750/800 (paper)", api_name="/batch_compare")
        # gr.State outputs are not returned by the API: 9 outputs - 1 State = 8
        check("API batch_compare (G3) answers", r is not None and len(r) == 8 and r[-1]["n"] == 23, f"(n = {r[-1]['n']})")
    finally:
        ui.close()

    print("5) model switch (N7)")
    variants = demo.variants()
    check("all 3 models have 15 checkpoints", variants == list(d.MODELS), f"{variants}")
    pick = d.random.Random(0).sample(sorted(demo.images), 30)
    votes = {}
    for v in variants:
        vruns = json.load(open(os.path.join(OUTPUT_DIR, "dl", d.MODELS[v]["results"]), encoding="utf-8"))["runs"]
        seed_mism = vote_mism = 0
        for rawid in pick:
            _, rows, p, pred = demo.predict_known(rawid, v)
            per_seed, vote = recorded(vruns, pool_names, rawid, v)
            seed_mism += sum(int(r["probs"].argmax()) != per_seed[r["seed"]] for r in rows)
            vote_mism += int(pred != vote)
            votes[(v, rawid)] = pred
            assert all(r["fold"] == demo.images[rawid]["fold"] for r in rows)
        check(f"{v}: per-seed decisions == recorded tta_test_predictions", seed_mism == 0, f"({seed_mism} of {3 * len(pick)} differ)")
        check(f"{v}: 3-seed vote == recorded vote", vote_mism == 0, f"({vote_mism} of {len(pick)} differ)")
        s = demo.model_summary(v)
        print(f"   {v}: out-of-fold accuracy {s['accuracy'] * 100:.1f}%, macro-F1 {s['macro_f1']:.3f}")
    out = d.classify_known(demo, pick[0], "visible_ss")
    check("classify_known with visible_ss is well-formed", out[0].shape == (448, 448, 3) and "Visible" in out[6])
    cmp = d.compare_all(demo, pick[0])
    check("compare_all: one row per model, decisions == the votes above",
          [t["model"] for t in cmp[-1]] == variants and all(CLASSES.index(t["pred"]) == votes[(t["model"], pick[0])] for t in cmp[-1]))
    check("compare_all: one Grad-CAM per model", all(c is not None and c.shape == (336, 336, 3) for c in cmp[1:-1]))
    ex = demo.examples()
    oof = demo.oof()
    good = lambda v, k: d.CLASSES[int(oof[v][k].argmax())] == demo.images[k]["label"]
    rules = {"(ก)": lambda k: demo.images[k]["tissue"] == "Bruise" and good("nir_ss", k) and not good("visible_ss", k),
             "(ข)": lambda k: not any(good(v, k) for v in oof),
             "(ค)": lambda k: good("visible_ss", k) and not good("nir_ss", k)}
    check("examples: all 3 groups present and every example matches its rule",
          {t[:3] for t, _ in ex} == set(rules) and all(rules[t[:3]](k) for t, k in ex), f"({len(ex)} examples)")
    for t, k in ex:
        print("   ", t)

    g_sections(demo, pool_names)
    print("DEMO SELFTEST", "ALL PASSED" if ok else "HAD FAILURES")
    sys.exit(0 if ok else 1)


def g_sections(demo, pool_names):
    from . import demo_kleynen as G
    from . import kleynen_models as KM
    dl = os.path.join(OUTPUT_DIR, "dl")
    k3 = json.load(open(os.path.join(dl, "k3_kleynen.json"), encoding="utf-8"))["predictions"]
    k2 = json.load(open(os.path.join(dl, "k2_merged.json"), encoding="utf-8"))["predictions"]["nir_ss (NIR, SS)"]
    rec_k = {bs: dict(zip(pool_names, k3[f"{bs}|A_paper"])) for bs in KM.GUI_SETS}
    rec_kp = {bs: dict(zip(pool_names, k3[f"{bs}|A_paper|p_defect"])) for bs in KM.GUI_SETS}
    rec_c = dict(zip(pool_names, k2))

    print("6) G1/G2 Kleynen vs CNN step by step")
    check("all 12 Kleynen GUI models present", KM.all_present())
    fold_ok = True
    for bs in KM.GUI_SETS:
        for f in range(5):
            m = KM.load_model(bs, f)
            fold_ok &= sorted(m["test_images"]) == sorted(k for k, v in demo.images.items() if v["fold"] == f) and m["erode_px"] == 5
        fold_ok &= KM.load_model(bs)["fold"] is None
    check("each Kleynen fold model was tested on exactly the images of its fold (fold_assignment.json), erosion 5 px", fold_ok)
    cmp = G.Compare(demo)
    check("severity labels for all 246 defective images", len(cmp.severity) == 246 and
          all((k in cmp.severity) == (v["defect_no"] is not None) for k, v in demo.images.items()))
    ex = cmp.examples()
    tissues = {demo.images[k]["tissue"] for _, k in ex}
    check("examples cover sound / stem / calyx / bruise / russet / rot", tissues == set(G.EXAMPLE_TISSUES), f"({len(ex)} examples)")
    grp = lambda k: int(demo.images[k]["label"] != "Healthy")
    rule = {"ทั้งคู่ถูก": lambda k: rec_k[KM.DEFAULT_SET][k] == grp(k) and rec_c[k] == grp(k),
            "CNN ถูก / Kleynen ผิด": lambda k: rec_c[k] == grp(k) and rec_k[KM.DEFAULT_SET][k] != grp(k),
            "Kleynen ถูก / CNN ผิด": lambda k: rec_k[KM.DEFAULT_SET][k] == grp(k) and rec_c[k] != grp(k)}
    check("every example matches its rule (recorded K3 / K2 decisions)", all(rule[t.split(" — ")[1]](k) for t, k in ex))
    bad, folds_bad, form_bad, times = [], 0, 0, []
    for _, rawid in ex:
        demo._cache.clear()
        cmp._models.clear()
        out = G.step_known(cmp, rawid)
        st = out[9]
        f = demo.images[rawid]["fold"]
        folds_bad += int({ff for (_, _, ff) in demo._cache} != {f} or {ff for (_, ff) in cmp._models} != {f}
                         or st["kleynen"]["fold"] != f or st["cnn"]["folds"] != [f])
        if (st["kleynen"]["pred"] != rec_k[KM.DEFAULT_SET][rawid] or abs(st["kleynen"]["p_defect"] - rec_kp[KM.DEFAULT_SET][rawid]) > 1e-5
                or st["cnn"]["pred"] != rec_c[rawid]):
            bad.append(rawid)
        form_bad += int(not (out[2].shape == out[3].shape == (430, 560, 3) and out[6].shape == (448, 448, 3) and len(out[1]) == 4
                             and abs(sum(out[7].values()) - 1) < 1e-6 and st["truth"]["severity"] == cmp.severity.get(rawid, "")))
        times.append((st["kleynen"]["ms"], st["cnn"]["ms"]))
        print(f"   {rawid} {demo.images[rawid]['tissue']:8s} Kleynen {st['kleynen']['pred']} ({st['kleynen']['ms']:.0f} ms)  "
              f"CNN {st['cnn']['pred']} ({st['cnn']['ms']:.0f} ms)  truth {st['truth']['group']}")
    check("G2: both methods use only the image's own test fold", folds_bad == 0)
    check("G2: Kleynen decision + posterior == K3, CNN 2-group decision == K2 on every example", not bad, f"{bad}")
    check("G2: outputs well-formed (4 bands, mask, defect map, Grad-CAM, 3-class probs, severity)", form_bad == 0)
    check("G2: processing times measured", all(a > 0 and b > 0 for a, b in times),
          f"| median Kleynen {np.median([a for a, _ in times]):.0f} ms, CNN {np.median([b for _, b in times]):.0f} ms")
    rawid = ex[0][1]
    st = G.step_known(cmp, rawid, "750/800 (NIR)")[9]
    check("G2: band-set switch 750/800 == K3 750/800", st["kleynen"]["band_set"] == "750/800 (NIR)" and
          st["kleynen"]["pred"] == rec_k["750/800 (NIR)"][rawid] and abs(st["kleynen"]["p_defect"] - rec_kp["750/800 (NIR)"][rawid]) < 1e-5)
    npy = os.path.join(d._CACHE, "selftest_upload_full.npy")
    np.save(npy, KM.load_ss_bands("RawA0004"))
    out = G.step_upload(cmp, [npy])
    check("G2 upload: all-data Kleynen model + 15 CNN models, no answer key",
          out[9]["kleynen"]["fold"] is None and out[9]["cnn"]["n_models"] == 15 and out[9]["truth"] is None)
    os.remove(npy)

    print("7) G3 batch (all 819 images, out-of-fold; first run ~3-4 min)")
    r = G.run_batch(cmp, "all")
    stt = r[8]
    check("G3 'all' == K5 on 526 apples (Kleynen 450/750/800 vs CNN NIR)", stt["k5_check"]["match"],
          f"| Kleynen {stt['kleynen']['acc_526_subset'] * 100:.1f}% CNN {stt['cnn']['acc_526_subset'] * 100:.1f}% "
          f"(K5 {stt['k5_check']['kleynen_k5'] * 100:.1f}% / {stt['k5_check']['cnn_k5'] * 100:.1f}%)")
    km = sum(cmp._kle[(KM.DEFAULT_SET, k)]["pred"] != rec_k[KM.DEFAULT_SET][k] for k in demo.images)
    cm_ = sum(cmp._cnn[k]["pred"] != rec_c[k] for k in demo.images)
    check("G3: every image's Kleynen decision == K3 and CNN decision == K2", km == 0 and cm_ == 0, f"({km} / {cm_} of 819 differ)")
    check("G3: 2x2 confusions sum to 819 and match K2 for the CNN",
          np.sum(stt["kleynen"]["confusion_2x2"]) == 819 and stt["cnn"]["confusion_2x2"] == [[551, 22], [27, 219]])
    check("G3: CSV export has one row per image", os.path.exists(r[7]) and sum(1 for _ in open(r[7], encoding="utf-8-sig")) == 820)
    check("G3: gallery = disagreements only", len(r[4]) == min(stt["n_disagree"], G.GALLERY_MAX) and
          all(cmp._kle[(KM.DEFAULT_SET, k)]["pred"] != cmp._cnn[k]["pred"] for k in r[6]))
    r2 = G.run_batch(cmp, "all", "750/800 (NIR)")
    check("G3 'all' 750/800 == K5", r2[8]["k5_check"]["match"], f"| {r2[8]['kleynen']['acc_526_subset'] * 100:.1f}%")
    k5 = json.load(open(os.path.join(dl, "k5_summary.json"), encoding="utf-8"))["rows"]
    rs = G.run_batch(cmp, "sev:slight")[8]
    ref = lambda row: k5[row]["paper_subset_280_246"]["slight_pct"]
    check("G3 subset severity slight: n = 76 and rates == K5",
          rs["n"] == 76 and abs(rs["per_group"][G.SEV_TH["slight"]]["kleynen"] * 100 - ref(G.kleynen_row(KM.DEFAULT_SET))) < 1e-6
          and abs(rs["per_group"][G.SEV_TH["slight"]]["cnn"] * 100 - ref("CNN merged NIR (nir_ss)")) < 1e-6)
    rf = G.run_batch(cmp, "fold:2")[8]
    check("G3 subset fold 2: n == images of fold 2", rf["n"] == sum(v["fold"] == 2 for v in demo.images.values()))


if __name__ == "__main__":
    main()
