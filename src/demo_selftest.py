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
    print("DEMO SELFTEST", "ALL PASSED" if ok else "HAD FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
