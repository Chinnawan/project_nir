"""Tests of the demo app against the REAL confirmatory checkpoints (run after run_confirm finishes).

  python -m src.demo_selftest            GPU if available
  set CUDA_VISIBLE_DEVICES=-1 & python -m src.demo_selftest --cpu-only-quick    CPU check (one image)

Checks:
  1. mode 1 on sample images (Healthy / Bruise / Russet): only the 3 seeds of the image's own test fold
     are loaded, the fold equals fold_assignment.json, the decision equals the one recorded in
     confirm_results.json (mean of the 3 seeds' TTA probabilities)
  2. belt mode (50 images): same fold rule + decisions agree with the recorded predictions
  3. mode 2 upload (npy of a CAPA cube and 4 tif bands) runs with all 15 models
  4. the real Gradio server starts on 127.0.0.1 only and its API answers (known / belt / upload)
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


def recorded_decision(runs, pool_names, rawid):
    """Decision recorded by run_confirm for this image: argmax of the mean of the 3 seeds' TTA probs."""
    probs = []
    for key, r in runs.items():
        if r["variant"] != "nir_ss":
            continue
        names = [pool_names[i] for i in r["test_idx"]]
        if rawid in names:
            probs.append(np.array(r["tta_test_probs"])[names.index(rawid)])
    assert len(probs) == 3, f"expected 3 recorded runs for {rawid}, found {len(probs)}"
    return int(np.mean(probs, axis=0).argmax())


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
        folds_loaded = {f for (_, f) in demo._cache}
        seeds_loaded = sorted(s for (s, _) in demo._cache)
        _, rows, p, pred = demo.predict_known(rawid)
        check(f"{rawid} ({want}, true {info['label']}): loaded only fold {info['fold']} seeds {seeds_loaded}",
              folds_loaded == {info["fold"]} and seeds_loaded == [3, 4, 5])
        check(f"{rawid}: all checkpoints' meta fold == fold_assignment fold", all(demo.get(s, f)[1]["fold"] == info["fold"]
                                                                                for (s, f) in demo._cache))
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
    check("belt used every fold's own models only", all(f in range(5) for (_, f) in demo._cache))
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
    finally:
        ui.close()
    print("DEMO SELFTEST", "ALL PASSED" if ok else "HAD FAILURES")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
