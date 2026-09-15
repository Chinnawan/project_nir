"""Phase 5: Grad-CAM visualization (does the model actually look at the
defect?) and statistical significance testing (bootstrap CI + McNemar's test)
across all trained model variants (classical + DL).
"""
import itertools
import json
import os

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

from .config import CHANNEL_GROUPS, CLASSES, OUTPUT_DIR, SEED
from .dataset import ApplesDataset, compute_channel_stats
from .models import build_model

DL_OUT = os.path.join(OUTPUT_DIR, "dl")
CLASSICAL_OUT = os.path.join(OUTPUT_DIR, "classical")
STATS_OUT = os.path.join(OUTPUT_DIR, "stats")
GRADCAM_OUT = os.path.join(DL_OUT, "gradcam")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def _rgb_preview(raw_uint8):
    """raw_uint8: (H,W,4) in BL,GR,RE,IR order -> pseudo-RGB (H,W,3) float in [0,1]
    using R=RE, G=GR, B=BL so the false-color image roughly separates NIR info
    (RE) into the red channel."""
    rgb = raw_uint8[:, :, [2, 1, 0]].astype(np.float32) / 255.0
    return np.clip(rgb, 0, 1)


def run_gradcam(model_name="full_SE", n_per_class=3):
    os.makedirs(GRADCAM_OUT, exist_ok=True)
    channel_group = "full"
    use_se = model_name.endswith("_SE")

    model = build_model(channel_group, num_classes=len(CLASSES), pretrained=False, use_se=use_se).to(DEVICE)
    state = torch.load(os.path.join(DL_OUT, f"{model_name}_best.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    target_layer = model.backbone.features[-1] if use_se else model.features[-1]
    cam_algo = GradCAM(model=model, target_layers=[target_layer])

    mean, std = compute_channel_stats()
    ds = ApplesDataset("test", channel_group, augment=False, mean=mean, std=std)

    per_class_saved = {c: 0 for c in range(len(CLASSES))}
    saved_paths = []
    for idx, (fp, label) in enumerate(ds.items):
        if per_class_saved[label] >= n_per_class:
            continue
        x, y = ds[idx]
        x_batch = x.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = model(x_batch).argmax(1).item()

        grayscale_cam = cam_algo(input_tensor=x_batch, targets=[ClassifierOutputTarget(label)])[0]

        raw = np.load(fp)
        rgb_img = _rgb_preview(raw)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fname = os.path.basename(fp).replace(".npy", "")
        correct = "correct" if pred == label else "WRONG"
        out_path = os.path.join(GRADCAM_OUT, f"{CLASSES[label]}_{fname}_{correct}.png")

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(rgb_img)
        axes[0].set_title(f"input ({CLASSES[label]})")
        axes[0].axis("off")
        axes[1].imshow(visualization)
        axes[1].set_title(f"Grad-CAM pred={CLASSES[pred]}")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        saved_paths.append(out_path)
        per_class_saved[label] += 1
        if all(v >= n_per_class for v in per_class_saved.values()):
            break

    print(f"Saved {len(saved_paths)} Grad-CAM visualizations to {GRADCAM_OUT}")
    return saved_paths


# ---------------------------------------------------------------------------
# Statistical significance testing
# ---------------------------------------------------------------------------

def bootstrap_macro_f1_ci(y_true, y_pred, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro")
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(scores.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def mcnemar_test(y_true, pred_a, pred_b):
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    both_correct = int(np.sum(a_correct & b_correct))
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    neither = int(np.sum(~a_correct & ~b_correct))
    table = [[both_correct, a_only], [b_only, neither]]
    result = mcnemar(table, exact=(a_only + b_only) < 25)
    return {"table": table, "statistic": float(result.statistic), "p_value": float(result.pvalue)}


def collect_predictions():
    """Gather (name -> (y_true, y_pred)) for every trained model, classical + DL.
    Relies on classical features and DL test loaders being built from the same
    sorted file listing (src.config.list_files), so label order lines up."""
    preds = {}

    classical_path = os.path.join(CLASSICAL_OUT, "predictions.json")
    if os.path.exists(classical_path):
        with open(classical_path) as f:
            data = json.load(f)
        y_true = data["y_test"]
        for name, y_pred in data.items():
            if name == "y_test":
                continue
            preds[name] = (y_true, y_pred)

    dl_path = os.path.join(DL_OUT, "metrics.json")
    if os.path.exists(dl_path):
        with open(dl_path) as f:
            data = json.load(f)
        for name, r in data.items():
            preds[name] = (r["test_labels"], r["test_predictions"])

    return preds


def run_statistical_tests():
    os.makedirs(STATS_OUT, exist_ok=True)
    preds = collect_predictions()
    if len(preds) < 2:
        print("Not enough trained models with saved predictions to compare yet.")
        return None

    results = {"bootstrap_ci": {}, "mcnemar_pairwise": {}}

    for name, (y_true, y_pred) in preds.items():
        results["bootstrap_ci"][name] = bootstrap_macro_f1_ci(y_true, y_pred)
        print(f"{name:15s} macro-F1 = {results['bootstrap_ci'][name]['mean']:.4f} "
              f"95% CI [{results['bootstrap_ci'][name]['ci_lo']:.4f}, {results['bootstrap_ci'][name]['ci_hi']:.4f}]")

    names = list(preds.keys())
    for a, b in itertools.combinations(names, 2):
        y_true_a, pred_a = preds[a]
        y_true_b, pred_b = preds[b]
        if list(y_true_a) != list(y_true_b):
            continue  # different test-set label order/content, not directly comparable
        key = f"{a}_vs_{b}"
        results["mcnemar_pairwise"][key] = mcnemar_test(y_true_a, pred_a, pred_b)
        p = results["mcnemar_pairwise"][key]["p_value"]
        sig = "significant (p<0.05)" if p < 0.05 else "not significant"
        print(f"McNemar {a} vs {b}: p={p:.4f} -> {sig}")

    with open(os.path.join(STATS_OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved statistical test results to {STATS_OUT}")
    return results


def print_per_class_f1(model_names=("nir", "full_SE"), internal_defect_gap_threshold=0.10):
    """Per-class F1 (from the classification_report already saved in
    outputs/dl/metrics.json) for the given models, plus a Focal Loss
    recommendation: if Internal_Defect (the smallest class, 102 images) has
    F1 more than `internal_defect_gap_threshold` below every other class,
    class imbalance is likely still hurting it and Focal Loss is worth
    trying; otherwise skip it and move on to Phase 6."""
    dl_path = os.path.join(DL_OUT, "metrics.json")
    if not os.path.exists(dl_path):
        print("outputs/dl/metrics.json not found -- run `python -m src.train_dl` first")
        return None

    with open(dl_path) as f:
        metrics = json.load(f)

    verdicts = {}
    print("\n=== Per-class F1 (single-split test set, 123 images) ===")
    for name in model_names:
        if name not in metrics:
            print(f"  {name}: no results found in metrics.json, skipping")
            continue
        report = metrics[name]["test_report"]
        per_class = {c: report[c]["f1-score"] for c in CLASSES}
        print(f"  {name}:")
        for c in CLASSES:
            print(f"    {c:16s} F1 = {per_class[c]:.3f}  (support={int(report[c]['support'])})")

        internal_f1 = per_class["Internal_Defect"]
        gaps = {c: per_class[c] - internal_f1 for c in CLASSES if c != "Internal_Defect"}
        worst_gap = max(gaps.values())
        needs_focal = worst_gap > internal_defect_gap_threshold
        verdicts[name] = {"per_class_f1": per_class, "worst_gap_vs_internal_defect": worst_gap,
                           "recommend_focal_loss": needs_focal}
        verdict_text = (f"Internal_Defect trails the best other class by {worst_gap:.3f} "
                         f"(> {internal_defect_gap_threshold}) -> consider Focal Loss"
                         if needs_focal else
                         f"Internal_Defect is within {worst_gap:.3f} of other classes "
                         f"(<= {internal_defect_gap_threshold}) -> Focal Loss not needed")
        print(f"    -> {verdict_text}")

    os.makedirs(DL_OUT, exist_ok=True)
    with open(os.path.join(DL_OUT, "per_class_f1.json"), "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"\nSaved -> {os.path.join(DL_OUT, 'per_class_f1.json')}")
    return verdicts


def main():
    run_gradcam("full_SE", n_per_class=3)
    run_statistical_tests()
    print_per_class_f1()


if __name__ == "__main__":
    main()
