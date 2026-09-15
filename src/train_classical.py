"""Phase 2: classical ML baseline (SVM, Random Forest, LightGBM) on hand-crafted
features, plus SHAP feature-importance analysis. Answers: is deep learning even
necessary on a dataset this small (~820 images)?
"""
import json
import os

import joblib
import numpy as np
import shap
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CLASSES, OUTPUT_DIR, SEED
from .features import build_feature_dataset, feature_names

OUT = os.path.join(OUTPUT_DIR, "classical")


def load_all_features():
    Xtr, ytr, _ = build_feature_dataset("train")
    Xval, yval, _ = build_feature_dataset("val")
    Xte, yte, _ = build_feature_dataset("test")
    return (Xtr, ytr), (Xval, yval), (Xte, yte)


def build_models():
    return {
        "SVM": SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                   probability=True, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=None,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "LightGBM": LGBMClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=SEED, verbosity=-1),
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "report": classification_report(y, pred, target_names=CLASSES, output_dict=True),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }, pred


def run_shap(name, model, scaler, X_train, X_test, names):
    fig_dir = os.path.join(OUT, "shap")
    os.makedirs(fig_dir, exist_ok=True)
    Xt_test = scaler.transform(X_test)
    Xt_train = scaler.transform(X_train)

    if name in ("RandomForest", "LightGBM"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(Xt_test)
    else:
        background = shap.sample(Xt_train, 50, random_state=SEED)
        subset = Xt_test[:40]
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(subset, nsamples=100)
        Xt_test = subset

    # shap_values can be a list (per-class) or a 3D array depending on model/version
    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=(0, 1))
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            mean_abs = np.abs(arr).mean(axis=(0, 2)) if arr.shape[0] == Xt_test.shape[0] else np.abs(arr).mean(axis=(0, 1))
        else:
            mean_abs = np.abs(arr).mean(axis=0)

    order = np.argsort(mean_abs)[::-1]
    top = [(names[i], float(mean_abs[i])) for i in order[:15]]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    labels = [t[0] for t in top][::-1]
    values = [t[1] for t in top][::-1]
    plt.barh(labels, values, color="#4C72B0")
    plt.xlabel("mean(|SHAP value|)")
    plt.title(f"SHAP feature importance — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{name}_shap_importance.png"), dpi=150)
    plt.close()

    return top


def main():
    os.makedirs(OUT, exist_ok=True)
    (Xtr, ytr), (Xval, yval), (Xte, yte) = load_all_features()
    names = feature_names()

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xval_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xval), scaler.transform(Xte)

    results = {}
    predictions = {"y_test": yte.tolist()}

    for name, model in build_models().items():
        model.fit(Xtr_s, ytr)

        val_metrics, _ = evaluate(model, Xval_s, yval)
        test_metrics, test_pred = evaluate(model, Xte_s, yte)

        print(f"[{name}] val macro-F1={val_metrics['macro_f1']:.4f} "
              f"| test macro-F1={test_metrics['macro_f1']:.4f} acc={test_metrics['accuracy']:.4f}")

        results[name] = {"val": val_metrics, "test": test_metrics}
        predictions[name] = test_pred.tolist()

        joblib.dump(model, os.path.join(OUT, f"{name}_model.joblib"))

        top_shap = run_shap(name, model, scaler, Xtr, Xte, names)
        results[name]["top_shap_features"] = top_shap
        print(f"  top SHAP features: {[f for f, _ in top_shap[:5]]}")

    joblib.dump(scaler, os.path.join(OUT, "scaler.joblib"))

    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(OUT, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)

    best = max(results.items(), key=lambda kv: kv[1]["val"]["macro_f1"])
    print(f"\nBest by val macro-F1: {best[0]} -> test macro-F1={best[1]['test']['macro_f1']:.4f}")
    print(f"Saved metrics/models/SHAP plots to {OUT}")


if __name__ == "__main__":
    main()
