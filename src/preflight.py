"""Read-only checklist for a CAPA evaluation/demo machine."""
import importlib.util
import os

from .config import OUTPUT_DIR, PROJECT_ROOT


def status(root=PROJECT_ROOT):
    checks = {
        "CAPA images": os.path.isdir(os.path.join(root, "images", "SingleSpectraImages")),
        "defect masks": os.path.isdir(os.path.join(root, "segmentations", "Segmented_defects")),
        "processed raw": os.path.isdir(os.path.join(root, "Processed_Dataset")),
        "processed aligned": os.path.isdir(os.path.join(root, "Processed_Dataset_SS")),
        "SS mapping": os.path.isfile(os.path.join(root, "outputs", "ss_mapping.json")),
        "K3 recorded": os.path.isfile(os.path.join(root, "outputs", "dl", "k3_kleynen.json")),
        "CNN checkpoints (15 NIR)": all(os.path.isfile(os.path.join(
            root, "outputs", "dl", "confirm_ckpt", f"nir_ss_seed{s}_fold{f}.pt"))
            for s in (3, 4, 5) for f in range(5)),
        "Kleynen GUI models (10 folds + 2 all)": all(os.path.isfile(os.path.join(
            root, "models", f"kleynen_{band}_{'all' if f is None else f'fold{f}'}.pkl"))
            for band in ("450_750_800", "750_800") for f in (*range(5), None)),
        "Python dependencies": all(importlib.util.find_spec(m) is not None
                                   for m in ("numpy", "cv2", "scipy", "sklearn", "torch", "gradio")),
    }
    return checks


def main():
    print("CAPA_PROJECT_ROOT:", PROJECT_ROOT)
    print("OUTPUT_DIR:", OUTPUT_DIR)
    archive = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "CAPA_AppleQualityGradingImageDatabase.rar")
    if os.path.isfile(archive):
        print("ARCHIVE FOUND (not extracted):", archive)
    for name, ready in status().items():
        print(f"{'OK' if ready else 'MISSING'}: {name}")
    print("Consent is not checked by this script; confirm permission separately before using CAPA.")


if __name__ == "__main__":
    main()
