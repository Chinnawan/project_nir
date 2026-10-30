"""Build Processed_Dataset_SS/ from the flat-field-corrected SingleSpectraImages.

The SingleSpectra images (Healthy/Stem/Calix/Defect NNN_{BL,GR,RE,IR}.tif, 560x430)
are the SAME 819 apples as RawA0001..RawA0819, just flat-field corrected and
cropped. This script mirrors Processed_Dataset/ file-for-file (same
train/val/test split, same RawA####.npy names, same class folders), so the
k-fold fold assignment of every apple is identical between the raw and the
SingleSpectra experiments (paired comparison).

Mapping RawA number -> SingleSpectra name:
  RawFile.txt lists, in RawA order, (tissue type, running number). Healthy/Stem/Calix
  keep their own running number; every other entry is a "Defect" numbered by its
  position among the non-Healthy/Stem/Calix entries (Defects.txt is missing from the
  dataset, so this ordering is verified against notes.txt anchors and, optionally,
  against image content via --verify-content).

Usage:
  python -m src.preprocess_ss                  # build dataset + mapping
  python -m src.preprocess_ss --verify-content # also match SS IR vs raw IR quadrant
"""
import argparse
import collections
import glob
import json
import os
import re

import cv2
import numpy as np

from .config import CLASSES, DATA_DIR, OUTPUT_DIR, PROJECT_ROOT
from .consent import require_consent

SS_DIR = os.path.join(PROJECT_ROOT, "images", "SingleSpectraImages")
RAW_DIR = os.path.join(PROJECT_ROOT, "images", "RawImages")
OUT_DIR = os.path.join(PROJECT_ROOT, "Processed_Dataset_SS")
MAPPING_PATH = os.path.join(OUTPUT_DIR, "ss_mapping.json")
VERIFY_PATH = os.path.join(OUTPUT_DIR, "ss_mapping_verification.json")
BANDS = ["BL", "GR", "RE", "IR"]  # stack order = [0,1,2,3], same as preprocess_pipeline.py

PLAN_B = {
    "Healthy": "Healthy", "Stem": "Healthy", "Calix": "Healthy",
    "Bruise": "Internal_Defect", "Rot": "Internal_Defect", "Flesh_damage": "Internal_Defect",
    "Russet": "Surface_Defect", "Scald": "Surface_Defect", "Hail_damage": "Surface_Defect",
    "Hail_damage_perf": "Surface_Defect", "Frost_damage": "Surface_Defect",
    "Limb_rub": "Surface_Defect", "Other": "Surface_Defect",
}

# notes.txt: "defectNNN = <type> <n>" / "<type>A-B = defectC-D" clues used as anchors
NOTES_ANCHORS = [
    ("Other", 15, 106), ("Other", 16, 107), ("Other", 17, 125),
    ("Bruise", 10, 181), ("Bruise", 55, 226),
    ("Flesh_damage", 5, 227), ("Flesh_damage", 14, 236),
    ("Flesh_damage", 15, 237), ("Flesh_damage", 24, 246),
]


def parse_rawfile():
    require_consent()
    with open(os.path.join(RAW_DIR, "RawFile.txt"), encoding="latin-1") as f:
        txt = f.read()
    rows = [(int(a), b, int(c)) for a, b, c in re.findall(r"RawFile\s+(\d+)\s+(\w+)\s+(\d+)", txt)]
    assert [r for r, _, _ in rows] == list(range(1, 820)), "RawFile.txt is not 1..819 in order"
    return rows


def build_mapping():
    """rawno -> {ss_name, tissue, tissue_no, cls, defect_no}"""
    rows = parse_rawfile()
    mapping, defect_k = {}, 0
    defect_by_type = {}
    for rawno, tissue, n in rows:
        if tissue in ("Healthy", "Stem", "Calix"):
            ss_name, defect_no = f"{tissue}{n:03d}", None
        else:
            defect_k += 1
            ss_name, defect_no = f"Defect{defect_k:04d}", defect_k
            defect_by_type[(tissue, n)] = defect_k
        mapping[rawno] = {"ss_name": ss_name, "tissue": tissue, "tissue_no": n,
                          "cls": PLAN_B[tissue], "defect_no": defect_no}

    for t, n, expected in NOTES_ANCHORS:
        got = defect_by_type.get((t, n))
        assert got == expected, f"notes.txt anchor mismatch: {t} {n} -> defect{got}, expected defect{expected}"
    assert defect_k == 246, f"expected 246 defects, got {defect_k}"
    counts = collections.Counter(m["cls"] for m in mapping.values())
    assert (counts["Healthy"], counts["Internal_Defect"], counts["Surface_Defect"]) == (573, 102, 144), counts
    return mapping


def read_bands(ss_name):
    require_consent()
    bands = []
    for b in BANDS:
        img = cv2.imread(os.path.join(SS_DIR, f"{ss_name}_{b}.tif"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"{ss_name}_{b}.tif")
        assert img.shape == (430, 560), (ss_name, b, img.shape)
        bands.append(img)
    return bands


def pad_square_resize(img, size=224, interpolation=cv2.INTER_AREA):
    """430x560 -> zero-pad top/bottom to 560x560 (keeps the whole apple, same
    transform is applied to the defect masks) -> resize to size x size."""
    h, w = img.shape[:2]
    pad = (w - h) // 2
    padded = cv2.copyMakeBorder(img, pad, w - h - pad, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return cv2.resize(padded, (size, size), interpolation=interpolation)


def build_dataset(mapping):
    """Mirror every file of Processed_Dataset/{split}/{cls}/RawA####.npy."""
    require_consent()
    n_written = 0
    for split in ("train", "val", "test"):
        for cls in CLASSES:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)
            for fp in sorted(glob.glob(os.path.join(DATA_DIR, split, cls, "*.npy"))):
                name = os.path.basename(fp)
                rawno = int(re.search(r"(\d+)", name).group(1))
                m = mapping[rawno]
                assert m["cls"] == cls, f"{name}: existing folder {cls} != mapped class {m['cls']}"
                cube = np.stack([pad_square_resize(b) for b in read_bands(m["ss_name"])], axis=-1)
                np.save(os.path.join(OUT_DIR, split, cls, name), cube)
                n_written += 1
    assert n_written == 819, n_written
    return n_written


def verify_content(mapping, n_distractors=10, seed=42):
    """Independent check of the RawA<->SingleSpectra pairing: the corrected SS IR
    image is a shifted crop of the raw IR quadrant, so normalized cross-correlation
    of SS IR against the raw IR quadrant should be highest for the mapped RawA."""
    require_consent()
    rng = np.random.RandomState(seed)
    scores_mapped, wins = [], 0
    worst = []
    for rawno, m in mapping.items():
        ss_ir = cv2.imread(os.path.join(SS_DIR, f"{m['ss_name']}_IR.tif"), cv2.IMREAD_GRAYSCALE)
        ss_small = cv2.resize(ss_ir, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        def ncc(r):
            raw = cv2.imread(os.path.join(RAW_DIR, f"RawA{r:04d}.tif"), cv2.IMREAD_GRAYSCALE)
            quad = cv2.resize(raw[0:512, 0:640], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            return float(cv2.matchTemplate(quad, ss_small, cv2.TM_CCOEFF_NORMED).max())

        s_map = ncc(rawno)
        others = rng.choice([r for r in range(1, 820) if r != rawno], n_distractors, replace=False)
        s_other = max(ncc(int(r)) for r in others)
        scores_mapped.append(s_map)
        if s_map > s_other:
            wins += 1
        else:
            worst.append({"rawno": rawno, "ss_name": m["ss_name"], "mapped": s_map, "best_distractor": s_other})
    result = {
        "n": len(mapping), "mapped_beats_all_%d_distractors" % n_distractors: wins,
        "fraction": wins / len(mapping), "mapped_score_mean": float(np.mean(scores_mapped)),
        "mapped_score_min": float(np.min(scores_mapped)), "failures": worst[:50],
    }
    with open(VERIFY_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-content", action="store_true")
    args = parser.parse_args()

    mapping = build_mapping()
    print("mapping OK: 9/9 notes.txt anchors match, 246 defects, class counts 573/102/144")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MAPPING_PATH, "w") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f, indent=1)

    if args.verify_content:
        res = verify_content(mapping)
        print(f"content check: mapped RawA wins for {res['mapped_beats_all_10_distractors']}/{res['n']} "
              f"(mean NCC {res['mapped_score_mean']:.3f}, min {res['mapped_score_min']:.3f})")
        if res["fraction"] < 0.99:
            print("WARNING: mapping not confirmed by image content -- see", VERIFY_PATH)
            return

    n = build_dataset(mapping)
    print(f"wrote {n} cubes (224x224x4, BL/GR/RE/IR) to {OUT_DIR}")


if __name__ == "__main__":
    main()
