"""Compare the recorded K3 run with the features-only ablation without changing K5."""
import json
import os

from .config import OUTPUT_DIR


def report(original, features_only):
    bs = "450/750/800 (paper)"
    def row(data):
        result = data["band_sets"][bs]["A_paper"]
        return result["paper_subset_280_246"], result["groups"]["stem+calyx"]

    old, old_stem = row(original)
    new, new_stem = row(features_only)
    paper = original.get("paper", {"sound": 94.3, "defective": 84.6})
    old_erosion = original.get("erode_px")
    if old_erosion is None:
        assert "erosion 5px" in original["method"]["fruit_mask"]
    else:
        assert old_erosion == 5
    assert features_only["erode_px"] == 5
    assert features_only["method"]["feature_variant"] == "paper_described_only"

    lines = [
        "# K3 feature ablation (450/750/800 nm)",
        "",
        "Both local rows use the same 5 out-of-fold splits and 526-image subset "
        "(280 sound + 246 defective). Published percentages are reference values, "
        "not paired out-of-fold measurements from this experiment.",
        "",
        "| Method | Sound (n=280) | Defective (n=246) | Accuracy (n=526) | Stem/calyx (n=293; supplementary) |",
        "|---|---:|---:|---:|---:|",
        f"| Kleynen 2005 (published) | {paper['sound']:.1f}% | {paper['defective']:.1f}% | "
        f"{(paper['sound'] * 280 + paper['defective'] * 246) / 526:.1f}% | separate detector |",
        f"| Local K3 enriched (recorded) | {old['sound_pct']:.1f}% | {old['defective_pct']:.1f}% | "
        f"{100 * old['accuracy']:.1f}% | {old_stem['pct']:.1f}% |",
        f"| Local K3 paper-described fruit features only (new) | {new['sound_pct']:.1f}% | "
        f"{new['defective_pct']:.1f}% | {100 * new['accuracy']:.1f}% | {new_stem['pct']:.1f}% |",
        "",
        "The new row removes only the two added region features. It is NOT an exact "
        "paper replication: 64-bin histogram, Otsu/erosion settings, whole SBW masks, "
        "cross-fitting and no separate stem/calyx matcher still differ. "
        "Do not replace the original K3/K5 figures with this exploratory ablation.",
        "",
    ]
    return "\n".join(lines)


def main():
    dl = os.path.join(OUTPUT_DIR, "dl")
    with open(os.path.join(dl, "k3_kleynen.json"), encoding="utf-8") as f:
        original = json.load(f)
    with open(os.path.join(dl, "k3_kleynen_features_only.json"), encoding="utf-8") as f:
        features_only = json.load(f)
    target = os.path.join(OUTPUT_DIR, "thesis", "k3_feature_ablation.md")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(report(original, features_only))
    print(target)


if __name__ == "__main__":
    main()
