"""Read-only consistency checks for the recorded CAPA/Kleynen experiments."""
import collections
import json
import os

from .config import OUTPUT_DIR


def audit(dl):
    def load(name):
        with open(os.path.join(dl, name), encoding="utf-8") as f:
            return json.load(f)

    fold = load("fold_assignment.json")["images"]
    k1, k2, k3, k5 = (load(f) for f in (
        "k1_severity.json", "k2_merged.json", "k3_kleynen.json", "k5_summary.json"))
    gui = load("g1_kleynen_models.json")
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    check(len(fold) == 819, "fold assignment must cover 819 images")
    counts = collections.Counter(info["tissue"] for info in fold.values())
    check((counts["Healthy"], counts["Stem"], counts["Calix"]) == (280, 148, 145),
          "sound/stem/calyx counts differ from CAPA records")
    check(sum(v for k, v in counts.items() if k not in ("Healthy", "Stem", "Calix")) == 246,
          "defective count differs from CAPA records")
    check(k1["n"] == 246 and sum(k1["counts"].values()) == 246, "severity count is not 246")
    check(len({info["fold"] for info in fold.values()}) == 5, "five test folds expected")

    for band in ("450/750/800 (paper)", "750/800 (NIR)", "450/500 (Visible)"):
        for setting, label in (("A_paper", "A: paper protocol"), ("B_all", "B: stem/calyx = sound")):
            name = f"Kleynen replicated {band} [{label}]"
            source = k3["band_sets"][band][setting]
            dest = k5["rows"][name]
            for metric in ("accuracy", "paper_subset_280_246", "confusion_2x2", "groups", "per_severity"):
                check(source[metric] == dest[metric], f"{name}: {metric} differs between K3 and K5")
            check(len(k3["predictions"][f"{band}|{setting}"]) == 819,
                  f"{name}: predictions do not cover 819 images")
    for name, source in (("CNN merged NIR (nir_ss)", "nir_ss (NIR, SS)"),
                         ("CNN merged Visible (visible_ss)", "visible_ss (Visible, SS)")):
        check(k2["models"][source] == k5["rows"][name], f"{name}: K2 and K5 differ")
        check(len(k2["predictions"][source]) == 819, f"{name}: predictions do not cover 819 images")

    for name, row in k5["rows"].items():
        cm = row["confusion_2x2"]["matrix"]
        check(sum(map(sum, cm)) == 819, f"{name}: confusion total is not 819")
        check(abs((cm[0][0] + cm[1][1]) / 819 - row["accuracy"]) < 1e-10,
              f"{name}: accuracy differs from confusion matrix")
        subset = row["paper_subset_280_246"]
        check(abs((subset["sound_pct"] * 280 + subset["defective_pct"] * 246) / 52600
                  - subset["accuracy"]) < 1e-10, f"{name}: 526-image accuracy differs from group rates")

    for band, recorded in gui["band_sets"].items():
        check(recorded["fold_models_reproduce_k3_decisions"]
              and recorded["max_posterior_diff_vs_k3"] < 1e-5,
              f"{band}: saved GUI model verification failed")
        expected = k3["band_sets"][band]["A_paper"]["paper_subset_280_246"]["accuracy"]
        check(abs(recorded["acc_526_from_fold_models"] - expected) < 1e-10,
              f"{band}: saved GUI accuracy differs from K3")

    for filename, variant in (("confirm_results.json", "nir_ss"),
                              ("n5_results.json", "visible_ss"),
                              ("k4_results.json", "nir_ss_2cls")):
        runs = [r for r in load(filename)["runs"].values()
                if r["variant"] == variant and r["seed"] in (3, 4, 5)]
        seen = collections.Counter(i for r in runs for i in r["test_idx"])
        check(len(runs) == 15 and len(seen) == 819 and all(v == 3 for v in seen.values()),
              f"{filename}: expected 15 runs and 3 test predictions per image")

    return issues


def main():
    issues = audit(os.path.join(OUTPUT_DIR, "dl"))
    if issues:
        for issue in issues:
            print("FAIL:", issue)
        raise SystemExit(1)
    print("PASS: recorded K1/K2/K3/K5 counts, fold coverage, and saved GUI verification agree.")
    print("This does not rerun models, inspect source images, or confirm paper fidelity/consent.")


if __name__ == "__main__":
    main()
