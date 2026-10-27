"""G4: tables and figures for the thesis -> outputs/thesis/  (charts / tables / diagrams only: NO CAPA apple images)

  python -m src.thesis_figures

  k5_table.md / k5_table.csv          the K5 table (paper format: 280 sound + 246 defective; + stem/calyx and all 819)
  fig_severity_accuracy.png/.svg      accuracy per severity: Kleynen paper, Kleynen replicated, CNN NIR, CNN Visible
  fig_nir_vs_visible.png/.svg         NIR vs Visible in the Kleynen method and in the CNN (526 apples) + McNemar
  fig_pipelines.png/.svg              flow diagram of both methods
  g4_numbers.json                     every number drawn, with its source
All numbers are read from outputs/dl/k5_summary.json (K5) and kleynen.PAPER; nothing is recomputed except the McNemar
test of CNN NIR vs CNN Visible on the 526 apples (K5 has it on 819).
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from .config import OUTPUT_DIR  # noqa: E402
from .kleynen import PAPER, SEV_ORDER  # noqa: E402

OUT = os.path.join(OUTPUT_DIR, "thesis")
DL = os.path.join(OUTPUT_DIR, "dl")
# reference palette (dataviz skill), light mode: categorical slots 1-4 in fixed order, surface + text tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ROWS = [  # (table label, K5 row name or None for the paper)
    ("Kleynen 2005 (paper numbers)", None),
    ("Kleynen replicated 450/750/800 [A: paper protocol]", "Kleynen replicated 450/750/800 (paper) [A: paper protocol]"),
    ("Kleynen replicated 750/800 (NIR) [A]", "Kleynen replicated 750/800 (NIR) [A: paper protocol]"),
    ("Kleynen replicated 450/500 (Visible) [A]", "Kleynen replicated 450/500 (Visible) [A: paper protocol]"),
    ("Kleynen replicated 450/750/800 [B: stem/calyx = sound]", "Kleynen replicated 450/750/800 (paper) [B: stem/calyx = sound]"),
    ("Kleynen replicated 450/750/800 [A, erosion 15 px: sensitivity]",
     "Kleynen replicated 450/750/800 (paper) [A, erosion 15px: sensitivity]"),
    ("CNN merged NIR (nir_ss)", "CNN merged NIR (nir_ss)"),
    ("CNN merged Visible (visible_ss)", "CNN merged Visible (visible_ss)"),
    ("CNN 2-class NIR (nir_ss_2cls)", "CNN 2-class NIR (nir_ss_2cls)"),
    ("CNN 2-class Visible (visible_ss_2cls)", "CNN 2-class Visible (visible_ss_2cls)"),
]
COLS = ["sound", "defective", "reject", "bruise", "serious", "slight", "acc_526", "stem_calyx", "acc_819"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(TEXT2)
    ax.tick_params(colors=TEXT2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def table_rows(k5):
    out = []
    for label, key in ROWS:
        if key is None:
            r = {"sound": PAPER["sound"], "defective": PAPER["defective"], **{s: PAPER[s] for s in SEV_ORDER},
                 "acc_526": round((280 * PAPER["sound"] + 246 * PAPER["defective"]) / 526, 1), "stem_calyx": None, "acc_819": None}
        else:
            q, g = k5[key]["paper_subset_280_246"], k5[key]
            r = {"sound": q["sound_pct"], "defective": q["defective_pct"], **{s: q[f"{s}_pct"] for s in SEV_ORDER},
                 "acc_526": q["accuracy"] * 100, "stem_calyx": g["groups"]["stem+calyx"]["pct"], "acc_819": g["accuracy"] * 100}
        out.append((label, {c: (None if v is None else round(float(v), 1)) for c, v in r.items()}))
    return out


def write_tables(rows, mc):
    with open(os.path.join(OUT, "k5_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + COLS)
        for label, r in rows:
            w.writerow([label] + ["" if r[c] is None else f"{r[c]:.1f}" for c in COLS])
    head = ["Method", "Sound", "Defective", "Reject", "Bruise", "Serious", "Slight", "Acc 526", "Stem/calyx", "Acc 819"]
    lines = ["# Table K5 — sound / defective, 2 groups (Kleynen 2005 Table 2 format)", "",
             "Correct classification rate (%). Sound = 280 healthy apples, defective = 246 apples (59 reject, 56 bruise, 55 serious, "
             "76 slight); *Acc 526* = the paper's set (280 + 246). *Stem/calyx* = 293 healthy-class images of the stem or calyx end "
             "(the paper used a separate stem/calyx detector, not replicated); *Acc 819* = whole dataset. All replicated / CNN "
             "numbers are out-of-fold (5 folds, StratifiedKFold seed 42). CNN merged = 3-class model, defective if "
             "P(internal) + P(surface) > 0.5. Paper Acc 526 computed as (280 × 94.3 + 246 × 84.6) / 526.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for label, r in rows:
        cells = ["–" if r[c] is None else f"{r[c]:.1f}" for c in COLS]
        if label.startswith("CNN merged NIR"):
            label, cells = f"**{label}**", [f"**{c}**" for c in cells]
        lines.append("| " + " | ".join([label] + cells) + " |")
    lines += ["", "McNemar (exact if discordant < 25):"]
    lines += [f"- {k}: {v['acc_a'] * 100:.1f}% vs {v['acc_b'] * 100:.1f}%, discordant {v['only_a_correct']} / {v['only_b_correct']}, "
              f"p = {v['p_value']:.1e} (n = {v['n']})" for k, v in mc.items()]
    lines += ["", "Source: outputs/dl/k5_summary.json (K5), Kleynen et al. 2005 (J. Food Eng. 69:41–49) Sect. 3 + Table 2."]
    open(os.path.join(OUT, "k5_table.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def bar_labels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=7.5, color=TEXT2)


def fig_severity(rows, counts):
    r = dict(rows)
    series = [("Kleynen 2005 (paper)", r["Kleynen 2005 (paper numbers)"]),
              ("Kleynen replicated (450/750/800)", r["Kleynen replicated 450/750/800 [A: paper protocol]"]),
              ("CNN NIR (nir_ss)", r["CNN merged NIR (nir_ss)"]),
              ("CNN Visible (visible_ss)", r["CNN merged Visible (visible_ss)"])]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    style(ax)
    x = np.arange(len(SEV_ORDER))
    w = 0.2
    for k, (name, vals) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * w, [vals[s] for s in SEV_ORDER], w - 0.02, color=SERIES[k], label=name, zorder=2)
        bar_labels(ax, bars)
    ax.set_xticks(x, [f"{s}\n(n = {counts[s]})" for s in SEV_ORDER], color=TEXT)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Defective apples correctly rejected (%)", color=TEXT2)
    ax.set_title("Accuracy by defect severity (2 groups, 246 defective apples)", color=TEXT, loc="left", fontsize=12)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), labelcolor=TEXT)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_severity_accuracy.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {name: {s: vals[s] for s in SEV_ORDER} for name, vals in series}


def fig_nir_vs_visible(rows, mc_k, mc_c):
    r = dict(rows)
    groups = ["Kleynen 2005 method\n(replicated, histogram Bayes + LDA)", "CNN\n(EfficientNet-B0, SingleSpectra)"]
    nir = [r["Kleynen replicated 750/800 (NIR) [A]"]["acc_526"], r["CNN merged NIR (nir_ss)"]["acc_526"]]
    vis = [r["Kleynen replicated 450/500 (Visible) [A]"]["acc_526"], r["CNN merged Visible (visible_ss)"]["acc_526"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    style(ax)
    x = np.arange(2)
    w = 0.32
    b1 = ax.bar(x - w / 2, nir, w - 0.03, color=SERIES[0], label="NIR (750 + 800 nm)", zorder=2)
    b2 = ax.bar(x + w / 2, vis, w - 0.03, color=SERIES[1], label="Visible (450 + 500 nm)", zorder=2)
    bar_labels(ax, list(b1) + list(b2))
    for i, m in enumerate((mc_k, mc_c)):
        top = max(nir[i], vis[i]) + 7
        ax.plot([i - w / 2, i - w / 2, i + w / 2, i + w / 2], [top - 2, top, top, top - 2], color=TEXT2, linewidth=1)
        ax.text(i, top + 1, f"McNemar p = {m['p_value']:.1e}", ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.set_xticks(x, groups, color=TEXT)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Accuracy, sound vs defective (%)", color=TEXT2)
    ax.set_title("NIR beats visible in both methods (280 sound + 246 defective, out-of-fold)", color=TEXT, loc="left", fontsize=11)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.17), labelcolor=TEXT)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_nir_vs_visible.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return {"kleynen": {"nir": nir[0], "visible": vis[0], "mcnemar": mc_k}, "cnn": {"nir": nir[1], "visible": vis[1], "mcnemar": mc_c}}


def fig_pipelines():
    steps_k = ["4 bands\n450/500/750/800 nm\n(SingleSpectra 430×560)", "Fruit mask\nthreshold 800 nm\n+ fill holes + closing\n+ erosion 5 px",
               "Pixel classes\nBayes, equal priors\n3-D histogram\n(450/750/800)", "Fruit features\nmean/median/std per band\nsound & defect tissue\n+ defect ratio, regions",
               "LDA\nsound / defective\n+ P(defective)"]
    steps_c = ["2 bands\n750 + 800 nm\n(SingleSpectra, 224×224)", "EfficientNet-B0\n(ImageNet init)\n5 folds × 3 seeds",
               "TTA 6 views\n× 3 seeds\nsoft vote", "3 classes\nhealthy / internal /\nsurface defect", "2 groups\ndefective if\nP(int) + P(surf) > 0.5"]
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=SURFACE)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    bw, bh, gap = 2.0, 1.25, 0.42
    for row, (title, steps, col, y) in enumerate((("Kleynen et al. 2005 (replicated, K3)", steps_k, SERIES[1], 2.75),
                                                  ("CNN NIR (nir_ss, final formula)", steps_c, SERIES[0], 0.45))):
        ax.text(0.15, y + bh + 0.18, title, fontsize=11.5, color=TEXT, fontweight="bold")
        for i, s in enumerate(steps):
            x0 = 0.15 + i * (bw + gap)
            ax.add_patch(FancyBboxPatch((x0, y), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.6,
                                        edgecolor=col, facecolor=SURFACE))
            ax.text(x0 + bw / 2, y + bh / 2, s, ha="center", va="center", fontsize=8.3, color=TEXT, linespacing=1.25)
            if i < len(steps) - 1:
                ax.add_patch(FancyArrowPatch((x0 + bw + 0.04, y + bh / 2), (x0 + bw + gap - 0.04, y + bh / 2), arrowstyle="-|>",
                                             mutation_scale=12, color=TEXT2, linewidth=1.2))
    ax.text(0.15, 0.08, "Both evaluated out-of-fold on the same 5 folds (fold_assignment.json): a test apple is never seen by the "
            "histograms/LDA or by the CNN that classify it.", fontsize=8.5, color=TEXT2)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"fig_pipelines.{ext}"), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    from .kleynen import context, severity_per_image
    from .nir_proof import mcnemar_exact
    os.makedirs(OUT, exist_ok=True)
    k5 = json.load(open(os.path.join(DL, "k5_summary.json"), encoding="utf-8"))
    k2 = json.load(open(os.path.join(DL, "k2_merged.json"), encoding="utf-8"))
    rows = table_rows(k5["rows"])
    ctx, _, _ = context()
    sev = severity_per_image(ctx)
    counts = {s: int(np.sum(sev == s)) for s in SEV_ORDER}
    tissue = np.asarray(ctx.tissue)
    yb = (np.asarray(ctx.labels) != 0).astype(int)
    paper = ~np.isin(tissue, ["Stem", "Calix"])
    pn, pv = (np.array(k2["predictions"][k]) for k in ("nir_ss (NIR, SS)", "visible_ss (Visible, SS)"))
    mc_c = mcnemar_exact(yb[paper], pn[paper], pv[paper])
    mc_c.update(acc_a=float((pn[paper] == yb[paper]).mean()), acc_b=float((pv[paper] == yb[paper]).mean()), n=int(paper.sum()))
    mc = {k: v for k, v in k5["mcnemar"].items()}
    mc["CNN merged NIR (nir_ss) vs CNN merged Visible (visible_ss) [paper_526] (G4)"] = mc_c
    mc_k = k5["mcnemar"]["Kleynen replicated 750/800 (NIR) [A: paper protocol] vs Kleynen replicated 450/500 (Visible) [A: paper protocol] [paper_526]"]
    write_tables(rows, mc)
    sev_vals = fig_severity(rows, counts)
    nv = fig_nir_vs_visible(rows, mc_k, mc_c)
    fig_pipelines()
    json.dump({"source": "outputs/dl/k5_summary.json, outputs/dl/k2_merged.json, kleynen.PAPER", "severity_counts": counts,
               "table_k5": dict(rows), "fig_severity_accuracy": sev_vals, "fig_nir_vs_visible": nv},
              open(os.path.join(OUT, "g4_numbers.json"), "w", encoding="utf-8"), indent=1)
    print("G4 ->", OUT, sorted(os.listdir(OUT)))
    print(f"  CNN NIR vs Visible on 526: {mc_c['acc_a'] * 100:.1f}% vs {mc_c['acc_b'] * 100:.1f}%, p = {mc_c['p_value']:.1e}")


if __name__ == "__main__":
    main()
