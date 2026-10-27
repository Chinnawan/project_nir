# Table K5 — sound / defective, 2 groups (Kleynen 2005 Table 2 format)

Correct classification rate (%). Sound = 280 healthy apples, defective = 246 apples (59 reject, 56 bruise, 55 serious, 76 slight); *Acc 526* = the paper's set (280 + 246). *Stem/calyx* = 293 healthy-class images of the stem or calyx end (the paper used a separate stem/calyx detector, not replicated); *Acc 819* = whole dataset. All replicated / CNN numbers are out-of-fold (5 folds, StratifiedKFold seed 42). CNN merged = 3-class model, defective if P(internal) + P(surface) > 0.5. Paper Acc 526 computed as (280 × 94.3 + 246 × 84.6) / 526.

| Method | Sound | Defective | Reject | Bruise | Serious | Slight | Acc 526 | Stem/calyx | Acc 819 |
|---|---|---|---|---|---|---|---|---|---|
| Kleynen 2005 (paper numbers) | 94.3 | 84.6 | 100.0 | 98.2 | 94.5 | 55.0 | 89.8 | – | – |
| Kleynen replicated 450/750/800 [A: paper protocol] | 86.1 | 78.0 | 93.2 | 92.9 | 83.6 | 51.3 | 82.3 | 25.6 | 62.0 |
| Kleynen replicated 750/800 (NIR) [A] | 89.3 | 80.5 | 91.5 | 92.9 | 85.5 | 59.2 | 85.2 | 16.0 | 60.4 |
| Kleynen replicated 450/500 (Visible) [A] | 75.0 | 58.9 | 72.9 | 67.9 | 54.5 | 44.7 | 67.5 | 23.2 | 51.6 |
| Kleynen replicated 450/750/800 [B: stem/calyx = sound] | 93.6 | 56.9 | 74.6 | 87.5 | 43.6 | 30.3 | 76.4 | 91.1 | 81.7 |
| Kleynen replicated 450/750/800 [A, erosion 15 px: sensitivity] | 88.2 | 82.5 | 98.3 | 94.6 | 90.9 | 55.3 | 85.6 | 8.5 | 58.0 |
| **CNN merged NIR (nir_ss)** | **96.8** | **89.0** | **94.9** | **96.4** | **85.5** | **81.6** | **93.2** | **95.6** | **94.0** |
| CNN merged Visible (visible_ss) | 94.6 | 78.0 | 94.9 | 69.6 | 74.5 | 73.7 | 86.9 | 96.6 | 90.4 |
| CNN 2-class NIR (nir_ss_2cls) | 96.4 | 88.6 | 94.9 | 96.4 | 87.3 | 78.9 | 92.8 | 96.6 | 94.1 |
| CNN 2-class Visible (visible_ss_2cls) | 93.2 | 75.6 | 94.9 | 57.1 | 74.5 | 75.0 | 85.0 | 94.9 | 88.5 |

McNemar (exact if discordant < 25):
- Kleynen replicated 450/750/800 (paper) [A: paper protocol] vs CNN merged NIR (nir_ss) [paper_526]: 82.3% vs 93.2%, discordant 24 / 81, p = 4.6e-08 (n = 526)
- Kleynen replicated 450/750/800 (paper) [B: stem/calyx = sound] vs CNN merged NIR (nir_ss) [all_819]: 81.7% vs 94.0%, discordant 25 / 126, p = 4.0e-16 (n = 819)
- Kleynen replicated 450/750/800 (paper) [A: paper protocol] vs Kleynen replicated 750/800 (NIR) [A: paper protocol] [paper_526]: 82.3% vs 85.2%, discordant 21 / 36, p = 6.4e-02 (n = 526)
- Kleynen replicated 750/800 (NIR) [A: paper protocol] vs Kleynen replicated 450/500 (Visible) [A: paper protocol] [paper_526]: 85.2% vs 67.5%, discordant 129 / 36, p = 7.9e-13 (n = 526)
- CNN merged NIR (nir_ss) vs CNN merged Visible (visible_ss) [all_819]: 94.0% vs 90.4%, discordant 60 / 30, p = 2.2e-03 (n = 819)
- Kleynen replicated 450/750/800 (paper) [A, erosion 15px: sensitivity] vs CNN merged NIR (nir_ss) [paper_526]: 85.6% vs 93.2%, discordant 22 / 62, p = 2.1e-05 (n = 526)
- Kleynen replicated 450/750/800 (paper) [A: paper protocol] vs CNN 2-class NIR (nir_ss_2cls) [paper_526]: 82.3% vs 92.8%, discordant 22 / 77, p = 5.7e-08 (n = 526)
- CNN 2-class NIR (nir_ss_2cls) vs CNN merged NIR (nir_ss) [all_819]: 94.1% vs 94.0%, discordant 8 / 7, p = 1.0e+00 (n = 819)
- CNN 2-class NIR (nir_ss_2cls) vs CNN 2-class Visible (visible_ss_2cls) [all_819]: 94.1% vs 88.5%, discordant 71 / 25, p = 4.4e-06 (n = 819)
- CNN merged NIR (nir_ss) vs CNN merged Visible (visible_ss) [paper_526] (G4): 93.2% vs 86.9%, discordant 55 / 22, p = 2.7e-04 (n = 526)

Source: outputs/dl/k5_summary.json (K5), Kleynen et al. 2005 (J. Food Eng. 69:41–49) Sect. 3 + Table 2.
