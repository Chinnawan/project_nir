# K3 feature ablation (450/750/800 nm)

Both local rows use the same 5 out-of-fold splits and 526-image subset (280 sound + 246 defective). Published percentages are reference values, not paired out-of-fold measurements from this experiment.

| Method | Sound (n=280) | Defective (n=246) | Accuracy (n=526) | Stem/calyx (n=293; supplementary) |
|---|---:|---:|---:|---:|
| Kleynen 2005 (published) | 94.3% | 84.6% | 89.8% | separate detector |
| Local K3 enriched (recorded) | 86.1% | 78.0% | 82.3% | 25.6% |
| Local K3 paper-described fruit features only (new) | 87.9% | 79.7% | 84.0% | 21.8% |

The new row removes only the two added region features. It is NOT an exact paper replication: 64-bin histogram, Otsu/erosion settings, whole SBW masks, cross-fitting and no separate stem/calyx matcher still differ. Do not replace the original K3/K5 figures with this exploratory ablation.
