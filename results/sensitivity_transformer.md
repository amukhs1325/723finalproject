# Does the result depend on the definition of creativity? (transformer)

Regenerate with `python scripts/sensitivity.py transformer`. The definitions were fixed before the numbers were
computed and all of them are shown. Judge cut is provisional (3.0) until the scorer freezes.

## 1. Eight definitions, same checkpoints

Gap = rate at the step the other seeds say is best under that definition, minus the rate at this
run's own lowest-loss checkpoint. Mean of 3 seeds, standard error from sampling.

| definition | size | rate at lowest loss | lowest-loss step | best step (mean of seeds) | best step is | held-out gap | SE | gap / SE | gap as % of rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base (judge ≥ 3) | small | 0.710 | 2626 | 4309 | later | -0.011 | 0.013 | -0.829 | -1.567 |
| base (judge ≥ 3) | medium | 0.717 | 762 | 1250 | later | 0.010 | 0.012 | 0.827 | 1.390 |
| base (judge ≥ 3) | large | 0.710 | 464 | 762 | later | -0.025 | 0.013 | -1.954 | -3.583 |
| judge ≥ 3.5 | small | 0.588 | 2626 | 5519 | later | 0.005 | 0.015 | 0.357 | 0.922 |
| judge ≥ 3.5 | medium | 0.594 | 762 | 1250 | later | -0.039 | 0.015 | -2.606 | -6.498 |
| judge ≥ 3.5 | large | 0.578 | 464 | 762 | later | -0.007 | 0.015 | -0.452 | -1.146 |
| judge ≥ 4 | small | 0.524 | 2626 | 9056 | later | 0.011 | 0.016 | 0.733 | 2.178 |
| judge ≥ 4 | medium | 0.539 | 762 | 1250 | later | -0.041 | 0.015 | -2.665 | -7.655 |
| judge ≥ 4 | large | 0.502 | 464 | 762 | later | -0.019 | 0.016 | -1.178 | -3.753 |
| far from training | small | 0.339 | 2626 | 762 | earlier | -0.023 | 0.012 | -1.902 | -6.803 |
| far from training | medium | 0.332 | 762 | 762 | same | 0.008 | 0.009 | 0.878 | 2.395 |
| far from training | large | 0.338 | 464 | 464 | same | -0.024 | 0.009 | -2.511 | -6.977 |
| not a splice | small | 0.406 | 2626 | 1250 | earlier | 0.033 | 0.013 | 2.489 | 8.129 |
| not a splice | medium | 0.451 | 762 | 762 | same | -0.024 | 0.010 | -2.383 | -5.318 |
| not a splice | large | 0.417 | 464 | 283 | earlier | -0.029 | 0.013 | -2.172 | -6.996 |
| plausible but atypical | small | 0.363 | 2626 | 762 | earlier | 0.054 | 0.014 | 3.870 | 14.873 |
| plausible but atypical | medium | 0.407 | 762 | 762 | same | -0.008 | 0.010 | -0.741 | -1.869 |
| plausible but atypical | large | 0.380 | 464 | 82 | earlier | -0.014 | 0.014 | -0.992 | -3.595 |
| 3-gram plausibility | small | 0.738 | 2626 | 2050 | earlier | -0.015 | 0.011 | -1.309 | -2.033 |
| 3-gram plausibility | medium | 0.719 | 762 | 1250 | later | 0.024 | 0.011 | 2.128 | 3.384 |
| 3-gram plausibility | large | 0.724 | 464 | 464 | same | -0.009 | 0.009 | -0.922 | -1.197 |
| real held-out names | small | 0.007 | 2626 | 1601 | earlier | 0.000 | 0.002 | 0.000 | 0.000 |
| real held-out names | medium | 0.008 | 762 | 2626 | later | 0.001 | 0.002 | 0.279 | 8.000 |
| real held-out names | large | 0.006 | 464 | 762 | later | 0.001 | 0.002 | 0.619 | 21.053 |

## 2. Is the judge just the 3-gram model again?

40,044 distinct novel names from the grid that the judge rated.

* Spearman, judge rating vs. 3-gram percentile: 0.67; vs. name length: -0.18; vs. distance from the training set: -0.51
* Judge >= 3 and 3-gram >= 0.10 agree on 80.0% of names. Judge passes 71.2%, 3-gram passes 73.7%.
* Of names the 3-gram passes, the judge rejects 15.2%. Of names the 3-gram rejects, the judge passes 33.2%.

## 3. Does loss determine creativity?

Each run passes through the same validation loss twice: once undertrained, once overfit. If the
score were loss in disguise the two visits would score the same. Mean of 3 seeds, base definition,
values interpolated along each side of the curve.

| size | validation loss | creativity, undertrained side | creativity, overfit side | copies, undertrained side | copies, overfit side |
| --- | --- | --- | --- | --- | --- |
| small | 2.447 | 0.655 | 0.698 | 0.024 | 0.050 |
| small | 2.497 | 0.589 | 0.685 | 0.020 | 0.060 |
| small | 2.597 | 0.419 | 0.666 | 0.018 | 0.074 |
| medium | 2.445 | 0.630 | 0.694 | 0.031 | 0.055 |
| medium | 2.495 | 0.551 | 0.672 | 0.022 | 0.076 |
| medium | 2.595 | 0.426 | 0.646 | 0.017 | 0.099 |
| large | 2.447 | 0.641 | 0.686 | 0.027 | 0.065 |
| large | 2.497 | 0.578 | 0.667 | 0.022 | 0.076 |

Across all 252 checkpoints, Spearman between validation loss and the base creativity rate is -0.86.
