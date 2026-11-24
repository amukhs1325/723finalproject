# Does the result depend on the definition of creativity?

Regenerate with `python scripts/sensitivity.py`. The definitions were fixed before the numbers were
computed and all of them are shown. Judge cut is provisional (3.0) until the scorer freezes.

## 1. Eight definitions, same checkpoints

Gap = rate at the step the other seeds say is best under that definition, minus the rate at this
run's own lowest-loss checkpoint. Mean of 3 seeds, standard error from sampling.

| definition | size | rate at lowest loss | lowest-loss step | best step (mean of seeds) | best step is | held-out gap | SE | gap / SE | gap as % of rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base (judge ≥ 3) | small | 0.719 | 1601 | 2050 | later | 0.005 | 0.013 | 0.389 | 0.724 |
| base (judge ≥ 3) | medium | 0.700 | 976 | 1601 | later | -0.012 | 0.013 | -0.994 | -1.777 |
| base (judge ≥ 3) | large | 0.692 | 464 | 221 | earlier | -0.008 | 0.013 | -0.570 | -1.111 |
| judge ≥ 3.5 | small | 0.614 | 1601 | 5519 | later | -0.029 | 0.012 | -2.482 | -4.693 |
| judge ≥ 3.5 | medium | 0.586 | 976 | 2050 | later | 0.021 | 0.014 | 1.466 | 3.515 |
| judge ≥ 3.5 | large | 0.582 | 464 | 762 | later | 0.004 | 0.014 | 0.276 | 0.665 |
| judge ≥ 4 | small | 0.553 | 1601 | 4309 | later | -0.018 | 0.015 | -1.197 | -3.338 |
| judge ≥ 4 | medium | 0.530 | 976 | 2626 | later | 0.025 | 0.014 | 1.756 | 4.776 |
| judge ≥ 4 | large | 0.518 | 464 | 762 | later | -0.000 | 0.015 | -0.014 | -0.042 |
| far from training | small | 0.344 | 1601 | 762 | earlier | -0.007 | 0.012 | -0.605 | -2.132 |
| far from training | medium | 0.309 | 976 | 172 | earlier | 0.013 | 0.013 | 0.981 | 4.086 |
| far from training | large | 0.304 | 464 | 135 | earlier | 0.042 | 0.012 | 3.487 | 13.672 |
| not a splice | small | 0.407 | 1601 | 976 | earlier | 0.025 | 0.013 | 1.907 | 6.110 |
| not a splice | medium | 0.416 | 976 | 221 | earlier | 0.018 | 0.013 | 1.445 | 4.338 |
| not a splice | large | 0.402 | 464 | 221 | earlier | 0.050 | 0.012 | 4.156 | 12.547 |
| plausible but atypical | small | 0.367 | 1601 | 221 | earlier | 0.032 | 0.014 | 2.240 | 8.655 |
| plausible but atypical | medium | 0.361 | 976 | 172 | earlier | 0.026 | 0.013 | 1.927 | 7.097 |
| plausible but atypical | large | 0.357 | 464 | 221 | earlier | 0.028 | 0.013 | 2.151 | 7.946 |
| 3-gram plausibility | small | 0.736 | 1601 | 2050 | later | 0.014 | 0.011 | 1.272 | 1.947 |
| 3-gram plausibility | medium | 0.727 | 976 | 762 | earlier | 0.004 | 0.011 | 0.378 | 0.596 |
| 3-gram plausibility | large | 0.716 | 464 | 595 | later | -0.011 | 0.007 | -1.680 | -1.583 |
| real held-out names | small | 0.004 | 1601 | 7070 | later | 0.004 | 0.002 | 2.090 | 100.000 |
| real held-out names | medium | 0.008 | 976 | 2050 | later | -0.000 | 0.002 | -0.150 | -4.348 |
| real held-out names | large | 0.009 | 464 | 762 | later | 0.001 | 0.003 | 0.393 | 10.714 |

## 2. Is the judge just the 3-gram model again?

38,355 distinct novel names from the grid that the judge rated.

* Spearman, judge rating vs. 3-gram percentile: 0.64; vs. name length: -0.16; vs. distance from the training set: -0.49
* Judge >= 3 and 3-gram >= 0.10 agree on 80.3% of names. Judge passes 74.7%, 3-gram passes 76.7%.
* Of names the 3-gram passes, the judge rejects 14.1%. Of names the 3-gram rejects, the judge passes 38.0%.

## 3. Does loss determine creativity?

Each run passes through the same validation loss twice: once undertrained, once overfit. If the
score were loss in disguise the two visits would score the same. Mean of 3 seeds, base definition,
values interpolated along each side of the curve.

| size | validation loss | creativity, undertrained side | creativity, overfit side | copies, undertrained side | copies, overfit side |
| --- | --- | --- | --- | --- | --- |
| small | 2.416 | 0.669 | 0.714 | 0.023 | 0.055 |
| small | 2.466 | 0.622 | 0.704 | 0.029 | 0.065 |
| small | 2.566 | 0.478 | 0.675 | 0.026 | 0.082 |
| medium | 2.414 | 0.672 | 0.707 | 0.029 | 0.065 |
| medium | 2.464 | 0.622 | 0.693 | 0.026 | 0.089 |
| medium | 2.564 | 0.482 | 0.640 | 0.025 | 0.115 |
| large | 2.416 | 0.664 | 0.652 | 0.029 | 0.080 |
| large | 2.466 | 0.624 | 0.644 | 0.025 | 0.108 |

Across all 252 checkpoints, Spearman between validation loss and the base creativity rate is -0.86.
