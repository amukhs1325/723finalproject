# Originality as a dial

Regenerate with `python scripts/originality_dial.py`. Written after the LSTM results were seen, so on
these runs it describes; it does not test. Judge cut is provisional (3.0) until the scorer freezes.

A name counts if it is creative under the base definition and its spelling distance from the nearest
training name is at least the bar. 1/6 is the base score. Gap = rate at the step the other seeds say
is best, minus the rate at this run's own lowest-loss checkpoint. Mean of 3 seeds.

## 1. Raising the bar

| distance at least | real held-out names that clear it | size | rate at lowest loss | lowest-loss step | best step (mean of seeds) | best step is | held-out gap | SE | gap / SE | gap as % of rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1/6 | 93% | small | 0.719 | 1601 | 2050 | later | 0.005 | 0.013 | 0.389 | 0.724 |
| 1/6 | 93% | medium | 0.700 | 976 | 1601 | later | -0.012 | 0.013 | -0.994 | -1.777 |
| 1/6 | 93% | large | 0.692 | 464 | 221 | earlier | -0.008 | 0.013 | -0.570 | -1.111 |
| 1/5 | 81% | small | 0.645 | 1601 | 2050 | later | -0.001 | 0.013 | -0.041 | -0.083 |
| 1/5 | 81% | medium | 0.621 | 976 | 1601 | later | -0.013 | 0.011 | -1.169 | -2.021 |
| 1/5 | 81% | large | 0.613 | 464 | 221 | earlier | 0.029 | 0.013 | 2.241 | 4.770 |
| 1/4 | 63% | small | 0.550 | 1601 | 1601 | same | -0.017 | 0.010 | -1.705 | -3.173 |
| 1/4 | 63% | medium | 0.522 | 976 | 221 | earlier | -0.014 | 0.013 | -1.121 | -2.698 |
| 1/4 | 63% | large | 0.514 | 464 | 221 | earlier | 0.034 | 0.013 | 2.645 | 6.536 |
| 2/7 | 51% | small | 0.466 | 1601 | 2050 | later | -0.007 | 0.010 | -0.771 | -1.595 |
| 2/7 | 51% | medium | 0.442 | 976 | 595 | earlier | 0.004 | 0.012 | 0.348 | 0.921 |
| 2/7 | 51% | large | 0.435 | 464 | 221 | earlier | 0.000 | 0.011 | 0.035 | 0.087 |
| 1/3 | 38% | small | 0.344 | 1601 | 762 | earlier | -0.007 | 0.012 | -0.605 | -2.132 |
| 1/3 | 38% | medium | 0.309 | 976 | 172 | earlier | 0.013 | 0.013 | 0.981 | 4.086 |
| 1/3 | 38% | large | 0.304 | 464 | 135 | earlier | 0.042 | 0.012 | 3.487 | 13.672 |
| 2/5 | 12% | small | 0.109 | 1601 | 172 | earlier | -0.011 | 0.011 | -0.992 | -9.692 |
| 2/5 | 12% | medium | 0.093 | 976 | 172 | earlier | 0.021 | 0.010 | 2.008 | 22.238 |
| 2/5 | 12% | large | 0.095 | 464 | 135 | earlier | 0.002 | 0.010 | 0.239 | 2.403 |
| 1/2 | 3% | small | 0.009 | 1601 | 976 | earlier | 0.000 | 0.006 | 0.006 | 0.420 |
| 1/2 | 3% | medium | 0.012 | 976 | 172 | earlier | 0.002 | 0.005 | 0.281 | 12.309 |
| 1/2 | 3% | large | 0.013 | 464 | 64 | earlier | 0.000 | 0.005 | 0.093 | 3.821 |

## 2. No bar: each creative name counts by its distance

Units are distance-weighted names per sample, so compare the last column with the tables above.

| size | rate at lowest loss | lowest-loss step | best step (mean of seeds) | best step is | held-out gap | SE | gap / SE | gap as % of rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | 0.216 | 1601 | 2050 | later | -0.003 | 0.004 | -0.859 | -1.585 |
| medium | 0.206 | 976 | 1250 | later | -0.007 | 0.004 | -1.760 | -3.429 |
| large | 0.204 | 464 | 221 | earlier | 0.003 | 0.004 | 0.792 | 1.525 |
