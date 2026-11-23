# Run summary

From `checkpoints.csv`. Regenerate with `python scripts/summarize_runs.py`.

Held-out gap: creativity at the step the other seeds say is most creative, minus creativity
at this run's own lowest-loss checkpoint. Positive means the lowest-loss checkpoint was not
the most creative one.

| arch | size | seeds | lowest val loss | at step (median) | creativity at lowest loss | creative step (median) | creativity there | held-out gap | gap SE | gap by seed | copies at lowest loss | copies at last step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | small | 3 | 2.366 | 1601 | 0.719 | 2050 | 0.726 | 0.007 | 0.013 | +0.023 -0.004 +0.001 | 0.028 | 0.108 |
| lstm | medium | 3 | 2.363 | 976 | 0.700 | 1250 | 0.688 | -0.012 | 0.013 | +0.004 -0.030 -0.011 | 0.044 | 0.655 |
| lstm | large | 3 | 2.365 | 464 | 0.693 | 221 | 0.684 | -0.009 | 0.013 | +0.004 -0.013 -0.019 | 0.042 | 0.924 |

## Per run

| run | arch | size | seed | params | best_step | best_val_loss | creativity_best | creativity_best_se | copy_rate_best | peak_step | creativity_peak | last_step | last_val_loss | creativity_last | copy_rate_last | heldout_step | creativity_heldout | heldout_gap | heldout_gap_se |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm_large_s0 | lstm | large | 0 | 400325 | 595 | 2.363 | 0.698 | 0.016 | 0.049 | 221 | 0.703 | 40000 | 3.755 | 0.044 | 0.930 | 221 | 0.703 | 0.004 | 0.024 |
| lstm_large_s1 | lstm | large | 1 | 400325 | 464 | 2.361 | 0.676 | 0.016 | 0.045 | 221 | 0.726 | 40000 | 3.785 | 0.051 | 0.920 | 595 | 0.664 | -0.013 | 0.024 |
| lstm_large_s2 | lstm | large | 2 | 400325 | 464 | 2.370 | 0.706 | 0.015 | 0.033 | 362 | 0.730 | 40000 | 3.755 | 0.053 | 0.923 | 221 | 0.687 | -0.019 | 0.021 |
| lstm_medium_s0 | lstm | medium | 0 | 99967 | 976 | 2.367 | 0.683 | 0.016 | 0.050 | 1250 | 0.731 | 40000 | 4.163 | 0.218 | 0.673 | 2050 | 0.687 | 0.004 | 0.021 |
| lstm_medium_s1 | lstm | medium | 1 | 99967 | 762 | 2.359 | 0.720 | 0.015 | 0.034 | 976 | 0.725 | 40000 | 4.116 | 0.251 | 0.638 | 1250 | 0.691 | -0.030 | 0.023 |
| lstm_medium_s2 | lstm | medium | 2 | 99967 | 976 | 2.362 | 0.697 | 0.015 | 0.049 | 2050 | 0.726 | 40000 | 4.159 | 0.220 | 0.654 | 1250 | 0.685 | -0.011 | 0.021 |
| lstm_small_s0 | lstm | small | 0 | 25321 | 1601 | 2.365 | 0.707 | 0.016 | 0.031 | 2626 | 0.759 | 40000 | 2.725 | 0.628 | 0.119 | 2050 | 0.731 | 0.023 | 0.022 |
| lstm_small_s1 | lstm | small | 1 | 25321 | 1601 | 2.367 | 0.731 | 0.017 | 0.026 | 1601 | 0.731 | 40000 | 2.702 | 0.672 | 0.108 | 2050 | 0.726 | -0.004 | 0.024 |
| lstm_small_s2 | lstm | small | 2 | 25321 | 1601 | 2.366 | 0.719 | 0.017 | 0.028 | 2050 | 0.769 | 40000 | 2.711 | 0.651 | 0.096 | 2626 | 0.720 | 0.001 | 0.023 |
