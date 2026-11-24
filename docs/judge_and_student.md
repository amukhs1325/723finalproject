# The judge, its student, and how a checkpoint gets a plausibility number

Added 22 Nov 2026. This is the part of the scorer that decides whether a novel name is
**plausible**. Novelty is unchanged and still fully deterministic.

## The short version

| Piece | What it is | Cost | Role |
| --- | --- | --- | --- |
| **Judge** | `gpt-4.1-mini-2025-04-14`, one name per call, asked the same 1 to 5 question as our human raters | one API call per name | the reference. Every plausibility number is defined by it. |
| **Student** | a 55,000-parameter character model trained to copy the judge (`student.pt`) | free, runs on a laptop | scores every name, but is **not trusted on its own** |
| **Estimate** | student on all names + judge on a random 200, combined | 200 calls per checkpoint | the number we report, with a standard error |

Nobody needs an API key to reproduce results: the judge's ratings are in `judge_cache.csv`
(13,102 names so far) and the student's weights are in the repo.

## 1. Soft ratings

The judge answers with one digit. Instead of keeping only that digit we read the model's
probability for each of 1, 2, 3, 4, 5 and use the expected value. A name the judge is torn on
between 3 and 4 gets 3.4. About a third of names are like that (top rating below 90%).

Why it matters: the yes/no cut can sit between whole numbers, and the five probabilities are a
richer training signal for the student than a single label.

## 2. The student (knowledge distillation)

A small bidirectional GRU over the letters of the name, trained to output the judge's five
probabilities. Five copies from different random seeds are averaged.

Training data, all judged once:

* **Off-policy pool, 5,000 names** that no model of ours produced: real held-out names, the
  dataset author's char-RNN output, n-gram samples, shuffled letters, random strings.
* **On-policy names, 2,000**: samples from checkpoints of a reference LSTM (below).
* The 180 names in the human study are never used for training.

What we measured (`results/student/eval.md`, student vs. judge on names the student never saw):

| Question | Answer |
| --- | --- |
| Soft or hard targets? | Soft. Training on the five probabilities beats training on the top rating only: agreement kappa 0.54 vs. 0.48 on on-policy names. |
| How many judge labels? | Still improving at 4,000: rank correlation 0.59, 0.71, 0.78, 0.83 at 500, 1,000, 2,000, 4,000 labels. |
| Does a student trained without neural samples work on them? | Mostly: 0.81 correlation on-policy vs. 0.83 off-policy. Adding 1,000 on-policy labels brings it to 0.83. |
| Random or least-certain names for those 1,000 labels? | Least-certain was slightly better (kappa 0.60 vs. 0.57), within what one run can distinguish. |
| Final student | 0.84 correlation with the judge, 87% yes/no agreement, on five held-out checkpoints. |

## 3. Why the student alone is not enough

On the five held-out checkpoints the student rated too many names plausible, and by a different
amount at each stage of training:

| Step | Judge says plausible | Student says plausible | Gap |
| --- | --- | --- | --- |
| 50 (undertrained) | 64.8% | 77.7% | +12.9 |
| 209 | 78.7% | 89.4% | +10.6 |
| 542 (lowest loss) | 80.2% | 88.8% | +8.6 |
| 1,405 (overfitting) | 82.9% | 89.3% | +6.5 |
| 12,000 (badly overfit) | 78.9% | 90.4% | +11.4 |

A constant error would only shift the curve. An error that changes with training step bends it,
and loss against training step is the comparison this project makes. So the student is never
reported by itself.

## 4. The estimate we report

For each checkpoint (`cvl/estimate.py`):

1. The student scores every novel name.
2. A random 200 of them go to the judge.
3. On those 200 we see exactly how far off the student is, and subtract that error from the
   student's average over all the names.

Because the 200 are random, the correction is unbiased however wrong the student is. A better
student only makes the result less noisy. Checked on the five fully judged checkpoints:

* The 95% interval contained the exact answer in 94.9% of 12,000 random trials.
* For the same accuracy it needs about half as many judge calls as judging a plain random sample
  (variance ratio 0.56).

`python scripts/eval_checkpoint.py <run> --judge-sample all` skips the estimate and judges
every novel name.

## 5. The reference LSTM

`cvl/reference_lm.py` is a 305,000-parameter character LSTM trained far past its best point.
**It is a fixture for building the scorer, not one of the project's models.** It exists so the
evaluation side has realistic checkpoints before the real runs land. It follows the proposed
shared interface (29 symbols, `$` ends a name, loss in nats per symbol), so it also shows that
`eval_checkpoint.py` and `make_figures.py` work end to end.

`results/reference/figures/reference_lstm.png` is the first full picture: lowest validation loss
at step 542, exact copies rising from 4% to 85% of samples by step 12,000. The judge-based
creativity rate was highest around step 209 to 336 (0.72) against 0.69 at the loss minimum. That
gap is about one and a half standard errors, so it is not a finding. One seed, a provisional cut,
and a fixture model: **a plumbing check, not a result.**

## What is still provisional

* The yes/no cut is 3.0 until our ratings set it (`scripts/human_tasks.py analyze --freeze`).
* One rater so far. On the calibration half only, the judge's rank correlation with that rater
  is 0.75, the student's 0.75, the 3-gram's 0.64. The test half has not been looked at.
* Whether few-shot examples from our own ratings improve the judge is not tested yet. It needs
  all three raters.

## Rebuilding everything

```bash
python scripts/judge_pool.py --fill                 # 5,000-name pool -> judge            (API)
python scripts/train_reference_lstm.py              # reference checkpoints, about 4 minutes
python scripts/eval_checkpoint.py checkpoints/reference_lstm --out results/reference/checkpoints.csv
python scripts/train_student.py select --fill       # pick and judge on-policy names        (API)
python scripts/train_student.py train               # all variants, eval.md, student.pt, about 7 minutes
python scripts/eval_checkpoint.py checkpoints/reference_lstm --out results/reference/checkpoints.csv --fill-judge
python scripts/make_figures.py results/reference/checkpoints.csv
```

With `judge_cache.csv` in the repo, none of the API steps makes a call.
