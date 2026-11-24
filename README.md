# Creativity vs. Loss

CMSC 723 (Fall 2026) final project by Andre Atkins, Agastya Mukherjee and Vraj Patel.

**Question:** is the language model with the lowest loss also the most creative? We train
small character-level models on fantasy character names, score each training checkpoint for
creativity (names that are new and still sound like real names), and compare where creativity
peaks with where validation loss bottoms out.

**Status:** data, shared sampling code and the bigram baseline are done. The creativity scorer
is at v2 (novelty views, length gate, cached LLM judge with a distilled local student) and
**not frozen yet**: it freezes once the three of us have done the 15-minute rating app (see
Human study below). Design and evidence are in `docs/evaluation_plan.md` and
`docs/judge_and_student.md`. The LSTM grid is trained and scored: 3 sizes × 3 seeds, 252
checkpoints in `results/checkpoints.csv`, summarized in `results/summary.md` and drawn in
`results/figures/grid_lstm.png` (see LSTM runs below). Those numbers use a provisional judge
cut and get recomputed, at no API cost, when the scorer freezes. Next: the transformer runs,
the decoding sweep and recovery. Who does what, and by when, is in `docs/work_split.md`.

This repo prepares the fantasy-name data and scores generated names for creativity.
Pretraining and post-training code plug into two calls:

```python
from cvl.data import load_split
from cvl.scorer import score, score_table

train = load_split("train")      # list[str], also "val", "test"
result = score(samples)          # samples: list[str] of generated names
# {'n', 'creativity_rate', 'diversity', 'novelty', 'plausibility', 'novel_rate',
#  'plausible_rate', 'copy_rate', 'heldout_hit_rate', 'mean_length',
#  'respelling_rate', 'near_copy_rate', 'derived_rate', 'famous_rate', 'gate_fail_rate',
#  'splice_rate_among_novel', 'plausible_rate_given_novel', 'mean_judge_rating',
#  'creativity_rate_ngram', 'plausibility_mode'}
table = score_table(samples)     # one row per sample: tier, nearest training names, judge rating
```

Scoring a whole run (one row per checkpoint in `results/checkpoints.csv`):

```bash
python scripts/eval_checkpoint.py checkpoints/<run dir>/ [--fill-judge]
python scripts/make_figures.py                      # one three-panel figure per run
```

`eval_checkpoint.py` opens any checkpoint that `cvl.sampling.load_checkpoint` knows. A model needs
`symbols`, `next_probs(context)` and, to have its loss and step recorded, `meta = {"step": ...,
"val_loss": ...}`. `cvl/reference_lm.py` is a worked example of that interface.

Samples can include a trailing newline or `$` end marker; the scorer strips them.

## Sampling (shared by every model)

```python
from cvl.sampling import sample, adjust
names = sample("checkpoints/bigram.json", n=1000, temperature=1.0, top_p=1.0, seed=0)
```

A model works with `sample` if it exposes `symbols` (output symbols including the end
marker `$`) and `next_probs(context) -> probabilities over symbols`. Neural models can wrap
themselves in those two, or write a faster batched sampler that calls `adjust(probs,
temperature, top_p)` so decoding is identical across models. Add new checkpoint formats in
`load_checkpoint`. Report validation loss in **nats per symbol, counting the end marker**,
to match `CharNGram.val_loss`.

## Bigram baseline (E1)

```bash
python scripts/run_bigram_baseline.py   # under a minute with RapidFuzz, about 15 without
```

Trains a Witten-Bell character bigram on train (`checkpoints/bigram.json`) and runs the
decoding sweep on it: temperature 0.5 to 1.5 and top-p 0.5 to 0.95, 3 seeds × 1,000 names
each. Results in `outputs/baseline/`.

## LSTM runs

```bash
python training/train_lstm.py --size small medium large --seed 0 1 2   # the grid: 9 runs
python scripts/eval_checkpoint.py checkpoints/lstm_medium_s0/ --fill-judge   # score one run
python scripts/make_figures.py        # a figure per run, plus results/figures/grid_lstm.png
python scripts/summarize_runs.py      # results/summary.md and summary.csv
```

`models/lstm.py` is one LSTM layer over 32-dimensional character embeddings. Only the hidden
width changes with size: 60, 138 and 296 units give 25,321, 99,967 and 400,325 parameters.
Every run uses the same recipe (Adam at 2e-3, batches of 128 names, no dropout or weight
decay) and saves the same 28 log-spaced checkpoints from step 50 to step 40,000, about 410
passes over the training names. That is far past the lowest validation loss on purpose, so
each run has undertrained, lowest-loss and overfit checkpoints. With all nine running at once
on a laptop CPU, a small run took 6 minutes, a medium one 13, and a large one close to 3 hours.
Weights are written to `checkpoints/lstm_{size}_s{seed}/` and are not committed. The scored
table, the samples and the figures are.

`scripts/summarize_runs.py` compares creativity at each run's lowest-loss checkpoint with
creativity at the step the *other* seeds of that size say is most creative. Choosing the step
from other seeds matters: the highest of 28 noisy numbers overstates the peak, so a run never
picks its own.

What the nine runs show so far (provisional cut, scorer not frozen):

| | small | medium | large |
| --- | --- | --- | --- |
| lowest validation loss (mean of 3 seeds) | 2.366 | 2.363 | 2.365 |
| reached at step (median) | 1,601 | 976 | 464 |
| creativity rate at that checkpoint | 0.719 | 0.700 | 0.692 |
| held-out gap to the most creative step | +0.005 ± 0.013 | −0.012 ± 0.013 | −0.008 ± 0.013 |
| exact copies at lowest loss → at step 40,000 | 3% → 11% | 4% → 66% | 4% → 92% |

* All three sizes reach the same validation loss; a bigger model gets there sooner and
  overfits harder afterwards.
* Creativity climbs while the model is undertrained (names stop being gibberish), sits on a
  plateau around the loss minimum, and falls once the model starts reciting training names.
* The lowest-loss checkpoint is on the plateau for every size. The held-out gap is within one
  standard error of zero, so for these LSTMs sampled at temperature 1 the lowest-loss
  checkpoint is as creative as any other, to within about a point.
* Past the minimum the two measures part ways in size, not direction: the large model's
  creativity rate drops from 0.69 to 0.05 while almost every sample becomes an exact copy.

### Does that depend on how creativity is defined?

`python scripts/sensitivity.py` rescores the same samples under eight definitions that were
fixed before looking, and reports all of them in `results/sensitivity.md`
(`results/figures/sensitivity_lstm.png`). It makes no API calls.

* **A stricter judge cut (3.5 or 4) or the 3-gram instead of the judge** does not change the
  picture: the lowest-loss checkpoint stays on the plateau.
* **Asking for more originality does.** Requiring the name to be far from every training name,
  not a splice of two training names, or plausible to the judge while atypical under the
  3-gram moves the best checkpoint *before* the loss minimum for all three sizes. The
  held-out gap is positive in 8 of those 9 cells, up to +0.050 ± 0.012 (large, no splices),
  which is 4% to 14% more creative names than the lowest-loss checkpoint gives.
* **The score is not loss in disguise.** At the same validation loss, a slightly overfit
  checkpoint is more creative than an undertrained one (0.68 against 0.48 for the small model
  at 0.2 nats above its minimum; `results/figures/loss_vs_creativity_lstm.png`).
* **The judge is not the 3-gram again.** Over 38,355 names their ratings correlate at 0.64 and
  the two pass/fail rules disagree on one name in five. The judge also prefers names closer
  to the training set (−0.49), so plausibility and novelty pull against each other name by name.

These definitions were tried after seeing the base result, so they are a robustness check, not
a new headline. The clean test is the transformer grid, which has not been trained: the
prediction that originality-weighted creativity peaks before the loss minimum is written
down here first.

## Setup

```bash
pip install -r requirements.txt   # numpy, pandas, matplotlib, sentence-transformers; rapidfuzz optional
python scripts/prepare_data.py    # data/processed/{train,val,test}.txt + meta.json
python -m pytest tests            # 36 tests
```

`scorer_config.json` is already committed (version 2). **Do not rerun `scripts/calibrate.py`**:
the deterministic thresholds are set. The one remaining change is the judge's cut, written by
`scripts/human_tasks.py analyze --freeze` after we have all rated. After that the scorer is
frozen for the whole project.

The first scoring run downloads the embedding model (about 90 MB) from Hugging Face. No API key
is needed to score, train or run tests. `OPENAI_API_KEY` is only needed to judge names that are
not in `judge_cache.csv` yet (`--fill-judge`).

Install RapidFuzz on Colab (`pip install rapidfuzz`). Without it, novelty uses a NumPy
fallback that gives identical numbers but takes about 10 seconds per 1,000 names.

## Data

Janelle Shane's crowdsourced D&D character names (`data/raw/DnD_characters_May2018.txt`),
20,887 entries, from [janelleshane/DnD-characters](https://github.com/janelleshane/DnD-characters).
All credit for the raw data goes to its author and the players who submitted names.

| Step | Result |
| --- | --- |
| Unit | Given name (first word), titles like "sir" and "captain" removed |
| Cleaning | Accents to plain letters, lowercase, a to z plus `'` and `-`, length 2 to 15 |
| Kept | 20,785 entries, 15,483 unique names |
| Split (seed 723) | train 12,386 · val 1,548 · test 1,549, no name in two splits |
| Vocabulary | 28 characters, mean length 6.3 |

`python scripts/prepare_data.py --unit full` keeps full names instead.
`DnD_character_raw_output.txt` is char-RNN output from the dataset author. It is **not**
training data; it is only used to test the scorer.

## Scorer (v2)

A sample is creative if it is **novel and plausible**. Full reasoning in `docs/evaluation_plan.md`.

**Novelty** is four deterministic lookups against the training set. `tier` is the first that applies:

| Tier | Rule | Example |
| --- | --- | --- |
| copy | exact training name | `thorin` |
| respelling | same sound key as a training name (`cvl.novelty.sound_key`) | `khris` → `chris` |
| near_copy | normalized Levenshtein similarity above 5/6 | `magnuss` → `magnus` |
| derived | embedding cosine to nearest training name at or above 0.875 (`all-MiniLM-L6-v2`) | `eldrath` → `eldra` |
| famous | on `data/external/famous_fantasy_names.txt` and not in train | `aragorn` |
| novel | none of the above | 83% of real validation names |

**Plausibility**, for novel names only: length 3 to 15, then either the reference 3-gram
percentile (`plausibility_mode: "ngram"`, the current setting) or the cached LLM judge
(`"judge"`, `cvl/judge.py`, one name per call, temperature 0, ratings in `judge_cache.csv`).
The judge becomes the setting only if it passes the acceptance test against our own ratings
(`scripts/human_tasks.py analyze --freeze`). `creativity_rate_ngram` is always reported.

**Judge, student, estimate.** The judge returns a probability for each rating 1 to 5 and we use
the expected value. Judging every sample of every checkpoint would be hundreds of thousands of
API calls, so a small local model, the student (`student.pt`), is trained to copy the judge.
The student alone over-rates names by 6 to 13 points, by an amount that changes with training
step, so it is never reported by itself. `eval_checkpoint.py` has the student score every novel
name, sends a random 200 to the judge, and uses those 200 to remove the student's error. The
result is `creativity_rate_judge` with a standard error. Full explanation and measurements:
`docs/judge_and_student.md`.

| Batch measure | Definition |
| --- | --- |
| creativity_rate | unique creative names / total samples (**headline number**) |
| diversity | unique names / total samples |
| copy_rate, respelling_rate, near_copy_rate, derived_rate, famous_rate, novel_rate | share of samples in each tier |
| gate_fail_rate | share of samples outside 3 to 15 letters |
| heldout_hit_rate | share of samples exactly matching a val or test name |

How the deterministic settings were chosen (`scripts/calibrate.py`, from real validation names,
before any model output):

* **n-gram order 3**: lowest validation perplexity among orders 2 to 6 (11.9; order 4 was 12.05).
* **n-gram plausibility at least 0.10**: the 10th percentile of real validation names.
* **spelling novelty at least 1/6**: 90% of real validation names clear it.
* **meaning cosine below 0.875**: 95% of real validation names are below it.

## Sanity check (`python scripts/sanity_check.py`, about 1 minute with RapidFuzz)

Scorer v2 with the 3-gram as the plausibility signal, which is the setting until the judge is
switched on at the freeze. These numbers will be regenerated then.

| 1,000 names from | creativity rate | novel rate | copy rate | diversity |
| --- | --- | --- | --- | --- |
| real held-out names | 0.718 | 0.830 | 0.000 | 1.000 |
| char-RNN output | 0.614 | 0.670 | 0.183 | 0.952 |
| bigram model | 0.373 | 0.805 | 0.055 | 0.830 |
| shuffled letters of real names | 0.236 | 0.974 | 0.003 | 1.000 |
| one-letter edits of training names | 0.212 | 0.420 | 0.020 | 1.000 |
| random strings | 0.024 | 0.993 | 0.001 | 1.000 |
| 20 names repeated | 0.017 | 1.000 | 0.000 | 0.020 |
| copied training names | 0.000 | 0.000 | 1.000 | 1.000 |

Known limits:

* The spelling view cannot flag a one-letter change in a name of six letters or fewer, and it
  should not: every real 3-letter validation name is one edit from a training name. For short
  names the copy, sound and meaning views carry novelty.
* The 3-gram is trained on the same split as the models, so it rewards what validation loss
  rewards. In `outputs/ngram_order_sweep.csv` creativity peaks next to the loss minimum. It
  also rates repeated letters ("aaaaaa") as plausible and gives shuffled letters 0.24. This is
  why the judge exists, and why the 3-gram number is kept only as a side column after the freeze.
* The meaning view is used only at the high end. For nonsense strings the cosine carries no
  information.

## Figures

```python
from cvl.plots import plot_training_curve, plot_plane
plot_training_curve(df, "fig1.png")   # df columns: step, val_loss, creativity_rate
plot_plane(df, "fig2.png", label_col="step")
```

`outputs/ngram_order_sweep.png` is a plumbing test of Fig. 1 using n-gram order in place of
training step. It is not a result.

## Human study

Plausibility is the only part of the score that is a judgment call, so it is grounded on our own
ratings. Each of us rates the same items, blind, in about 15 minutes.

**To rate:** open https://bloohunnits.github.io/723finalproject/ on your phone or laptop, tap
through it, and send Andre the block of text it shows at the end. It is an ordinary web page: no
account, no install. The page is `outputs/human_study/rating_app.html`, served from the
`gh-pages` branch of Andre's fork. Full instructions are in `outputs/human_study/INSTRUCTIONS.md`.

| Part | Items | Question | Used for |
| --- | --- | --- | --- |
| 1 | 120 names, rated 1 to 5 | could this be a fantasy character's given name? | half set the judge's cut, half test the judge |
| 2 | 60 pairs, same or different | is the first name just a variant of the second? | checking that the novelty tiers flag what a person would call the same name |

What the app does that a spreadsheet does not: each rater gets their own random order, ten names
are shown a second time at least 40 items later (to measure how consistent each rater is with
themselves), and the time spent on each item is recorded. The app holds only ids and names. Where
a name came from is in the key files, which are not in the repo.

**To analyze** (Andre):

```bash
python scripts/human_tasks.py import ratings_from_vraj.txt      # once per rater
python scripts/human_tasks.py analyze --fill-judge --freeze
```

`import` turns a pasted result into `rater_<name>_A.csv`, `_B.csv` and `_repeats.csv`, and refuses
an incomplete one. `analyze` reports agreement between raters, picks the judge's cut on the
calibration half, runs the acceptance test on the test half, and with `--freeze` writes the result
into `scorer_config.json`. `--fill-judge` sends the 120 Part 1 names to the judge (120 small API
calls, needs `OPENAI_API_KEY`). `make` regenerates the sheets and keys deterministically.

## Layout

```
cvl/data.py        cleaning, splitting, load_split
cvl/ngram.py       reference character n-gram
cvl/novelty.py     novelty views: spelling, sound, meaning, splice
cvl/judge.py       LLM judge: soft ratings, on-disk cache (judge_cache.csv)
cvl/student.py     the judge's local copy (student.pt)
cvl/estimate.py    student on all names + judge on a sample -> unbiased plausibility share
cvl/reference_lm.py  small LSTM fixture for building the scorer; not a project model
cvl/agreement.py   Krippendorff's alpha, Cohen's kappa
cvl/scorer.py      score, score_table
cvl/plots.py       Fig. 1 and Fig. 2 helpers
models/lstm.py     the project's LSTM in three sizes (25k, 100k, 400k parameters)
training/train_lstm.py   trains one run or the whole grid, log-spaced checkpoints
scripts/           prepare_data, calibrate, sanity_check, human_tasks, eval_checkpoint, make_figures,
                   summarize_runs, sensitivity, judge_pool, train_student, train_reference_lstm
results/           checkpoints.csv (one row per scored checkpoint), summary.md, figures/, samples/,
                   pool/ (names sent to the judge), student/ (student vs. judge), reference/ (fixture run)
tests/             unit tests
docs/              evaluation_plan.md (scorer design), judge_and_student.md (plausibility pipeline),
                   work_split.md (who does what, dates)
```
