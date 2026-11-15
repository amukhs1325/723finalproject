# Creativity vs. Loss

CMSC 723 (Fall 2026) final project by Andre Atkins, Agastya Mukherjee and Vraj Patel.

**Question:** is the language model with the lowest loss also the most creative? We train
small character-level models on fantasy character names, score each training checkpoint for
creativity (names that are new and still sound like real names), and compare where creativity
peaks with where validation loss bottoms out.

**Status:** data, creativity scorer, shared sampling code and the bigram baseline are done.
Next: the shared training loop and the LSTM and transformer runs.

This repo prepares the fantasy-name data and scores generated names for creativity.
Pretraining and post-training code plug into two calls:

```python
from cvl.data import load_split
from cvl.scorer import score, score_table

train = load_split("train")      # list[str], also "val", "test"
result = score(samples)          # samples: list[str] of generated names
# {'n', 'creativity_rate', 'diversity', 'novelty', 'plausibility', 'novel_rate',
#  'plausible_rate', 'copy_rate', 'heldout_hit_rate', 'mean_length'}
table = score_table(samples)     # one row per sample with nearest training name
```

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
python scripts/run_bigram_baseline.py   # about 15 minutes without RapidFuzz
```

Trains a Witten-Bell character bigram on train (`checkpoints/bigram.json`) and runs the
decoding sweep on it: temperature 0.5 to 1.5 and top-p 0.5 to 0.95, 3 seeds × 1,000 names
each. Results in `outputs/baseline/`.

## Setup

```bash
pip install -r requirements.txt   # numpy, pandas, matplotlib; rapidfuzz optional
python scripts/prepare_data.py    # data/processed/{train,val,test}.txt + meta.json
python tests/test_scorer.py       # 10 tests
```

`scorer_config.json` is already committed. **Do not rerun `scripts/calibrate.py`** once any
experiment has been scored: the thresholds are frozen for the whole project.

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

## Scorer

| Measure | Definition |
| --- | --- |
| novelty | 1 minus the highest normalized Levenshtein similarity to any training name (0 = exact copy) |
| plausibility | percentile of the name's mean log-prob under a character 3-gram (Witten-Bell) trained on train, against real validation names |
| creative | novelty at least 1/6 and plausibility at least 0.10 |
| creativity_rate | unique creative names / total samples (**headline number**) |
| diversity | unique names / total samples |
| copy_rate | share of samples exactly matching a training name |
| heldout_hit_rate | share of samples exactly matching a val or test name |

How the frozen settings were chosen (`scripts/calibrate.py`, before any model output):

* **n-gram order 3**: lowest validation perplexity among orders 2 to 6 (11.9; order 4 was 12.05).
* **plausibility at least 0.10**: the 10th percentile of real validation names.
* **novelty at least 1/6**: 90% of real validation names clear it.

## Sanity check (`python scripts/sanity_check.py`, about 3 minutes without RapidFuzz)

| 1,000 names from | creativity rate | diversity | copy rate |
| --- | --- | --- | --- |
| real held-out names | 0.816 | 1.000 | 0.000 |
| char-RNN output | 0.693 | 0.952 | 0.183 |
| bigram model | 0.503 | 0.830 | 0.055 |
| one-letter edits of training names | 0.265 | 1.000 | 0.020 |
| shuffled letters of real names | 0.251 | 1.000 | 0.003 |
| random strings | 0.028 | 1.000 | 0.001 |
| 20 names repeated | 0.017 | 0.020 | 0.000 |
| copied training names | 0.000 | 1.000 | 1.000 |

Known limits:

* About half of one-letter edits count as novel, because one changed letter in a name of
  6 letters or fewer clears 1/6. This matches real data: 43% of real validation names are
  one edit from a training name.
* The 3-gram rates repeated letters ("aaaaaa") about as plausible as real names. Check
  for this in error analysis.
* The reference is trained on the same split as the models, so copied names score a little
  more plausible (0.54) than real unseen ones (0.49). Novelty handles copies separately.

## Figures

```python
from cvl.plots import plot_training_curve, plot_plane
plot_training_curve(df, "fig1.png")   # df columns: step, val_loss, creativity_rate
plot_plane(df, "fig2.png", label_col="step")
```

`outputs/ngram_order_sweep.png` is a plumbing test of Fig. 1 using n-gram order in place of
training step. It is not a result.

## Human check

```bash
python scripts/human_check.py make --samples outputs/samples/*.txt --n 60
# each teammate fills outputs/human_check/sheet.csv -> rater_<name>.csv
python scripts/human_check.py analyze
```

`sheet.csv` (names only) is in the repo. The answer key `key.csv` is not, so ratings stay
blind; Agastya holds it and adds it after everyone has rated.

## Layout

```
cvl/data.py        cleaning, splitting, load_split
cvl/ngram.py       reference character n-gram
cvl/novelty.py     nearest training name (RapidFuzz or NumPy)
cvl/scorer.py      score, score_table
cvl/plots.py       Fig. 1 and Fig. 2 helpers
scripts/           prepare_data, calibrate, sanity_check, human_check
tests/             unit tests
```
