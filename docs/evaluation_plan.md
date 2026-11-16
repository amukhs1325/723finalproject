# Evaluation plan: creativity score v2

Owner: Andre. Status: **steps 1 to 4 and 7 of the build order are implemented on branch
`andre/scorer-v2`. Waiting on ratings (step 5) before the freeze (step 6).**
Written 2026-11-16 against commit `7efdbb0`.

## What stays, what changes

The proposal defines a creative sample as **novel and plausible**, with uniqueness counted at the
batch level. Novelty is deterministic (distance to the nearest training name clears a threshold),
plausibility comes from an LLM judge calibrated to human ratings, and the judge only sees names
that already passed novelty. The current scorer implements the first half with one view (spelling)
and replaces the judge with a character 3-gram.

| | v1 (in repo) | v2 (this plan) |
| --- | --- | --- |
| Novelty | spelling only: 1 − normalized Levenshtein similarity to nearest training name | four views: copy, spelling, sound, meaning. A name is novel only if it clears all four, and is not on an outside famous-names list |
| Plausibility | percentile under a 3-gram trained on the training set | length gate, then an LLM judge grounded on human ratings. The 3-gram stays as a reported side column |
| Uniqueness | unique creative names / samples | unchanged |
| Human study | 60 control-set names rated for "creativity" | two short tasks: plausibility ratings and same-name pair judgments |
| Splits | seed 723 | **unchanged**, so model training is never blocked or invalidated by this work |

Why v1 is not enough: the 3-gram is trained on the same data as the models, so plausibility rewards
the same thing validation loss rewards. In `outputs/ngram_order_sweep.csv` creativity peaks at the
order with the lowest validation loss. Reporting only that score would build the answer to RQ1 into
the metric.

## 1. Novelty

Each generated name is compared with the 12,386 training names in four ways. Every view is a
nearest-neighbor query with a fixed rule, so the result is deterministic.

| View | Query | Flags a name when | Catches |
| --- | --- | --- | --- |
| Copy | exact match | it is a training name | memorization |
| Spelling | normalized Levenshtein similarity (already in `cvl/novelty.py`) | similarity > 5/6 | `magnuss` → `magnus` |
| Sound | respelling key: fold `ph→f`, `ck/kh/q→k`, soft `c→s`, `y→i`, doubled consonants, final silent `e`, then exact match on the key | its key equals a training name's key | `khris` → `chris`, `jonn` → `jon`, `kaius` → `caius` |
| Meaning | cosine to nearest training name in a fixed sentence-embedding space (`all-MiniLM-L6-v2`, CPU, pinned revision) | cosine ≥ τ, τ = 95th percentile of real validation names (0.875 today) | same root with a new ending: `eldrath` → `eldra`, `jonasz` → `jonas`, `malikian` → `malik` |

Plus one outside check: a list of famous fantasy names that are **not** in the training set
(`aragorn`, `elminster`, ...). A match there is reported as `famous_hit` and is not novel. Without
it a model cannot leak these, but the judge will love them, so the list protects the judge stage.

Tier label per sample, first match wins: `copy`, `respelling` (sound), `near_copy` (spelling),
`derived` (meaning), `famous`, `novel`. The continuous `novelty` column stays as it is.

One descriptive flag, not a filter: `splice` (a prefix of one training name joined to a suffix of
another, both at least 3 letters). It is reported for novel names only.

### What the prototype measured (real validation names, n = 1,548)

| Rule | Share of real unseen names it flags |
| --- | --- |
| Spelling, similarity > 5/6 | 6.8% |
| Sound, respelling key | 7.8% |
| Spelling or sound | 13.0% |
| Meaning at τ = 0.875, beyond the two above | +3.5% |
| **All views together** | **about 16.5%, so about 83.5% of real unseen names count as novel** |

Rules that were tested and rejected, with the reason:

* **Double Metaphone equality** (named in our planning notes) flags 76% of real unseen names and
  30% of random strings. It drops vowels, so `aku` matches `ack`. Even restricted to pairs within
  two edits it flags 49%. Too coarse to be a filter. The respelling key does the intended job.
* **Recombination as a filter.** 44% of real unseen names (68% of those with six or more letters)
  are already a training prefix plus a training suffix; the bigram model manages 5%. Recombination
  is what real names look like, so it is a descriptor.
* **"One edit away" as near-copy for short names.** Every real 3-letter validation name and 90% of
  4-letter ones are one edit from a training name. Short-name space is saturated, so for names of
  six letters or fewer the copy, sound and meaning views carry novelty. This is a property of the
  data, and it goes in the report.
* **Meaning view on invented strings.** Spearman correlation with spelling similarity is 0.27, so
  it is a different signal, but for nonsense strings the cosine is uninformative (random strings
  average 0.63 against 0.73 for real names). It is used only at the high end, as a "same root"
  detector, never as a general distance.

Thresholds are set from real validation names only, before any neural checkpoint is scored.

## 2. Plausibility

Applied only to names that passed novelty.

1. **Gate (deterministic).** Length 3 to 15 and the 28-character alphabet. Two-letter names are
   0.3% of the training set but were 14% of bigram samples and passed v1 68% of the time.
2. **LLM judge.** One question per name: *could this be the given name of a character in a
   fantasy role-playing game?* Answer 1 to 5. Fixed rubric, temperature 0, one pinned model
   snapshot (`gpt-4.1-mini-2025-04-14`), **one name per call**. Output is cached in
   `judge_cache.csv` (name, rating, model, prompt hash) and committed, so every later score is a
   table lookup and anyone can reproduce our numbers without an API key.
   Why one name per call: in a pilot on 63 names outside the human study, rating 25 names per
   call changed the rating of 30% of names when their batch-mates changed (always by one point).
   That would let the quality of a checkpoint's other samples leak into each rating, which is
   the exact comparison this project makes. Alone, all 63 names got the same rating twice.
3. **Plausible** means rating ≥ c. The cut c is chosen on the calibration half of the human
   ratings to best match the human majority, then never moved.

The 3-gram percentile stays in the output as `ngram_plausibility` for comparison and as the
fallback (see acceptance test).

## 3. Human grounding

Two tasks, the same items for all three of us, about 20 minutes each. Both can be built today from
material already in the repo. Neither waits for a neural model.

**Task A, plausibility (120 names).** All novelty-passing, drawn to span the range: real held-out
names, the dataset author's char-RNN output, n-gram samples at orders 2, 3 and 5, shuffled-letter
and random strings. Rate 1 to 5 on the judge's exact question. Sources hidden, order shuffled.
Split 60 calibration / 60 test by a fixed seed.

**Task B, same name? (60 pairs).** A candidate and the training name a view matched it to, drawn
from each view's flagged set, from just under each threshold, and from clearly distant pairs.
Answer yes or no: *is the left name just a respelling or trivial variant of the right?*

What gets reported:

| Number | From | Purpose |
| --- | --- | --- |
| Krippendorff's α (ordinal for A, nominal for B) | all three raters | the inter-annotator agreement the report template requires |
| Spearman of judge vs. mean human rating, test half | Task A | is the judge grounded |
| Same for the 3-gram percentile | Task A | does the judge beat the cheap signal |
| Accuracy and Cohen's κ of `plausible` vs. human majority, test half | Task A | quality of the final yes/no |
| Precision of each novelty view against the human majority | Task B | do the flags mean what we say |

Later, once checkpoints exist: 60 more names from an early, a lowest-loss and an overfit checkpoint,
rated the same way. The judge is **not** retuned on them. This only checks that calibration
transfers to real model output.

**Acceptance test, fixed now.** The judge becomes the plausibility signal if, on the test half, its
Spearman with the human mean is within 0.10 of the average human-to-human Spearman and is higher
than the 3-gram's. If it fails, the 3-gram with the length gate is the plausibility signal and the
report says so.

## 4. The score

Per name: `tier`, `novel`, `gate_ok`, `judge_rating`, `plausible`, `creative = novel and plausible`,
plus `novelty`, `nearest`, `splice`, `ngram_plausibility`.

Per batch (1,000 samples per checkpoint, temperature 1, top-p 1, fixed seed):

* `creativity_rate` = unique creative names / samples. **Headline.**
* `diversity` = unique names / samples.
* Tier shares: `copy_rate`, `respelling_rate`, `near_copy_rate`, `derived_rate`, `famous_rate`,
  `novel_rate`.
* `plausible_rate_given_novel`, `mean_judge_rating`.
* `creativity_rate_ngram`: the v1-style number, kept so the two can be plotted together.

Figures this enables: loss and creativity vs. step; tier shares vs. step as a stacked area, which
shows memorization growing; novelty against plausibility per checkpoint.

## 5. Interface for everyone else

Nothing teammates call changes shape.

```python
from cvl.scorer import score, score_table     # same names, more columns
python scripts/eval_checkpoint.py <run dir>   # sample, score novelty, judge cache misses, write results/checkpoints.csv
```

Scoring is two-phase so training code never calls an API: novelty and the gate run offline; names
missing from the judge cache are collected and judged in one batch step; then rates are computed.
Recovery methods that need labels (rejection-sampling fine-tuning) read the same cache.

## 6. Build order

| Step | Output | Needs |
| --- | --- | --- |
| 1 | Sound and meaning views, tier labels, famous list, tests | nothing |
| 2 | Length gate, pluggable plausibility (3-gram or judge), new columns | step 1 |
| 3 | Task A and Task B sheets, analysis script with α and κ | step 1 |
| 4 | Judge module: prompt, batching, cache | a model decision (below) |
| 5 | Ratings from all three of us | step 3 |
| 6 | Calibrate c, run acceptance test, write `scorer_config.json` v2, rerun sanity check and bigram baseline, **freeze** | steps 4 and 5 |
| 7 | `scripts/eval_checkpoint.py`, `results/checkpoints.csv` | step 2; tested on the n-gram checkpoints |

Steps 1 to 4 and 7 need no one else. Model training can run the whole time on the existing splits.
Checkpoints trained before the freeze are simply scored after it.

## 7. Risks to state in the report

* The judge was calibrated on names from weak generators and real names. The 60-name follow-up on
  real checkpoints is the check on that.
* Fine-tuning on names the judge liked and then grading with the judge is circular. Recovery
  results also get a small human-rated sample.
* The embedding model and judge are outside models. Both are pinned and the judge's outputs are
  cached in the repo.
* Dataset contains real-world names (`dave`, `kevin`) and a few non-names (`unknown`, `and`, `of`).
  The split is left alone; they are noted in the dataset section and the error analysis.

## 8. Decisions needed before step 4

1. Settled: `gpt-4.1-mini-2025-04-14` on Andre's OpenAI key. One name per call means roughly
   one call per unique novel name across the project (on the order of 200,000 if every checkpoint
   of every run is judged at 1,000 samples).
2. Team agreement that the scorer changes once more and then freezes at step 6. The README
   currently says it is already frozen; only the bigram baseline has been scored so far.
