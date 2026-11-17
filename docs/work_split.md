# Dividing the rest of the work

Proposed 2026-11-16. Report due Friday 18 December. Change anything here in review.

The idea is that nobody waits on anybody. We agree on a small interface once, then each
model has its own training script, and the scorer reads whatever checkpoints exist.

## Who owns what

| Owner | Area | Files |
| --- | --- | --- |
| Andre | Scoring, human study, results, error analysis | `cvl/scorer.py`, `cvl/novelty.py`, `cvl/judge.py`, `scripts/eval_checkpoint.py`, `scripts/human_tasks.py`, `results/` |
| Agastya | Shared interface, LSTM runs | `cvl/vocab.py`, `cvl/sampling.py`, `models/lstm.py`, `training/train_lstm.py` |
| Vraj | Transformer runs, decoding baseline, recovery | `models/transformer.py`, `training/train_transformer.py`, `experiments/decoding.py`, `experiments/recovery.py` |

The report asks which files each of us wrote, so keep ownership as listed.

## Tasks

**Andre**

| | Task | Needs |
| --- | --- | --- |
| A1 | Scorer v2, rating sheets, checkpoint evaluation | done in this PR |
| A2 | Freeze the scorer: calibrate the judge on our ratings, run the acceptance test, regenerate sanity check and baseline | ratings from all three of us |
| A3 | Figure script that redraws everything from `results/checkpoints.csv`: loss and creativity vs. step, tier shares vs. step, by size, by architecture | nothing; built on the n-gram rows first |
| A4 | Follow-up ratings: 60 names from an early, a lowest-loss and an overfit checkpoint, to check the judge transfers to real model output | first checkpoints |
| A5 | Error analysis: hand-label 150 generated names (copy, respelling, fragment, gibberish, real-world name, good) | generated names |

**Agastya**

| | Task | Needs |
| --- | --- | --- |
| G1 | Interface PR (below). Give Andre and Vraj write access to the repo. | nothing |
| G2 | `models/lstm.py` and its training script. One medium run first, checkpoints log-spaced, trained well past the validation-loss minimum. | G1 |
| G3 | Full LSTM grid: 3 sizes × 3 seeds | G2 |
| G4 | Optional: MLP with a fixed context window on the same script, to keep the proposal's model ladder | G2 |

**Vraj**

| | Task | Needs |
| --- | --- | --- |
| V1 | `models/transformer.py` with its own training script, sizes matched to the LSTM parameter counts. One medium run first. | the interface text below, not Agastya's code |
| V2 | Full transformer grid: 3 sizes × 3 seeds | V1 |
| V3 | Generalize the bigram decoding sweep to any checkpoint. Run it on the lowest-loss LSTM and transformer. | written against `checkpoints/bigram.json` today |
| V4 | Rejection-sampling fine-tuning: sample from the lowest-loss checkpoint, keep the names scored creative, fine-tune, compare with the best decoding setting at matched validation loss | written against a toy model, then pointed at real checkpoints |
| V5 | Stretch: one more recovery method (unlikelihood or DPO) | V4 |
| V6 | Overleaf and the related-work list. The report needs 10 citations; the proposal has 5. Everyone sends two. | nothing |

**Everyone**

* Rate both sheets in `outputs/human_study/` (about 20 minutes).
* Write your own report sections in your own words. AI help is allowed for code, experiments
  and figures and has to be listed in the AI Disclosure section. It is not allowed for the
  report text.

## The interface to agree on (G1)

```
cvl/vocab.py      SYMBOLS = ["$", "'", "-", "a".."z"]   # 29 outputs, "$" ends a name
                  BOS id = 29 (input only) · encode(name) · decode(ids)

every model       forward(ids) -> logits over the 29 outputs
                  val loss in nats per symbol, end marker counted

checkpoints/{arch}_{size}_s{seed}/step{N}.pt   # state_dict, config, step, val_loss
checkpoints/{arch}_{size}_s{seed}/log.csv      # step, train_loss, val_loss

cvl/sampling.py   load_checkpoint() opens .pt and returns a model with
                  symbols, next_probs(context) and meta = {"step", "val_loss"}
                  (that is all scripts/eval_checkpoint.py needs)

sizes             small ≈ 25k · medium ≈ 100k · large ≈ 400k parameters, matched across architectures
seeds             0, 1, 2
```

Each model keeps its own training loop. A loop is about forty lines; sharing one would make
the transformer wait on the LSTM. Weights stay out of git. Result CSVs and figures go in.

## Dates

| By | What is true | Who |
| --- | --- | --- |
| Tue 17 Nov | Interface PR merged. Everyone can push. | Agastya |
| Thu 19 Nov | **Sync 1.** One LSTM run and one transformer run through `eval_checkpoint.py`: the first real figure. | all |
| Sun 22 Nov | Ratings in from all three. Judge calibrated, scorer frozen. | all, then Andre |
| Wed 25 Nov | **Sync 2.** Both grids trained and scored. Decoding sweep done on neural checkpoints. | all |
| Tue 1 Dec | Recovery results. Error analysis labeled. Results frozen. | Vraj, Andre |
| 2 to 9 Dec | HW2 due on the 2nd, Exam 2 on the 9th. Draft your own sections in the gaps. | all |
| Wed 16 Dec | Full 8-page draft. Pangram check on the PDF and on pasted text. | all |
| Fri 18 Dec | PDF to Gradescope. Public repo link emailed to the course staff. | all |

## Cut from the proposal, to list in the report

The report has a required "proposed vs. accomplished" section. As things stand these are out,
each needs a one-line reason there: the town-name dataset, the MLP (unless G4 happens),
unlikelihood training and DPO (unless V5 happens).

## One trap

Fine-tuning on names the scorer liked and then grading with the same scorer is circular. The
recovered model also gets a small human-rated sample.
