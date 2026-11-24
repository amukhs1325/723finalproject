"""Pick the reference n-gram order and set the deterministic thresholds, once.

    python scripts/calibrate.py           # refuses to overwrite an existing config
    python scripts/calibrate.py --force   # only before any experiment has been scored

Rules (decided before seeing any model output):
  * n-gram order: the order with the lowest perplexity on the validation split.
  * plausibility threshold: 0.10, i.e. the 10th percentile of real validation names.
  * novelty threshold: the novelty that real validation names clear 90% of the time.
  * meaning threshold: the 95th percentile of real validation names' cosine to their
    nearest training name. Only the top 5% of real names are that close to a training name.
  * length gate: 3 to 15 letters (two-letter names are 0.3% of the training split).

The judge cut is not set here. It comes from human ratings: scripts/human_tasks.py freeze.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import PROCESSED_DIR, load_split  # noqa: E402
from cvl.ngram import CharNGram  # noqa: E402
from cvl.judge import JUDGE_MODEL, PROMPT_HASH  # noqa: E402
from cvl.novelty import MEANING_MODEL, MEANING_REVISION, MeaningIndex, NearestName  # noqa: E402
from cvl.scorer import CONFIG_PATH, split_fingerprint  # noqa: E402

PLAUSIBILITY_PERCENTILE = 0.10
NOVELTY_CLEAR_RATE = 0.90
MEANING_PERCENTILE = 0.95
MIN_LEN, MAX_LEN = 3, 15

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if CONFIG_PATH.exists() and not args.force:
        sys.exit(f"{CONFIG_PATH.name} already exists; thresholds are frozen. Use --force only "
                 "if no experiment has been scored yet.")

    train, val = load_split("train"), load_split("val")

    ppl = {k: CharNGram(k).fit(train).perplexity(val) for k in range(2, 7)}
    order = min(ppl, key=ppl.get)
    print("val perplexity by order:", {k: round(v, 2) for k, v in ppl.items()}, "-> order", order)

    nn = NearestName(train)
    print(f"computing novelty for {len(val)} validation names ...")
    nov = np.array([nn.novelty(n) for n in val])
    novelty_threshold = float(np.quantile(nov, 1 - NOVELTY_CLEAR_RATE))
    print("val novelty percentiles (10/25/50/75/90):",
          np.round(np.quantile(nov, [.1, .25, .5, .75, .9]), 3).tolist())

    meaning = MeaningIndex(train, cache_dir=PROCESSED_DIR / ".cache")
    cos = np.array([s for s, _ in meaning.nearest_many(val)])
    meaning_threshold = float(np.quantile(cos, MEANING_PERCENTILE))
    print("val cosine-to-nearest percentiles (50/90/95/97.5):",
          np.round(np.quantile(cos, [.5, .9, .95, .975]), 3).tolist())

    cfg = {
        "version": 2,
        "ngram_order": order,
        "novelty_threshold": novelty_threshold,  # full precision: values sit on fractions like 1/6
        "plausibility_threshold": PLAUSIBILITY_PERCENTILE,
        "meaning_model": MEANING_MODEL,
        "meaning_revision": MEANING_REVISION,
        "meaning_threshold": meaning_threshold,
        "min_len": MIN_LEN,
        "max_len": MAX_LEN,
        "plausibility_mode": "ngram",      # switched to "judge" by human_tasks.py freeze
        "judge_model": JUDGE_MODEL,
        "judge_prompt_hash": PROMPT_HASH,
        "judge_cut": None,
        "train_fingerprint": split_fingerprint(train),
        "notes": {
            "order_rule": "lowest validation perplexity among orders 2-6",
            "val_perplexity": {str(k): round(v, 3) for k, v in ppl.items()},
            "novelty_rule": f"{NOVELTY_CLEAR_RATE:.0%} of real validation names clear it",
            "val_novelty_quantiles": {str(q): round(float(np.quantile(nov, q)), 4)
                                      for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)},
            "plausibility_rule": "10th percentile of real validation names",
            "meaning_rule": f"{MEANING_PERCENTILE:.0%} of real validation names are below it",
            "val_cosine_quantiles": {str(q): round(float(np.quantile(cos, q)), 4)
                                     for q in (0.5, 0.75, 0.9, 0.95, 0.975)},
        },
    }
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"wrote {CONFIG_PATH.name}: order={order}, novelty>={cfg['novelty_threshold']}, "
          f"plausibility>={PLAUSIBILITY_PERCENTILE}, meaning<{meaning_threshold:.4f}")
