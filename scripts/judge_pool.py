"""Build the pool of names the student model learns from, and have the judge rate it.

    python scripts/judge_pool.py            # build results/pool/offpolicy.csv, list what is unjudged
    python scripts/judge_pool.py --fill     # also send unjudged names to the judge (needs OPENAI_API_KEY)

The pool is 5,000 names that passed novelty and the length gate, from sources that span
the whole plausibility range: real held-out names, the dataset author's char-RNN output,
n-gram samples of several orders, one-letter edits, shuffled letters and random strings.
None of them comes from a model we study, hence "off-policy". Names used in the human
study are left out, so the human test of the student stays clean.
"""
import argparse
import random
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvl.data import ROOT, clean_name, load_split  # noqa: E402
from cvl.judge import Judge  # noqa: E402
from cvl.ngram import CharNGram  # noqa: E402
from cvl.sampling import sample  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402
from human_tasks import OUT as STUDY, RNN_FILE, one_edit  # noqa: E402

POOL = ROOT / "results" / "pool" / "offpolicy.csv"
SEED = 723
QUOTA = {"real held-out": 1200, "char-RNN": 800, "bigram": 500, "3-gram": 700, "4-gram": 500,
         "5-gram": 400, "6-gram": 200, "one-letter edits": 200, "shuffled letters": 300,
         "random strings": 200}


def study_names() -> set[str]:
    """Every name a human rater sees. Never used to train the student."""
    names = set(pd.read_csv(STUDY / "sheet_A.csv")["name"])
    b = pd.read_csv(STUDY / "sheet_B.csv")
    return names | set(b["left"]) | set(b["right"])


def build() -> pd.DataFrame:
    rng = random.Random(SEED)
    scorer = default_scorer()
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    blocks = RNN_FILE.read_text(encoding="utf-8", errors="replace").split("\n\n")
    lengths = [len(n) for n in train]
    src = {
        "real held-out": val + test,
        "char-RNN": [n for n in (clean_name(b.strip().splitlines()[0]) for b in blocks if b.strip()) if n],
        "one-letter edits": [one_edit(n, rng) for n in rng.sample(train, 2000)],
        "shuffled letters": ["".join(rng.sample(n, len(n))) for n in rng.sample(val + test, 1500)],
        "random strings": ["".join(rng.choices(string.ascii_lowercase, k=rng.choice(lengths)))
                           for _ in range(1500)],
    }
    for label, order in (("bigram", 2), ("3-gram", 3), ("4-gram", 4), ("5-gram", 5), ("6-gram", 6)):
        src[label] = sample(CharNGram(order).fit(train), n=6000, seed=SEED + 100 + order)
    used, rows = study_names(), []
    for source, quota in QUOTA.items():
        cand = list(dict.fromkeys(src[source]))
        scorer._prepare(cand)
        ok = [n for n in cand if n not in used and (r := scorer.score_name(n))["novel"] and r["gate_ok"]]
        if len(ok) < quota:
            sys.exit(f"only {len(ok)} usable names from {source}, need {quota}")
        chosen = rng.sample(ok, quota)
        used.update(chosen)
        rows += [{"name": n, "source": source} for n in chosen]
    df = pd.DataFrame(rows)
    POOL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(POOL, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", action="store_true", help="send unjudged names to the judge")
    args = ap.parse_args()
    df = build()
    judge = Judge()
    names = df["name"].tolist() + pd.read_csv(STUDY / "sheet_A.csv")["name"].tolist()
    todo = judge.missing(names)
    print(f"pool: {len(df)} names in {POOL.relative_to(ROOT)}; {len(todo)} names not judged yet "
          "(pool plus the 120 human-study names)")
    if todo and args.fill:
        judge.fill(names)
