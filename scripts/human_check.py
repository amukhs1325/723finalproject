"""Human check of the scorer.

Make the rating sheet (60 names, shuffled, sources hidden):
    python scripts/human_check.py make --samples a.txt b.txt c.txt --n 60

Each teammate copies outputs/human_check/sheet.csv, fills the rating column
(1 = not creative, 5 = very creative: new and still sounds like a fantasy name),
and saves it as rater_<name>.csv in the same folder. Then:
    python scripts/human_check.py analyze

Reports agreement between raters and the Spearman correlation between the
mean human rating and the scorer's novelty, plausibility and creative flag.
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT  # noqa: E402
from cvl.scorer import clean_sample, default_scorer  # noqa: E402

OUT = ROOT / "outputs" / "human_check"


def make(sample_files: list[str], n: int, seed: int) -> None:
    rng = random.Random(seed)
    pools = {}
    for f in sample_files:
        names = [clean_sample(x) for x in Path(f).read_text().splitlines()]
        pools[Path(f).stem] = list(dict.fromkeys(x for x in names if x))
    per = max(1, n // len(pools))
    picked = []
    for src, names in pools.items():
        for name in rng.sample(names, min(per, len(names))):
            picked.append((name, src))
    rng.shuffle(picked)
    picked = picked[:n]

    scorer = default_scorer()
    key = pd.DataFrame([{"id": i + 1, "name": nm, "source": src, **scorer.score_name(nm)}
                        for i, (nm, src) in enumerate(picked)])
    OUT.mkdir(parents=True, exist_ok=True)
    key.to_csv(OUT / "key.csv", index=False)
    key[["id", "name"]].assign(rating="").to_csv(OUT / "sheet.csv", index=False)
    print(f"wrote {OUT / 'sheet.csv'} ({len(key)} names) and the hidden key.csv")


def analyze() -> None:
    key = pd.read_csv(OUT / "key.csv")
    raters = sorted(OUT.glob("rater_*.csv"))
    if not raters:
        sys.exit(f"no rater_*.csv files in {OUT}")
    ratings = pd.concat([pd.read_csv(f)[["id", "rating"]].rename(columns={"rating": f.stem[6:]})
                         .set_index("id") for f in raters], axis=1)
    print(f"{len(raters)} raters: {', '.join(ratings.columns)}")
    if ratings.shape[1] > 1:
        print("\nrater agreement (Spearman):")
        print(ratings.corr(method="spearman").round(2).to_string())
    df = key.set_index("id").join(ratings.mean(axis=1).rename("human"))
    df["creative"] = df["creative"].astype(int)
    print("\nmean human rating vs. scorer (Spearman):")
    for col in ["novelty", "plausibility", "creative"]:
        print(f"  {col:13s} {df['human'].corr(df[col], method='spearman'):+.2f}")
    print("\nmean human rating by source:")
    print(df.groupby("source")["human"].mean().round(2).to_string())
    df.to_csv(OUT / "analysis.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--samples", nargs="+", required=True, help="text files, one name per line")
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--seed", type=int, default=723)
    sub.add_parser("analyze")
    a = ap.parse_args()
    make(a.samples, a.n, a.seed) if a.cmd == "make" else analyze()
