"""Does the scorer rank obvious cases the right way round?

    python scripts/sanity_check.py

Scores 1,000 names from each test set and writes:
  outputs/sanity_check.csv         one row per set
  outputs/sanity_plane.png         sets on the novelty vs. plausibility plane
  outputs/ngram_order_sweep.png    plumbing test of the training-curve plot
  outputs/samples/*.txt            the samples, reusable for the human check

Expected ordering:
  real held-out names     high novelty, high plausibility, high creativity
  copied training names   novelty 0, creativity 0
  one-letter edits        mostly below the novelty threshold
  shuffled letters        low plausibility
  random strings          lowest plausibility
  bigram / char-RNN       in between
  20 names repeated       low diversity, low creativity rate
"""
import math
import random
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, clean_name, load_split  # noqa: E402
from cvl.ngram import CharNGram  # noqa: E402
from cvl.plots import plot_plane, plot_training_curve  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402

N = 1000
OUT = ROOT / "outputs"
RNN_FILE = ROOT / "data" / "raw" / "DnD_character_raw_output.txt"


def one_edit(name: str, rng: random.Random) -> str:
    letters = string.ascii_lowercase
    i = rng.randrange(len(name))
    op = rng.choice(["sub", "ins", "del"]) if len(name) > 2 else "sub"
    if op == "sub":
        return name[:i] + rng.choice(letters.replace(name[i], "")) + name[i + 1:]
    if op == "ins":
        return name[:i] + rng.choice(letters) + name[i:]
    return name[:i] + name[i + 1:]


def char_rnn_names(rng: random.Random) -> list[str]:
    """Given names from the repo's char-RNN output (first line of each block)."""
    blocks = RNN_FILE.read_text(encoding="utf-8", errors="replace").split("\n\n")
    names = [clean_name(b.strip().splitlines()[0]) for b in blocks if b.strip()]
    names = [n for n in names if n]
    return rng.sample(names, N)


def main():
    rng = random.Random(723)
    train, test = load_split("train"), load_split("test")
    scorer = default_scorer()

    lengths = [len(n) for n in train]
    sets = {
        "real held-out": rng.sample(test, N),
        "copied train": rng.sample(train, N),
        "one-letter edits": [one_edit(n, rng) for n in rng.sample(train, N)],
        "shuffled letters": ["".join(rng.sample(n, len(n))) for n in rng.sample(test, N)],
        "random strings": ["".join(rng.choices(string.ascii_lowercase, k=rng.choice(lengths)))
                           for _ in range(N)],
        "bigram": [CharNGram(2).fit(train).sample(rng) for _ in range(N)],
        "char-RNN": char_rnn_names(rng),
        "20 names repeated": [rng.choice(test[:20]) for _ in range(N)],
    }

    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    rows = []
    for label, names in sets.items():
        (OUT / "samples" / f"{label.replace(' ', '_')}.txt").write_text("\n".join(names) + "\n")
        r = scorer.score(names)
        rows.append({"set": label, **r})
        print(f"scored {label}")
    df = pd.DataFrame(rows)
    cols = ["set", "creativity_rate", "diversity", "novelty", "plausibility", "novel_rate",
            "plausible_rate", "copy_rate", "heldout_hit_rate"]
    df[cols].to_csv(OUT / "sanity_check.csv", index=False)
    print()
    print(df[cols].round(3).to_string(index=False))

    plot_plane(df.rename(columns={"set": "label"}), OUT / "sanity_plane.png",
               title="Scorer sanity check: 1,000 names per set")

    # Plumbing test for Fig. 1: n-gram order stands in for training step.
    # NOT a result: the order-3 model is judged by the order-3 reference.
    sweep = []
    val = load_split("val")
    for k in range(1, 8):
        lm = CharNGram(k).fit(train)
        names = [lm.sample(rng) for _ in range(N)]
        sweep.append({"step": k, "val_loss": math.log(lm.perplexity(val)),
                      **scorer.score(names)})
    sw = pd.DataFrame(sweep)
    sw.to_csv(OUT / "ngram_order_sweep.csv", index=False)
    print("\nn-gram order sweep (plumbing test, not a result):")
    print(sw[["step", "val_loss", "creativity_rate", "novel_rate", "plausible_rate",
              "copy_rate", "diversity"]].round(3).to_string(index=False))
    plot_training_curve(sw, OUT / "ngram_order_sweep.png",
                        title="Plumbing test: n-gram order in place of training step",
                        x_label="n-gram order (stand-in for training step)")


if __name__ == "__main__":
    main()
