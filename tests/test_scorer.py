"""Scorer tests. Run: python -m pytest tests  (or: python tests/test_scorer.py)"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvl.data import clean_name, load_split  # noqa: E402
from cvl.ngram import CharNGram  # noqa: E402
from cvl.novelty import NearestName, levenshtein, similarity  # noqa: E402
from cvl.scorer import Scorer, default_scorer  # noqa: E402

TRAIN = ["dave", "elaria", "thorin", "magnus", "milo", "rowan", "kestrel", "zildric"]
VAL = ["davos", "elara", "thoran", "mira", "rowen", "kaelen"]
TEST = ["garn", "maegil"]


def small_scorer(novelty_threshold=0.2):
    return Scorer(TRAIN, VAL, TEST, ngram_order=2, novelty_threshold=novelty_threshold,
                  plausibility_threshold=0.1)


def test_clean_name():
    assert clean_name("Sir Nathandriel Kelemaneril") == "nathandriel"
    assert clean_name("Zoë,") == "zoe"
    assert clean_name("The Great Gonzo") == "great"
    assert clean_name("jon39") is None
    assert clean_name("Sir") is None
    assert clean_name("D'alrian Mist") == "d'alrian"
    assert clean_name("Arden Elsewhere", unit="full") == "arden elsewhere"


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0
    assert abs(similarity("dave", "davey") - 0.8) < 1e-12


def test_numpy_matches_reference():
    rng = random.Random(0)
    nn = NearestName(TRAIN, use_rapidfuzz=False)
    for _ in range(200):
        q = "".join(rng.choices("adeiklmnorstvz", k=rng.randint(1, 9)))
        sim, _ = nn.nearest(q)
        assert abs(sim - max(similarity(q, t) for t in TRAIN)) < 1e-12


def test_ngram_is_a_distribution():
    lm = CharNGram(3).fit(TRAIN)
    symbols = sorted(lm.vocab)
    for ctx in ["^^", "da", "zz", "lo"]:
        total = sum(lm.prob(ctx, s) for s in symbols)
        assert 0.97 < total <= 1.0 + 1e-12


def test_copies_are_not_creative():
    s = small_scorer()
    r = s.score(["dave"] * 3 + ["thorin"])
    assert r["copy_rate"] == 1.0
    assert r["creativity_rate"] == 0.0
    assert r["novelty"] == 0.0


def test_gibberish_is_not_plausible():
    s = small_scorer()
    assert s.score_name("xqzzkt")["plausibility"] < s.score_name("elarin")["plausibility"]
    assert not s.score_name("qqqqxj")["creative"]


def test_repeats_lower_creativity_and_diversity():
    s = small_scorer(novelty_threshold=0.0)
    varied = s.score(["elarin", "thoris", "maglo", "rowik"])
    repeated = s.score(["elarin"] * 4)
    assert repeated["diversity"] == 0.25
    assert repeated["creativity_rate"] <= 0.25
    assert varied["diversity"] == 1.0


def test_heldout_hits_and_cleaning():
    s = small_scorer()
    r = s.score(["Elara\n", " garn$ "])
    assert r["heldout_hit_rate"] == 1.0
    assert r["copy_rate"] == 0.0


def test_threshold_tolerance():
    s = Scorer(TRAIN, VAL, TEST, ngram_order=2, novelty_threshold=1 - 5 / 6)
    # "kestrels" is one insertion from "kestrel": novelty exactly 1/8 < 1/6
    assert not s.score_name("kestrels")["novel"]
    # "milon" vs "milo": 1/5 >= 1/6
    assert s.score_name("milon")["novel"]


def test_adjust_temperature_and_top_p():
    import numpy as np
    from cvl.sampling import adjust
    p = np.array([0.5, 0.3, 0.15, 0.05])
    assert np.allclose(adjust(p), p)
    assert adjust(p, temperature=2.0)[3] > p[3]          # flatter
    assert adjust(p, temperature=0.5)[0] > p[0]          # sharper
    assert np.allclose(adjust(p, top_p=0.8), [0.625, 0.375, 0, 0])
    assert np.allclose(adjust(p, temperature=0), [1, 0, 0, 0])
    batch = adjust(np.stack([p, p[::-1]]), top_p=0.5)
    assert np.allclose(batch.sum(1), 1)


def test_ngram_checkpoint_and_sample():
    from cvl.sampling import sample
    m = CharNGram(2).fit(TRAIN)
    m2 = CharNGram.from_dict(m.to_dict())
    assert abs(m.perplexity(VAL) - m2.perplexity(VAL)) < 1e-9
    a = sample(m2, 20, seed=3)
    assert a == sample(m2, 20, seed=3)                  # same seed, same names
    assert all("$" not in x and "^" not in x for x in a)


def test_project_scorer_loads():
    """Uses the real frozen config; skipped quietly if data is not prepared."""
    try:
        s = default_scorer()
    except FileNotFoundError:
        return
    real = load_split("test")[:50]
    r = s.score(real)
    assert r["copy_rate"] == 0.0 and r["heldout_hit_rate"] == 1.0
    assert r["creativity_rate"] > 0.5


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
