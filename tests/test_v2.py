"""Tests for scorer v2: novelty views, tiers, gate, judge cache, agreement statistics.

Run: python -m pytest tests
None of these call the network or load the embedding model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvl.agreement import cohen_kappa, krippendorff_alpha  # noqa: E402
from cvl.judge import Judge, MissingJudgments  # noqa: E402
from cvl.novelty import SoundIndex, SpliceIndex, sound_key  # noqa: E402
from cvl.scorer import Scorer  # noqa: E402

TRAIN = ["dave", "chris", "jon", "thorin", "magnus", "milo", "rowan", "kestrel", "zildric",
         "eldra", "caius", "sarah"]
VAL = ["davos", "elara", "thoran", "mira", "rowen", "kaelen"]
TEST = ["garn", "maegil"]


class FakeMeaning:
    """Stands in for MeaningIndex: says 'eldrath' is close to 'eldra', nothing else is."""

    def nearest_many(self, names):
        return [(0.95, "eldra") if n == "eldrath" else (0.30, "milo") for n in names]


def scorer(**kw):
    return Scorer(TRAIN, VAL, TEST, ngram_order=2, novelty_threshold=1 - 5 / 6,
                  plausibility_threshold=0.0, **kw)


def test_sound_key_folds_respellings():
    assert sound_key("khris") == sound_key("chris") == sound_key("kris")
    assert sound_key("jonn") == sound_key("jon")
    assert sound_key("davee") == sound_key("dave")
    assert sound_key("kaius") == sound_key("caius")
    assert sound_key("sara") == sound_key("sarah")
    assert sound_key("book") != sound_key("bok")          # vowels inside a name are kept
    assert sound_key("torvus") != sound_key("corvus")     # a different consonant is a different name
    assert SoundIndex(TRAIN).match("khris") == "chris"
    assert SoundIndex(TRAIN).match("zorvek") is None


def test_splice_is_prefix_plus_suffix():
    sp = SpliceIndex(["thorin", "magnus"])
    assert sp.is_splice("thognus")        # tho + gnus
    assert not sp.is_splice("thoxyz")
    assert not sp.is_splice("thori")      # too short to hold two pieces of 3


def test_tiers_first_match_wins():
    s = scorer(meaning=FakeMeaning(), meaning_threshold=0.875, famous={"aragorn", "thorin"})
    tier = lambda n: s.score_name(n)["tier"]  # noqa: E731
    assert tier("thorin") == "copy"           # in train, even though it is also on the famous list
    assert tier("khris") == "respelling"
    assert tier("kestrels") == "near_copy"    # 7/8 similar
    assert tier("eldrath") == "derived"
    assert tier("aragorn") == "famous"
    assert tier("zorvek") == "novel"
    assert "thorin" not in s.famous           # training names are dropped from the list
    for n in ["thorin", "khris", "kestrels", "eldrath", "aragorn"]:
        assert not s.score_name(n)["creative"]
    assert s.score_name("zorvek")["creative"]


def test_length_gate():
    s = scorer()
    assert not s.score_name("zo")["gate_ok"] and not s.score_name("zo")["creative"]
    assert not s.score_name("z" * 16)["creative"]
    assert s.score_name("zor")["gate_ok"]
    r = s.score(["zo", "zorvek", "velmira", "x"])
    assert r["gate_fail_rate"] == 0.5


def test_batch_rates_add_up():
    s = scorer(meaning=FakeMeaning(), meaning_threshold=0.875, famous={"aragorn"})
    r = s.score(["thorin", "khris", "kestrels", "eldrath", "aragorn", "zorvek", "zorvek", "velmira"])
    parts = r["copy_rate"] + r["respelling_rate"] + r["near_copy_rate"] + r["derived_rate"] \
        + r["famous_rate"] + r["novel_rate"]
    assert abs(parts - 1.0) < 1e-12
    assert r["novel_rate"] == 3 / 8
    assert r["creativity_rate"] == 2 / 8      # zorvek counted once
    assert r["diversity"] == 7 / 8


def test_judge_mode_uses_cache_and_only_for_novel_names(tmp_path):
    cache = tmp_path / "judge_cache.csv"
    j = Judge(model="test-model", cache_path=cache)
    j._ratings.update({"zorvek": 5, "velmira": 2})
    j._save()
    j = Judge(model="test-model", cache_path=cache)          # reload from disk
    assert j.rating("zorvek") == 5 and len(j) == 2
    assert Judge(model="other-model", cache_path=cache).rating("zorvek") is None

    s = scorer(plausibility_mode="judge", judge=j, judge_cut=3)
    r = s.score(["zorvek", "velmira", "thorin", "khris", "zo"])   # last three never reach the judge
    assert r["creativity_rate"] == 1 / 5
    assert r["plausible_rate_given_novel"] == 1 / 3               # zorvek yes, velmira no, zo gated
    assert r["mean_judge_rating"] == 3.5
    assert s.needs_judging(["zorvek", "thorin", "khris", "zo", "braxomel"]) == ["braxomel"]
    try:
        s.score(["braxomel"])
        raise AssertionError("expected MissingJudgments")
    except MissingJudgments as e:
        assert e.names == ["braxomel"]


def test_krippendorff_matches_published_example():
    # Krippendorff's 4-rater, 12-item example (None = missing rating).
    a = [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None]
    b = [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3]
    c = [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None]
    d = [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None]
    units = [list(u) for u in zip(a, b, c, d)]
    assert abs(krippendorff_alpha(units, "nominal") - 0.743) < 0.001
    assert abs(krippendorff_alpha(units, "ordinal") - 0.815) < 0.001
    assert krippendorff_alpha([[1, 1], [2, 2], [3, 3]], "ordinal") == 1.0


def test_cohen_kappa():
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-12
