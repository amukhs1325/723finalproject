"""Tests for the judge's local student, the corrected estimator and the reference LSTM.

Run: python -m pytest tests
No network. Models here are tiny and train in a couple of seconds.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cvl.estimate import pick_sample, plausible_share, student_p_plausible  # noqa: E402
from cvl.judge import Judge  # noqa: E402


class FakeStudent:
    """Says every name is 80% likely to be rated 5 and 20% likely to be rated 1."""

    def predict(self, names):
        return np.tile(np.array([0.2, 0, 0, 0, 0.8]), (len(names), 1))


def fake_judge(tmp_path, names, good):
    j = Judge(model="test-model", cache_path=tmp_path / "cache.csv")
    for n in names:
        j.put(n, [0, 0, 0, 0, 1] if n in good else [1, 0, 0, 0, 0])
    return j


def test_student_p_plausible_uses_levels_at_or_above_the_cut():
    p = np.array([[0.1, 0.2, 0.3, 0.25, 0.15]])
    assert abs(student_p_plausible(p, 3.0)[0] - 0.70) < 1e-12
    assert abs(student_p_plausible(p, 3.5)[0] - 0.40) < 1e-12
    assert abs(student_p_plausible(p, 2.5)[0] - 0.70) < 1e-12


def test_sample_is_reproducible_and_ignores_order():
    names = [f"name{i}" for i in range(500)]
    a = pick_sample(names, 50)
    assert a == pick_sample(list(reversed(names)), 50) and len(set(a)) == 50
    assert pick_sample(names, 9999) == sorted(names)


def test_exact_when_every_name_is_judged(tmp_path):
    names = [f"n{i}" for i in range(40)]
    j = fake_judge(tmp_path, names, set(names[:10]))
    r = plausible_share(names, j, FakeStudent(), 3.0, None)
    assert r["estimate"] == 0.25 and r["se"] == 0.0 and r["method"] == "judge, all names"


def test_reports_missing_names_instead_of_guessing(tmp_path):
    names = [f"n{i}" for i in range(300)]
    j = fake_judge(tmp_path, names[:5], set())
    r = plausible_share(names, j, FakeStudent(), 3.0, 50)
    assert r["estimate"] is None and 45 <= len(r["missing"]) <= 50
    assert abs(r["student_only"] - 0.8) < 1e-12


def test_correction_removes_the_students_bias(tmp_path):
    """The student says 80% everywhere; the truth is 30%. The corrected estimate must land near 30%."""
    names = [f"n{i}" for i in range(2000)]
    good = set(np.random.default_rng(0).choice(names, 600, replace=False))
    j = fake_judge(tmp_path, names, good)
    r = plausible_share(names, j, FakeStudent(), 3.0, 200)
    assert r["method"] == "student + judged sample" and r["judged"] == 200
    assert abs(r["estimate"] - 0.30) < 3 * r["se"] and 0.02 < r["se"] < 0.04
    assert abs(r["student_only"] - 0.80) < 1e-12


def test_student_learns_and_round_trips(tmp_path):
    from cvl.student import Student, _fit_one
    rng = np.random.default_rng(0)
    letters = list("abcdefghijklmnopqrstuvwy")
    names = ["".join(rng.choice(letters, rng.integers(4, 9))) for _ in range(300)]
    names += [n[:2] + "zzz" + n[2:] for n in names[:300]]          # "zzz" marks a bad name
    probs = np.array([[1, 0, 0, 0, 0] if "zzz" in n else [0, 0, 0, 0, 1] for n in names], dtype=float)
    order = rng.permutation(len(names))
    tr, va = order[:500], order[500:]
    member, _ = _fit_one([names[i] for i in tr], probs[tr], [names[i] for i in va], probs[va], seed=0, epochs=12)
    s = Student([member], {"note": "test"})
    r = s.rating(["marzzzlow", "marlow"])
    assert r[0] < 2.5 < r[1]
    assert s.uncertainty(["marlow"]).shape == (1,)
    # A name's rating must not depend on what else is in the batch (padding must not leak in).
    alone = s.member_probs(["marlow"])[0, 0]
    in_batch = s.member_probs(["marlow", "averyveryverylongname"])[0, 0]
    assert np.allclose(alone, in_batch, atol=1e-9)
    s.save(tmp_path / "s.pt")
    again = Student.load(tmp_path / "s.pt")
    assert np.allclose(again.predict(["marlow", "qzzzt"]), s.predict(["marlow", "qzzzt"])) and again.meta == {"note": "test"}


def test_reference_lstm_follows_the_shared_interface(tmp_path):
    import torch
    from cvl.reference_lm import SYMBOLS, ReferenceLSTM, encode_batch
    from cvl.sampling import load_checkpoint, sample
    assert SYMBOLS[0] == "$" and len(SYMBOLS) == 29 and SYMBOLS == sorted(SYMBOLS)
    x, y = encode_batch(["ab", "a"])
    assert x.shape == (2, 3) and y[0].tolist() == [SYMBOLS.index("a"), SYMBOLS.index("b"), 0] and y[1, 2] == -100
    torch.manual_seed(0)
    m = ReferenceLSTM(hidden=16)
    m.meta = {"step": 7, "val_loss": 3.2}
    assert abs(m.next_probs("ab").sum() - 1) < 1e-9
    names = sample(m, 30, seed=5)
    assert names == sample(m, 30, seed=5) and all("$" not in n and len(n) <= 20 for n in names)
    assert 3.0 < m.loss(["thorin", "mira"]) < 3.8            # untrained: close to ln(29) = 3.37
    torch.save(m.checkpoint(), tmp_path / "step7.pt")
    again = load_checkpoint(tmp_path / "step7.pt")
    assert again.meta == {"step": 7, "val_loss": 3.2} and sample(again, 30, seed=5) == names
