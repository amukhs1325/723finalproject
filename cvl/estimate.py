"""Estimate how many of a checkpoint's novel names the judge would call plausible,
without judging all of them.

The problem. The judge is one API call per name. The student (cvl/student.py) is free
but wrong in a systematic way: on the reference LSTM it rated 5 to 13 points too many
names plausible, and the size of the error changed with the training step. Using it
alone would bend the very curve this project measures.

The fix (a difference estimator, the idea behind "prediction-powered inference",
Angelopoulos et al. 2023). Score every name with the student. Send a small random
sample to the judge. On that sample we see exactly how far off the student is, and
subtract that error from the student's average over all names:

    estimate = mean over ALL names of student's P(plausible)
             + mean over the SAMPLE of (judge says plausible - student's P(plausible))

Because the sample is random, the correction is unbiased no matter how wrong the
student is. A good student only makes the estimate less noisy: here the same accuracy
needs about half as many judge calls as judging a plain random sample.

The sample is drawn with a seed derived from the names themselves, so the same
checkpoint always gets the same sample and the result can be recomputed from
judge_cache.csv alone.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

PROVISIONAL_CUT = 3.0     # used until the human ratings set judge_cut in scorer_config.json


def student_p_plausible(probs: np.ndarray, cut: float) -> np.ndarray:
    """Student's probability that the judge's rating is at or above the cut."""
    first = max(0, min(4, math.ceil(cut) - 1))       # index of the lowest rating level >= cut
    return probs[:, first:].sum(1)


def pick_sample(names: list[str], m: int) -> list[str]:
    """A reproducible uniform random sample of m names."""
    ordered = sorted(names)
    seed = int(hashlib.sha256("\n".join(ordered).encode()).hexdigest()[:8], 16)
    idx = np.random.default_rng(seed).choice(len(ordered), size=min(m, len(ordered)), replace=False)
    return [ordered[i] for i in sorted(idx)]


def plausible_share(names: list[str], judge, student, cut: float, m: int | None) -> dict:
    """Share of `names` (unique, novel, gate-passing) that the judge would rate >= cut.

    m = how many names to judge. None or m >= len(names) means judge all of them (exact).
    Returns estimate, standard error, the names that still need judging, and the uncorrected
    student-only number for comparison. `estimate` is None while judge ratings are missing.
    """
    names = list(dict.fromkeys(names))
    n = len(names)
    if n == 0:
        return {"estimate": 0.0, "se": 0.0, "n": 0, "judged": 0, "missing": [], "student_only": float("nan"),
                "method": "empty"}
    sample = names if (m is None or m >= n) else pick_sample(names, m)
    missing = judge.missing(sample) if judge is not None else list(sample)
    q = student_p_plausible(student.predict(names), cut) if student is not None else None
    out = {"n": n, "judged": len(sample) - len(missing), "missing": missing,
           "student_only": float(q.mean()) if q is not None else float("nan")}
    if missing:
        return {**out, "estimate": None, "se": None, "method": "needs judge ratings"}
    y = np.array([judge.rating(x) >= cut for x in sample], dtype=float)
    if len(sample) == n:
        return {**out, "estimate": float(y.mean()), "se": 0.0, "method": "judge, all names"}
    if q is None:                                    # no student: plain random-sample mean
        se = math.sqrt((1 - len(sample) / n) * y.var(ddof=1) / len(sample))
        return {**out, "estimate": float(y.mean()), "se": se, "method": "judge, random sample"}
    pos = {x: i for i, x in enumerate(names)}
    diff = y - q[[pos[x] for x in sample]]
    est = float(q.mean() + diff.mean())
    se = math.sqrt((1 - len(sample) / n) * diff.var(ddof=1) / len(sample))
    return {**out, "estimate": min(1.0, max(0.0, est)), "se": se, "method": "student + judged sample"}
