"""Agreement statistics for the human study. No dependencies beyond the standard library.

krippendorff_alpha  several raters, missing ratings allowed, nominal or ordinal
cohen_kappa         two binary/categorical label lists
"""
from __future__ import annotations

from collections import Counter


def krippendorff_alpha(units: list[list], level: str = "ordinal") -> float:
    """Krippendorff's alpha.

    units: one list per rated item, holding that item's ratings (None = missing).
    level: "nominal" or "ordinal".
    """
    if level not in ("nominal", "ordinal"):
        raise ValueError("level must be 'nominal' or 'ordinal'")
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]          # items with one rating carry no information
    values = sorted({v for u in units for v in u})
    if len(values) < 2:
        return 1.0
    # Coincidence matrix: each ordered pair of ratings within an item, weighted 1/(m-1).
    o = Counter()
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[(a, b)] += 1.0 / (m - 1)
    n_c = {v: sum(o[(v, w)] for w in values) for v in values}
    n = sum(n_c.values())

    def delta(a, b) -> float:
        if a == b:
            return 0.0
        if level == "nominal":
            return 1.0
        lo, hi = sorted((values.index(a), values.index(b)))
        between = sum(n_c[values[g]] for g in range(lo, hi + 1))
        return (between - (n_c[a] + n_c[b]) / 2.0) ** 2

    observed = sum(o[(a, b)] * delta(a, b) for a in values for b in values)
    expected = sum(n_c[a] * n_c[b] * delta(a, b) for a in values for b in values) / (n - 1)
    return 1.0 if expected == 0 else 1.0 - observed / expected


def cohen_kappa(a: list, b: list) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
