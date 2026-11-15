"""Nearest-training-name similarity, used for novelty.

similarity(a, b) = 1 - levenshtein(a, b) / max(len(a), len(b))   (0 to 1)
novelty(name)    = 1 - max similarity to any training name

Uses RapidFuzz when installed (fast); otherwise a NumPy implementation that gives
identical numbers, about 10 seconds per 1,000 names.
"""
from __future__ import annotations

import numpy as np

try:  # optional speed-up
    from rapidfuzz import process
    from rapidfuzz.distance import Levenshtein as _RFLev
    HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - depends on environment
    HAVE_RAPIDFUZZ = False


def levenshtein(a: str, b: str) -> int:
    """Plain edit distance, used in tests and as a reference."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


class NearestName:
    """Finds the most similar training name for each query."""

    def __init__(self, train_names: list[str], use_rapidfuzz: bool | None = None):
        self.train = list(dict.fromkeys(train_names))
        self.train_set = set(self.train)
        self.use_rf = HAVE_RAPIDFUZZ if use_rapidfuzz is None else (use_rapidfuzz and HAVE_RAPIDFUZZ)
        if not self.use_rf:
            self._build_numpy()

    def _build_numpy(self):
        self.lens = np.array([len(x) for x in self.train], dtype=np.int32)
        width = int(self.lens.max())
        codes = np.full((len(self.train), width), -1, dtype=np.int32)
        for i, name in enumerate(self.train):
            codes[i, :len(name)] = [ord(c) for c in name]
        self.codes = codes
        self.rows = np.arange(len(self.train))

    def _distances_numpy(self, query: str) -> np.ndarray:
        n, width = self.codes.shape
        prev = np.broadcast_to(np.arange(width + 1, dtype=np.int32), (n, width + 1)).copy()
        cur = np.empty_like(prev)
        for i, ch in enumerate(query, 1):
            cur[:, 0] = i
            mismatch = (self.codes != ord(ch)).astype(np.int32)
            sub = prev[:, :-1] + mismatch  # diagonal moves, all j at once
            dele = prev[:, 1:] + 1         # moves from the row above
            best = np.minimum(sub, dele)
            for j in range(1, width + 1):  # insertion depends on the cell to the left
                cur[:, j] = np.minimum(best[:, j - 1], cur[:, j - 1] + 1)
            prev, cur = cur, prev
        return prev[self.rows, self.lens]

    def nearest(self, query: str) -> tuple[float, str]:
        """(highest similarity, the training name it came from)."""
        if query in self.train_set:
            return 1.0, query
        if not query:
            return 0.0, ""
        if self.use_rf:
            match, sim, _ = process.extractOne(query, self.train,
                                               scorer=_RFLev.normalized_similarity)
            return float(sim), match
        dist = self._distances_numpy(query)
        denom = np.maximum(self.lens, len(query))
        sims = 1.0 - dist / denom
        k = int(np.argmax(sims))
        return float(sims[k]), self.train[k]

    def novelty(self, query: str) -> float:
        return 1.0 - self.nearest(query)[0]
