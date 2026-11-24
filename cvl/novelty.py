"""Novelty views: how close is a generated name to anything in the training set?

Four deterministic nearest-neighbor queries (see docs/evaluation_plan.md):
    copy      exact match                                   (scorer)
    spelling  NearestName: normalized Levenshtein similarity
    sound     SoundIndex: same respelling key as a training name
    meaning   MeaningIndex: cosine in a fixed sentence-embedding space
plus SpliceIndex, a descriptor (training prefix + training suffix), not a filter.

Spelling view:

similarity(a, b) = 1 - levenshtein(a, b) / max(len(a), len(b))   (0 to 1)
novelty(name)    = 1 - max similarity to any training name

Uses RapidFuzz when installed (fast); otherwise a NumPy implementation that gives
identical numbers, about 10 seconds per 1,000 names.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

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


# ---- sound view ---------------------------------------------------------
# Respelling key: fold spellings that sound the same. Applied in order.
# Double Metaphone was tested and rejected: it drops vowels and matches 76% of
# real held-out names to some training name.
SOUND_RULES = [
    (r"['\-]", ""),
    (r"ph", "f"), (r"ck", "k"), (r"kh", "k"), (r"chr", "kr"),
    (r"qu", "kw"), (r"q", "k"), (r"x", "ks"),
    (r"c(?=[eiy])", "s"), (r"c", "k"), (r"y", "i"),
    (r"([^aeiou])\1+", r"\1"),      # doubled consonants anywhere: jonn -> jon
    (r"([aeiou])\1+$", r"\1"),      # doubled vowel at the end:    davee -> dave
    (r"(?<=[^aeiou])e$", ""),        # final silent e:              dave -> dav
    (r"(?<=[aeiou])h$", ""),         # final h after a vowel:       sarah -> sara
]
_SOUND_RULES = [(re.compile(p), r) for p, r in SOUND_RULES]


def sound_key(name: str) -> str:
    for pat, rep in _SOUND_RULES:
        name = pat.sub(rep, name)
    return name


class SoundIndex:
    """Training names by respelling key."""

    def __init__(self, train_names: list[str]):
        self.by_key: dict[str, str] = {}
        for name in train_names:
            self.by_key.setdefault(sound_key(name), name)

    def match(self, query: str) -> str | None:
        """The training name that is a respelling of `query`, if any."""
        key = sound_key(query)
        return self.by_key.get(key) if key else None


# ---- splice descriptor ----------------------------------------------------
class SpliceIndex:
    """Is a name a training prefix joined to a training suffix (each >= k letters)?

    44% of real held-out names are, so this describes a name; it does not make it a copy.
    """

    def __init__(self, train_names: list[str], k: int = 3):
        self.k = k
        self.prefixes: set[str] = set()
        self.suffixes: set[str] = set()
        for t in train_names:
            for i in range(k, len(t) + 1):
                self.prefixes.add(t[:i])
                self.suffixes.add(t[-i:])

    def is_splice(self, name: str) -> bool:
        k = self.k
        return any(name[:i] in self.prefixes and name[i:] in self.suffixes
                   for i in range(k, len(name) - k + 1))


# ---- meaning view -----------------------------------------------------------
MEANING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MEANING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class MeaningIndex:
    """Cosine similarity to the nearest training name in a fixed embedding space.

    Used only at the high end (see the threshold in scorer_config.json), where it
    finds names built on the same root: eldrath -> eldra, jonasz -> jonas. For
    nonsense strings the cosine carries no information, so it is never used as a
    general distance. Runs on CPU with a pinned model revision.
    """

    def __init__(self, train_names: list[str], cache_dir: Path | None = None,
                 model: str = MEANING_MODEL, revision: str = MEANING_REVISION):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError("the meaning view needs sentence-transformers: "
                              "pip install sentence-transformers") from e
        self.train = list(dict.fromkeys(train_names))
        self.model_name, self.revision = model, revision
        self._model = SentenceTransformer(model, revision=revision, device="cpu")
        self._cache: dict[str, tuple[float, str]] = {}
        tag = hashlib.sha256(("\n".join(self.train) + model + revision).encode()).hexdigest()[:12]
        path = Path(cache_dir) / f"train_emb_{tag}.npy" if cache_dir else None
        if path is not None and path.exists():
            self.train_emb = np.load(path)
        else:
            self.train_emb = self._encode(self.train)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.train_emb)

    def _encode(self, names: list[str]) -> np.ndarray:
        return self._model.encode(names, batch_size=256, normalize_embeddings=True,
                                  show_progress_bar=False).astype(np.float32)

    def nearest_many(self, names: list[str]) -> list[tuple[float, str]]:
        """(cosine to the nearest training name, that name) for each query."""
        todo = [n for n in dict.fromkeys(names) if n and n not in self._cache]
        for start in range(0, len(todo), 2048):
            chunk = todo[start:start + 2048]
            sims = self._encode(chunk) @ self.train_emb.T
            best = sims.argmax(1)
            for i, n in enumerate(chunk):
                self._cache[n] = (float(sims[i, best[i]]), self.train[int(best[i])])
        return [self._cache.get(n, (0.0, "")) for n in names]

    def nearest(self, name: str) -> tuple[float, str]:
        return self.nearest_many([name])[0]
