"""Fixed reference character n-gram used to judge plausibility.

Interpolated Witten-Bell smoothing, order 4 by default. It is trained once on
the train split and never changes, and it is never one of the models under
study, so plausibility does not measure a model against itself.
"""
from __future__ import annotations

import math
from collections import defaultdict

BOS = "^"
EOS = "$"


class CharNGram:
    def __init__(self, order: int = 4):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        # counts[k][context] -> {char: count}, context is a string of length k
        self.counts: list[dict[str, dict[str, int]]] = [defaultdict(lambda: defaultdict(int))
                                                        for _ in range(order)]
        self.vocab: set[str] = set()
        self._ctx_total: list[dict[str, int]] = []
        self._ctx_types: list[dict[str, int]] = []

    def _events(self, name: str):
        padded = BOS * (self.order - 1) + name + EOS
        for i in range(self.order - 1, len(padded)):
            yield padded[i - self.order + 1:i], padded[i]

    def fit(self, names: list[str]) -> "CharNGram":
        for name in names:
            if BOS in name or EOS in name:
                raise ValueError(f"name contains reserved symbol: {name!r}")
            for ctx, ch in self._events(name):
                self.vocab.add(ch)
                for k in range(self.order):
                    self.counts[k][ctx[len(ctx) - k:] if k else ""][ch] += 1
        self._ctx_total = [{c: sum(d.values()) for c, d in level.items()} for level in self.counts]
        self._ctx_types = [{c: len(d) for c, d in level.items()} for level in self.counts]
        return self

    def prob(self, ctx: str, ch: str) -> float:
        """P(ch | last order-1 chars of ctx), interpolated Witten-Bell."""
        p = 1.0 / (len(self.vocab) + 1)  # uniform floor (+1 leaves room for unseen chars)
        for k in range(self.order):
            h = ctx[len(ctx) - k:] if k else ""
            total = self._ctx_total[k].get(h, 0)
            if total == 0:
                continue
            types = self._ctx_types[k][h]
            c = self.counts[k][h].get(ch, 0)
            p = (c + types * p) / (total + types)
        return p

    def logprob(self, name: str) -> float:
        """Total log2 probability of the name, including the end marker."""
        return sum(math.log2(self.prob(ctx, ch)) for ctx, ch in self._events(name))

    def avg_logprob(self, name: str) -> float:
        """Mean log2 probability per predicted symbol (len(name) + 1 symbols)."""
        return self.logprob(name) / (len(name) + 1)

    # ---- interface shared with neural models (see cvl/sampling.py) -------
    @property
    def symbols(self) -> list[str]:
        """Output symbols in a fixed order; EOS ("$") ends a name."""
        return sorted(self.vocab)

    def next_probs(self, context: str):
        """Probability of each symbol in `symbols` after `context` (the name so far)."""
        import numpy as np
        cache = self.__dict__.setdefault("_next_cache", {})
        key = (BOS * (self.order - 1) + context)[-(self.order - 1):] if self.order > 1 else ""
        if key not in cache:
            p = np.array([self.prob(key, s) for s in self.symbols])
            cache[key] = p / p.sum()
        return cache[key]

    def val_loss(self, names: list[str]) -> float:
        """Cross-entropy in nats per symbol (characters + end marker)."""
        return math.log(self.perplexity(names))

    def to_dict(self) -> dict:
        return {"type": "char_ngram", "order": self.order,
                "counts": [{c: dict(d) for c, d in level.items()} for level in self.counts]}

    @classmethod
    def from_dict(cls, d: dict) -> "CharNGram":
        m = cls(d["order"])
        for k, level in enumerate(d["counts"]):
            for ctx, nxt in level.items():
                for ch, c in nxt.items():
                    m.counts[k][ctx][ch] = c
                    m.vocab.add(ch)
        m._ctx_total = [{c: sum(v.values()) for c, v in level.items()} for level in m.counts]
        m._ctx_types = [{c: len(v) for c, v in level.items()} for level in m.counts]
        return m

    def sample(self, rng, max_len: int = 20) -> str:
        """Draw one name (used for baselines and tests, not by the scorer)."""
        symbols = sorted(self.vocab)
        ctx, out = BOS * (self.order - 1), []
        for _ in range(max_len):
            weights = [self.prob(ctx, s) for s in symbols]
            ch = rng.choices(symbols, weights=weights)[0]
            if ch == EOS:
                break
            out.append(ch)
            ctx = (ctx + ch)[-(self.order - 1):] if self.order > 1 else ""
        return "".join(out)

    def perplexity(self, names: list[str]) -> float:
        total, n = 0.0, 0
        for name in names:
            total += self.logprob(name)
            n += len(name) + 1
        return 2 ** (-total / n)
