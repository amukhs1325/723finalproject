"""One sampling function for every model in the project.

    from cvl.sampling import sample
    names = sample("checkpoints/bigram.json", n=1000, temperature=1.0, top_p=1.0, seed=0)

A model works with `sample` if it has:
    symbols: list[str]                    output symbols, including the end marker "$"
    next_probs(context: str) -> ndarray   probabilities over `symbols` given the name so far

Neural models (LSTM, transformer) only need a small wrapper exposing those two
things; temperature and top-p are applied here, identically for every model.
For speed they can instead implement their own batched sampler, as long as it
calls `adjust` below so decoding is the same everywhere.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EOS = "$"


def adjust(probs: np.ndarray, temperature: float = 1.0, top_p: float = 1.0) -> np.ndarray:
    """Apply temperature, then nucleus (top-p) truncation; returns a distribution.

    Works on one distribution (1-D) or a batch (2-D, one row per distribution).
    temperature 0 means greedy (argmax).
    """
    p = np.asarray(probs, dtype=np.float64)
    single = p.ndim == 1
    p = np.atleast_2d(p).copy()
    if temperature <= 0:
        out = np.zeros_like(p)
        out[np.arange(len(p)), p.argmax(1)] = 1.0
        return out[0] if single else out
    if temperature != 1.0:
        logp = np.log(np.clip(p, 1e-300, None)) / temperature
        logp -= logp.max(1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(1, keepdims=True)
    if top_p < 1.0:
        order = np.argsort(-p, axis=1)
        sorted_p = np.take_along_axis(p, order, 1)
        cum = np.cumsum(sorted_p, 1)
        keep_sorted = (cum - sorted_p) < top_p  # keep until the mass reaches top_p
        keep = np.zeros_like(keep_sorted)
        np.put_along_axis(keep, order, keep_sorted, 1)
        p = np.where(keep, p, 0.0)
        p /= p.sum(1, keepdims=True)
    return p[0] if single else p


def load_checkpoint(path: str | Path):
    """Load a saved model. N-gram checkpoints are JSON; neural lanes add theirs here."""
    path = Path(path)
    if path.suffix == ".json":
        from .ngram import CharNGram
        d = json.loads(path.read_text())
        if d.get("type") == "char_ngram":
            return CharNGram.from_dict(d)
    if path.suffix == ".pt":
        import torch
        d = torch.load(path, map_location="cpu", weights_only=True)
        if d.get("type") == "reference_lstm":
            from .reference_lm import ReferenceLSTM
            return ReferenceLSTM.from_checkpoint(d)
        if d.get("type") == "lstm":
            from models.lstm import CharLSTM
            return CharLSTM.from_checkpoint(d)
    raise ValueError(f"don't know how to load {path}; add a loader in cvl/sampling.py")


def sample(model, n: int, temperature: float = 1.0, top_p: float = 1.0,
           seed: int = 0, max_len: int = 20) -> list[str]:
    """Draw n names. `model` is a model object or a checkpoint path."""
    if isinstance(model, (str, Path)):
        model = load_checkpoint(model)
    if hasattr(model, "sample_batch"):     # a model's own batched sampler; it must call adjust()
        return model.sample_batch(n, temperature=temperature, top_p=top_p, seed=seed, max_len=max_len)
    rng = np.random.default_rng(seed)
    symbols = model.symbols
    eos = symbols.index(EOS)
    names = []
    for _ in range(n):
        out = []
        for _ in range(max_len):
            p = adjust(model.next_probs("".join(out)), temperature, top_p)
            k = rng.choice(len(symbols), p=p)
            if k == eos:
                break
            out.append(symbols[k])
        names.append("".join(out))
    return names
