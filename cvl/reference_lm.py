"""A small character LSTM used as a fixture for building and testing the scorer.

This is NOT one of the project's reported models (those are models/lstm.py and
models/transformer.py, owned by Agastya and Vraj). It exists so the evaluation side
has realistic checkpoints to work with before the real runs land: names from an
early, a lowest-loss and an overfit checkpoint, for checking that the judge's local
student still works on real model output ("on-policy" names).

It follows the proposed shared interface: 29 output symbols with "$" as the end
marker, validation loss in nats per symbol with the end marker counted, and a
checkpoint that carries its own step and val_loss.
"""
from __future__ import annotations

import string

import numpy as np
import torch
import torch.nn as nn

from .sampling import adjust

SYMBOLS = ["$", "'", "-"] + list(string.ascii_lowercase)   # same order as sorted(CharNGram.vocab)
EOS, BOS, PAD = 0, len(SYMBOLS), len(SYMBOLS) + 1
STOI = {c: i for i, c in enumerate(SYMBOLS)}


def encode_batch(names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs (BOS + name) and targets (name + EOS), padded. Target padding is -100."""
    width = max(len(n) for n in names) + 1
    x = torch.full((len(names), width), PAD, dtype=torch.long)
    y = torch.full((len(names), width), -100, dtype=torch.long)
    for i, n in enumerate(names):
        ids = [STOI[c] for c in n]
        x[i, :len(ids) + 1] = torch.tensor([BOS] + ids)
        y[i, :len(ids) + 1] = torch.tensor(ids + [EOS])
    return x, y


class ReferenceLSTM(nn.Module):
    def __init__(self, hidden: int = 256, emb: int = 32):
        super().__init__()
        self.config = {"hidden": hidden, "emb": emb}
        self.emb = nn.Embedding(len(SYMBOLS) + 2, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True)
        self.out = nn.Linear(hidden, len(SYMBOLS))
        self.meta: dict = {}

    def forward(self, x: torch.Tensor, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.out(h), state

    @torch.no_grad()
    def loss(self, names: list[str]) -> float:
        """Cross-entropy in nats per symbol, end marker counted."""
        self.eval()
        total, count = 0.0, 0
        for i in range(0, len(names), 1024):
            x, y = encode_batch(names[i:i + 1024])
            logits, _ = self(x)
            total += nn.functional.cross_entropy(logits.reshape(-1, len(SYMBOLS)), y.reshape(-1),
                                                 ignore_index=-100, reduction="sum").item()
            count += int((y != -100).sum())
        return total / count

    # ---- interface used by cvl.sampling and scripts/eval_checkpoint.py ----------
    symbols = SYMBOLS

    @torch.no_grad()
    def next_probs(self, context: str) -> np.ndarray:
        self.eval()
        x = torch.tensor([[BOS] + [STOI[c] for c in context]])
        logits, _ = self(x)
        return torch.softmax(logits[0, -1].double(), -1).numpy()

    @torch.no_grad()
    def sample_batch(self, n: int, temperature: float = 1.0, top_p: float = 1.0,
                     seed: int = 0, max_len: int = 20) -> list[str]:
        """Same decoding as cvl.sampling.sample (it calls adjust), all n names at once."""
        self.eval()
        rng = np.random.default_rng(seed)
        x = torch.full((n, 1), BOS, dtype=torch.long)
        state, out, alive = None, [[] for _ in range(n)], np.ones(n, dtype=bool)
        for _ in range(max_len):
            logits, state = self(x, state)
            p = adjust(torch.softmax(logits[:, -1].double(), -1).numpy(), temperature, top_p)
            nxt = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
            for i in np.flatnonzero(alive):
                if nxt[i] == EOS:
                    alive[i] = False
                else:
                    out[i].append(SYMBOLS[nxt[i]])
            if not alive.any():
                break
            x = torch.tensor(nxt).unsqueeze(1)
        return ["".join(o) for o in out]

    def checkpoint(self) -> dict:
        return {"type": "reference_lstm", "config": self.config, "meta": self.meta,
                "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, d: dict) -> "ReferenceLSTM":
        m = cls(**d["config"])
        m.load_state_dict(d["state_dict"])
        m.meta = dict(d["meta"])
        return m.eval()
