"""The project's character LSTM, in three sizes.

    from models.lstm import CharLSTM
    model = CharLSTM.of_size("medium")

One LSTM layer over 32-dimensional character embeddings, then a linear layer over the
29 output symbols. Only the hidden width changes between sizes:

    small    hidden  60     25,321 parameters
    medium   hidden 138     99,967
    large    hidden 296    400,325

These are the 25k / 100k / 400k targets in docs/work_split.md, so a transformer can be
matched to them. No dropout and no weight decay: the runs are meant to overfit, because
the question is what happens to the names on both sides of the lowest-loss checkpoint.

The network, the loss and the sampler are the ones in cvl/reference_lm.py, which the
scorer was built against. This class adds the size table and its own checkpoint type.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.reference_lm import ReferenceLSTM  # noqa: E402

SIZES = {"small": 60, "medium": 138, "large": 296}
EMB = 32


class CharLSTM(ReferenceLSTM):
    def __init__(self, hidden: int = SIZES["medium"], emb: int = EMB):
        super().__init__(hidden=hidden, emb=emb)

    @classmethod
    def of_size(cls, size: str) -> "CharLSTM":
        return cls(hidden=SIZES[size])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def checkpoint(self) -> dict:
        return {**super().checkpoint(), "type": "lstm"}
