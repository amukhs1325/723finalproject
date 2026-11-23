"""The student: a small local model distilled from the LLM judge.

Why it exists. The judge is one API call per name. Scoring every checkpoint of every
run would be a few hundred thousand calls. The student is trained to reproduce the
judge's rating from the letters of the name alone, so it can score any number of
names for free, offline, on a laptop.

How it is trained (knowledge distillation). The judge does not just say "4": it gives
a probability for each of 1 to 5 (cvl/judge.py). The student is trained to match that
whole distribution, which carries more information per name than the single most
likely rating. Five copies are trained from different random seeds and averaged (an
ensemble); how much the copies disagree is the student's own measure of how unsure
it is, which is used to decide which new names are worth sending to the judge.

    student = Student.load()                 # student.pt at the repo root
    student.rating(["torvus", "xqzvk"])      # expected rating per name, 1.0 to 5.0
    student.predict(["torvus"])              # [[p1..p5]]
    student.uncertainty(["torvus"])          # entropy of the averaged prediction

The judge stays the reference. The student is only trusted as far as it agrees with
the judge on names it never saw (scripts/train_student.py reports that).
"""
from __future__ import annotations

import string
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
STUDENT_PATH = ROOT / "student.pt"
CHARS = ["'", "-"] + list(string.ascii_lowercase)
PAD = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}
LEVELS = np.arange(1, 6, dtype=np.float64)


def encode(names: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(1, max(len(n) for n in names))
    x = torch.full((len(names), width), PAD, dtype=torch.long)
    for i, n in enumerate(names):
        ids = [STOI[c] for c in n if c in STOI]
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x, x != PAD


class NameRater(nn.Module):
    """Characters -> bidirectional GRU -> mean and max over positions -> 5 rating logits."""

    def __init__(self, emb: int = 32, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.config = {"emb": emb, "hidden": hidden, "dropout": dropout}
        self.emb = nn.Embedding(len(CHARS) + 1, emb, padding_idx=PAD)
        self.gru = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(4 * hidden + 1, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 5))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        length = mask.sum(1, keepdim=True).clamp(min=1)
        # Packed, so the GRU never reads padding: a name's rating must not depend on how long
        # the other names in its batch are.
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.emb(x)), length.squeeze(1).cpu(),
                                                   batch_first=True, enforce_sorted=False)
        h, _ = nn.utils.rnn.pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=x.size(1))
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / length
        mx = h.masked_fill(~m, -1e9).max(1).values
        return self.head(torch.cat([mean, mx, length / 15.0], dim=1))


def _fit_one(train_names, train_probs, val_names, val_probs, seed: int, epochs: int = 80,
             patience: int = 10, lr: float = 2e-3, batch: int = 128) -> tuple[NameRater, float]:
    """Train one copy. Loss is cross-entropy against the judge's distribution (soft targets)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NameRater()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    y = torch.tensor(np.asarray(train_probs), dtype=torch.float32)
    xv, mv = encode(val_names)
    yv = torch.tensor(np.asarray(val_probs), dtype=torch.float32)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_names))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x, m = encode([train_names[j] for j in idx])
            loss = -(y[idx] * torch.log_softmax(model(x, m), -1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = -(yv * torch.log_softmax(model(xv, mv), -1)).sum(1).mean().item()
        if val < best - 1e-4:
            best, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model.eval(), best


class Student:
    """An ensemble of NameRater copies, averaged."""

    def __init__(self, members: list[NameRater], meta: dict | None = None):
        self.members = [m.eval() for m in members]
        self.meta = meta or {}
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def fit(cls, train_names, train_probs, val_names, val_probs, k: int = 5, seed: int = 0,
            meta: dict | None = None) -> "Student":
        members = [_fit_one(list(train_names), train_probs, list(val_names), val_probs, seed + i)[0]
                   for i in range(k)]
        return cls(members, meta)

    # ---- prediction -----------------------------------------------------------
    @torch.no_grad()
    def member_probs(self, names: list[str]) -> np.ndarray:
        """[members, names, 5]"""
        out = []
        for start in range(0, len(names), 2048):
            x, m = encode(names[start:start + 2048])
            out.append(np.stack([torch.softmax(mem(x, m).double(), -1).numpy() for mem in self.members]))
        return np.concatenate(out, axis=1)

    def predict(self, names: list[str]) -> np.ndarray:
        """[names, 5]: the ensemble's averaged distribution over ratings 1 to 5."""
        todo = [n for n in dict.fromkeys(names) if n not in self._cache]
        if todo:
            for n, p in zip(todo, self.member_probs(todo).mean(0)):
                self._cache[n] = p
        return np.stack([self._cache[n] for n in names]) if names else np.zeros((0, 5))

    def rating(self, names: list[str]) -> np.ndarray:
        return self.predict(names) @ LEVELS

    def uncertainty(self, names: list[str]) -> np.ndarray:
        """Entropy of the averaged prediction: high when the student is torn or its copies disagree."""
        p = np.clip(self.predict(names), 1e-12, 1.0)
        return -(p * np.log(p)).sum(1)

    # ---- persistence ------------------------------------------------------------
    def save(self, path: Path = STUDENT_PATH) -> None:
        torch.save({"type": "name_rater_ensemble", "config": self.members[0].config, "meta": self.meta,
                    "members": [m.state_dict() for m in self.members]}, path)

    @classmethod
    def load(cls, path: Path = STUDENT_PATH) -> "Student":
        d = torch.load(path, map_location="cpu", weights_only=True)
        members = []
        for state in d["members"]:
            m = NameRater(**d["config"])
            m.load_state_dict(state)
            members.append(m)
        return cls(members, d.get("meta", {}))
