"""The project's character transformer, in three sizes matched to the LSTM.

    from models.transformer import CharTransformer
    model = CharTransformer.of_size("medium")

A decoder-only transformer: token and learned position embeddings, pre-norm blocks with
causal self-attention and a 4x GELU feed-forward layer, a final layer norm and a linear
layer over the 29 output symbols. No dropout and no weight decay, like the LSTM, because
the runs are meant to overfit.

    size     layers  width  heads   parameters   LSTM of the same size
    small       2      30     2       24,989          25,321
    medium      4      44     4       99,029          99,967
    large       4      90     6      401,249         400,325

Each is within 1.4% of its LSTM. It uses the same symbols, the same loss (nats per symbol,
end marker counted) and the same decoding (cvl.sampling.adjust) as the LSTM, so
scripts/eval_checkpoint.py scores both the same way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.reference_lm import BOS, EOS, PAD, STOI, SYMBOLS, encode_batch  # noqa: E402
from cvl.sampling import adjust  # noqa: E402

SIZES = {"small": dict(layers=2, width=30, heads=2),
         "medium": dict(layers=4, width=44, heads=4),
         "large": dict(layers=4, width=90, heads=6)}
BLOCK = 24          # positions: the start marker plus up to 23 characters (names are at most 15, samples at most 20)


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.up, self.down = nn.Linear(width, 4 * width), nn.Linear(4 * width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)      # a position only sees earlier ones
        h = h + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return h + self.down(F.gelu(self.up(self.ln2(h))))


class CharTransformer(nn.Module):
    symbols = SYMBOLS

    def __init__(self, layers: int = 4, width: int = 44, heads: int = 4):
        super().__init__()
        self.config = {"layers": layers, "width": width, "heads": heads}
        self.tok = nn.Embedding(len(SYMBOLS) + 2, width, padding_idx=PAD)
        self.pos = nn.Embedding(BLOCK, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(width)
        self.out = nn.Linear(width, len(SYMBOLS))
        self.meta: dict = {}

    @classmethod
    def of_size(cls, size: str) -> "CharTransformer":
        return cls(**SIZES[size])

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, state=None):
        """Returns (logits, None). The second value mirrors the LSTM so one training loop fits both.

        Padding sits to the right of every name and attention is causal, so padded positions
        can never reach a real one; their targets are ignored by the loss.
        """
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            h = block(h)
        return self.out(self.ln(h)), None

    @torch.no_grad()
    def loss(self, names: list[str]) -> float:
        """Cross-entropy in nats per symbol, end marker counted."""
        self.eval()
        total, count = 0.0, 0
        for i in range(0, len(names), 1024):
            x, y = encode_batch(names[i:i + 1024])
            logits, _ = self(x)
            total += F.cross_entropy(logits.reshape(-1, len(SYMBOLS)), y.reshape(-1),
                                     ignore_index=-100, reduction="sum").item()
            count += int((y != -100).sum())
        return total / count

    # ---- interface used by cvl.sampling and scripts/eval_checkpoint.py ----------
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
        out, alive = [[] for _ in range(n)], np.ones(n, dtype=bool)
        for _ in range(min(max_len, BLOCK)):
            logits, _ = self(x)
            p = adjust(torch.softmax(logits[:, -1].double(), -1).numpy(), temperature, top_p)
            nxt = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
            for i in np.flatnonzero(alive):
                if nxt[i] == EOS:
                    alive[i] = False
                else:
                    out[i].append(SYMBOLS[nxt[i]])
            if not alive.any():
                break
            x = torch.cat([x, torch.tensor(nxt).unsqueeze(1)], 1)
        return ["".join(o) for o in out]

    def checkpoint(self) -> dict:
        return {"type": "transformer", "config": self.config, "meta": self.meta,
                "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, d: dict) -> "CharTransformer":
        m = cls(**d["config"])
        m.load_state_dict(d["state_dict"])
        m.meta = dict(d["meta"])
        return m.eval()
