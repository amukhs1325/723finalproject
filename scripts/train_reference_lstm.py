"""Train the reference LSTM fixture (cvl/reference_lm.py) and save log-spaced checkpoints.

    python scripts/train_reference_lstm.py          # about 2 minutes on a laptop CPU

Trains far past the validation-loss minimum on purpose, so the checkpoints cover an
undertrained model, the lowest-loss model and an overfit one. Writes
checkpoints/reference_lstm/step{N}.pt and log.csv. Weights are not committed.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, load_split  # noqa: E402
from cvl.reference_lm import SYMBOLS, ReferenceLSTM, encode_batch  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoints", type=int, default=24)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train, val = load_split("train"), load_split("val")
    out = ROOT / "checkpoints" / "reference_lstm"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("step*.pt"):
        old.unlink()
    save_at = sorted({int(round(x)) for x in np.geomspace(50, args.steps, args.checkpoints)})

    model = ReferenceLSTM(hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(train):,} training names")
    rows, running = [], []
    for step in range(1, args.steps + 1):
        model.train()
        x, y = encode_batch([train[i] for i in rng.integers(0, len(train), args.batch)])
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(SYMBOLS)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(loss.item())
        if step in save_at:
            val_loss, train_loss = model.loss(val), model.loss(train[:2000])
            model.meta = {"step": step, "val_loss": val_loss, "train_loss": train_loss}
            torch.save(model.checkpoint(), out / f"step{step}.pt")
            rows.append({"step": step, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})
            print(f"step {step:>6}  train {train_loss:.3f}  val {val_loss:.3f}", flush=True)
            running = []
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["val_loss"])
    print(f"lowest validation loss {best['val_loss']} at step {best['step']}; "
          f"final step {rows[-1]['step']} has {rows[-1]['val_loss']}")
