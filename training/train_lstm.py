"""Train the LSTM grid and save log-spaced checkpoints.

    python training/train_lstm.py --size medium --seed 0        # one run; minutes for small and medium, far longer for large
    python training/train_lstm.py --size small medium large --seed 0 1 2    # the full grid

Each run writes
    checkpoints/lstm_{size}_s{seed}/step{N}.pt     weights, config, step, val_loss
    checkpoints/lstm_{size}_s{seed}/log.csv        step, train_loss, val_loss
and scripts/eval_checkpoint.py scores the run directory as is. Weights are not committed.

Every run uses the same recipe: Adam at 2e-3, batches of 128 names, gradient clipping at 1,
no dropout, no weight decay, and the same 28 checkpoint steps from 50 to 40,000. That is
about 410 passes over the 12,386 training names, far past the lowest validation loss for
every size, so each run covers an undertrained, a lowest-loss and an overfit model.
The seed sets both the initial weights and the order of the training batches.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, load_split  # noqa: E402
from cvl.reference_lm import SYMBOLS, encode_batch  # noqa: E402
from models.lstm import SIZES, CharLSTM  # noqa: E402

STEPS, CHECKPOINTS, BATCH, LR = 40000, 28, 128, 2e-3


def save_steps(steps: int = STEPS, k: int = CHECKPOINTS) -> list[int]:
    return sorted({int(round(x)) for x in np.geomspace(50, steps, k)})


def make_model(arch: str, size: str):
    if arch == "lstm":
        return CharLSTM.of_size(size)
    if arch == "transformer":
        from models.transformer import CharTransformer
        return CharTransformer.of_size(size)
    raise ValueError(arch)


def train_run(size: str, seed: int, steps: int = STEPS, out_root: Path | None = None,
              train: list[str] | None = None, val: list[str] | None = None, quiet: bool = False,
              arch: str = "lstm", lr: float = LR) -> Path:
    """One run. The loop is shared by both architectures (training/train_transformer.py calls it)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = load_split("train") if train is None else train
    val = load_split("val") if val is None else val
    out = (out_root or ROOT / "checkpoints") / f"{arch}_{size}_s{seed}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("step*.pt"):
        old.unlink()
    save_at = set(save_steps(steps))

    model = make_model(arch, size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if not quiet:
        print(f"{arch}_{size}_s{seed}: {model.n_params():,} parameters", flush=True)
    rows, t0 = [], time.time()
    for step in range(1, steps + 1):
        model.train()
        x, y = encode_batch([train[i] for i in rng.integers(0, len(train), BATCH)])
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, len(SYMBOLS)), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step in save_at:
            val_loss, train_loss = model.loss(val), model.loss(train[:2000])
            model.meta = {"step": step, "val_loss": val_loss, "train_loss": train_loss,
                          "size": size, "seed": seed, "params": model.n_params()}
            torch.save(model.checkpoint(), out / f"step{step}.pt")
            rows.append({"step": step, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4)})
            if not quiet:
                print(f"  step {step:>6}  train {train_loss:.3f}  val {val_loss:.3f}", flush=True)
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: r["val_loss"])
    if not quiet:
        print(f"  lowest validation loss {best['val_loss']} at step {best['step']}, "
              f"last step has {rows[-1]['val_loss']}  ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", nargs="+", default=["medium"], choices=list(SIZES))
    ap.add_argument("--seed", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=STEPS)
    args = ap.parse_args()
    torch.set_num_threads(2)
    for size in args.size:
        for seed in args.seed:
            train_run(size, seed, args.steps)
