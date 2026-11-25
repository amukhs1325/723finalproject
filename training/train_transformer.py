"""Train the transformer grid and save log-spaced checkpoints.

    python training/train_transformer.py --size medium --seed 0
    python training/train_transformer.py --size small medium large --seed 0 1 2    # the full grid

Same loop as the LSTM (training/train_lstm.py: train_run): batches of 128 names in the same
order for a given seed, Adam, gradient clipping at 1, no dropout, no weight decay, and the
same 28 checkpoint steps from 50 to 40,000. Each run writes
    checkpoints/transformer_{size}_s{seed}/step{N}.pt and log.csv
which scripts/eval_checkpoint.py scores as is. Weights are not committed.

The learning rate is the one thing chosen separately for this architecture: 1e-3, against
2e-3 for the LSTM. It was picked from 4,000-step pilot runs (seed 0, each size, at 2e-3,
1e-3 and 5e-4) by lowest validation loss alone, before any transformer sample was scored:

    lowest validation loss     2e-3     1e-3     5e-4
    small                      2.396    2.396    2.396
    medium                     2.405    2.392    2.394
    large                      2.400    2.396    2.393

1e-3 and 5e-4 tie; 1e-3 is closer to the LSTM's and overfits inside the same step budget.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.transformer import SIZES  # noqa: E402
from training.train_lstm import STEPS, train_run  # noqa: E402

LR = 1e-3

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", nargs="+", default=["medium"], choices=list(SIZES))
    ap.add_argument("--seed", nargs="+", type=int, default=[0])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--out-root", type=Path, default=None, help="write runs somewhere other than checkpoints/")
    args = ap.parse_args()
    torch.set_num_threads(1)
    for size in args.size:
        for seed in args.seed:
            train_run(size, seed, args.steps, out_root=args.out_root, arch="transformer", lr=args.lr)
