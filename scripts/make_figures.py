"""Draw one figure per run from a checkpoints table.

    python scripts/make_figures.py                                   # results/checkpoints.csv -> results/figures/
    python scripts/make_figures.py results/reference/checkpoints.csv

Each figure is three panels over training step: validation loss, creativity rate, and
where the samples went (novel, respelling, near copy, derived, exact copy). Runs named
{arch}_{size}_s{seed} also get one grid figure per architecture (grid_{arch}.png): a column
per size, every seed as a thin line and their mean as a heavy one. Only rows sampled at
temperature 1 and top-p 1 are drawn. Rerun any time the table changes.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT  # noqa: E402
from cvl.plots import plot_grid, plot_run, split_run  # noqa: E402

if __name__ == "__main__":
    table = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "checkpoints.csv").resolve()
    df = pd.read_csv(table)
    df = df[(df["temperature"] == 1.0) & (df["top_p"] == 1.0) & df["step"].notna()]
    out_dir = table.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for run, g in df.groupby("run"):
        if len(g) < 3:
            continue
        path = plot_run(g, out_dir / f"{run}.png", title=run)
        peak_col = "creativity_rate_judge" if g["creativity_rate_judge"].notna().all() else "creativity_rate"
        print(f"{path}: lowest loss at step {int(g.loc[g['val_loss'].idxmin(), 'step'])}, "
              f"creativity peak at step {int(g.loc[g[peak_col].idxmax(), 'step'])}")
    for arch in sorted({k[0] for k in df["run"].map(split_run) if k}):
        print(plot_grid(df, out_dir / f"grid_{arch}.png", arch=arch))
