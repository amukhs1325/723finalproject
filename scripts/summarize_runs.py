"""Summarize the runs in results/checkpoints.csv: where loss is lowest and where creativity is.

    python scripts/summarize_runs.py                 # writes results/summary.csv and results/summary.md

Per run (temperature 1, top-p 1 rows only):
    best_step         the checkpoint with the lowest validation loss
    creativity_best   creativity rate at that checkpoint, with its standard error
    peak_step         the checkpoint with the highest creativity rate, and the rate there
    held-out gap      see below

Picking the highest of 28 noisy numbers and reporting it overstates the peak (it is the
best of many draws). So the comparison that counts is made out of sample: for each run,
the "creative step" is chosen from the OTHER seeds of the same size (the step where their
mean creativity is highest), and this run's creativity at that step is compared with its
creativity at its own lowest-loss checkpoint. The run being scored never helps choose the
step. The gap's standard error combines the two estimates' errors; the two checkpoints
were sampled independently.

Creativity is the judge-based rate when every row has one, otherwise the scorer's own.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT  # noqa: E402
from cvl.plots import split_run  # noqa: E402


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["temperature"] == 1.0) & (df["top_p"] == 1.0) & df["step"].notna()].copy()
    keys = d["run"].map(split_run)
    d = d[keys.notna()]
    d[["arch", "size", "run_seed"]] = pd.DataFrame(keys[keys.notna()].tolist(), index=d.index)
    judged = d["creativity_rate_judge"].notna().all()
    d["c"] = d["creativity_rate_judge"] if judged else d["creativity_rate"]
    d["c_se"] = d["creativity_rate_se"].fillna(0) if judged else np.sqrt(d["creativity_rate"] * (1 - d["creativity_rate"]) / d["n"])
    rows = []
    for (arch, size), g in d.groupby(["arch", "size"]):
        for seed, run in g.groupby("run_seed"):
            run = run.set_index("step").sort_index()
            best, peak = run["val_loss"].idxmin(), run["c"].idxmax()
            row = {"run": run["run"].iloc[0], "arch": arch, "size": size, "seed": seed,
                   "params": run["params"].iloc[0] if "params" in run else None,
                   "best_step": int(best), "best_val_loss": run.loc[best, "val_loss"],
                   "creativity_best": run.loc[best, "c"], "creativity_best_se": run.loc[best, "c_se"],
                   "copy_rate_best": run.loc[best, "copy_rate"],
                   "peak_step": int(peak), "creativity_peak": run.loc[peak, "c"],
                   "last_step": int(run.index.max()), "last_val_loss": run["val_loss"].iloc[-1],
                   "creativity_last": run["c"].iloc[-1], "copy_rate_last": run["copy_rate"].iloc[-1]}
            others = g[g["run_seed"] != seed]
            if len(others):
                step = others.groupby("step")["c"].mean().idxmax()
                if step in run.index:
                    row.update({"heldout_step": int(step), "creativity_heldout": run.loc[step, "c"],
                                "heldout_gap": run.loc[step, "c"] - run.loc[best, "c"],
                                "heldout_gap_se": float(np.hypot(run.loc[step, "c_se"], run.loc[best, "c_se"]))
                                if step != best else 0.0})
            rows.append(row)
    return pd.DataFrame(rows)


def md_table(t: pd.DataFrame) -> str:
    def cell(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(t.columns) + " |", "| " + " | ".join("---" for _ in t.columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)


def by_size(s: pd.DataFrame) -> pd.DataFrame:
    order = {"small": 0, "medium": 1, "large": 2}
    out = []
    for (arch, size), g in s.groupby(["arch", "size"]):
        k = len(g)
        gap = g["heldout_gap"].dropna() if "heldout_gap" in g else pd.Series(dtype=float)
        out.append({"arch": arch, "size": size, "seeds": k,
                    "lowest val loss": g["best_val_loss"].mean(),
                    "at step (median)": int(g["best_step"].median()),
                    "creativity at lowest loss": g["creativity_best"].mean(),
                    "creative step (median)": int(g["heldout_step"].median()) if len(gap) else None,
                    "creativity there": g["creativity_heldout"].mean() if len(gap) else None,
                    "held-out gap": gap.mean() if len(gap) else None,
                    # within-run sampling error of the mean gap; seed-to-seed spread is the column after
                    "gap SE": float(np.sqrt((g["heldout_gap_se"] ** 2).sum()) / len(gap)) if len(gap) else None,
                    "gap by seed": " ".join(f"{x:+.3f}" for x in gap) if len(gap) else "",
                    "copies at lowest loss": g["copy_rate_best"].mean(),
                    "copies at last step": g["copy_rate_last"].mean()})
    t = pd.DataFrame(out)
    return t.sort_values(["arch", "size"], key=lambda c: c.map(order) if c.name == "size" else c).reset_index(drop=True)


if __name__ == "__main__":
    table = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "checkpoints.csv").resolve()
    s = summarize(pd.read_csv(table))
    if s.empty:
        sys.exit("no runs named {arch}_{size}_s{seed} in the table")
    s.to_csv(table.parent / "summary.csv", index=False)
    t = by_size(s)
    md = ["# Run summary", "",
          f"From `{table.name}`. Regenerate with `python scripts/summarize_runs.py`.", "",
          "Held-out gap: creativity at the step the other seeds say is most creative, minus creativity",
          "at this run's own lowest-loss checkpoint. Positive means the lowest-loss checkpoint was not",
          "the most creative one.", "", md_table(t), "",
          "## Per run", "", md_table(s), ""]
    (table.parent / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(t.round(3).to_string(index=False))
    print(f"\nwrote {table.parent / 'summary.csv'} and summary.md")
