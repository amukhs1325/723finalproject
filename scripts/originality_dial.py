"""How far from the training names does a name have to be? Turn that into a dial.

    python scripts/originality_dial.py     # reads results/samples/ and judge_cache.csv; no API calls

The score already uses distance from the nearest training name, but only as a pass/fail bar:
a name is novel if at least 1/6 of its letters differ (scorer_config.json, set so that 90% of
real held-out names clear it). Past that bar a name one letter away from a training name counts
exactly as much as a name unlike any of them. scripts/sensitivity.py tried one higher bar (1/3)
and found the most creative checkpoint moved before the lowest-loss one.

This script shows every setting instead of one, so nobody has to trust the choice of bar:

    bar b    creative = base definition (novel, judge >= 3), and distance >= b,
             for b = 1/6 (the base score itself), 1/5, 1/4, 2/7, 1/3, 2/5, 1/2
    graded   no bar: every creative name counts by its distance instead of counting as 1

Each bar is also described by the share of real held-out names that clear it, which says how
demanding it is in terms that do not depend on any model.

WRITTEN AFTER the LSTM grid and the eight-definition check were seen. On these runs it is a
description, not a test. The test is the same analysis on runs that did not exist when this
was written (fresh LSTM seeds, the transformer grid).

Outputs: results/originality.csv (per checkpoint), results/originality.md,
results/figures/originality_dial_lstm.png.
"""
import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, load_split  # noqa: E402
from cvl.estimate import student_p_plausible  # noqa: E402
from cvl.novelty import NearestName  # noqa: E402
from cvl.plots import INK, MUTED, SERIES, SIZE_ORDER, plt, split_run  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402
from cvl.student import Student  # noqa: E402
from scripts.sensitivity import domain_rate, judged_checkpoints, lstm_table  # noqa: E402
from scripts.summarize_runs import by_size, md_table, summarize  # noqa: E402

RESULTS = ROOT / "results"
CUT = 3.0
BARS = [Fraction(1, 6), Fraction(1, 5), Fraction(1, 4), Fraction(2, 7), Fraction(1, 3), Fraction(2, 5), Fraction(1, 2)]
EPS = 1e-9


def key_of(bar: Fraction) -> str:
    return f"bar_{bar.numerator}_{bar.denominator}"


def gaps(d: pd.DataFrame, key: str) -> pd.DataFrame:
    """Held-out gap per size for one column of d (same rule as results/summary.md)."""
    g = d.rename(columns={key: "creativity_rate_judge", key + "_se": "creativity_rate_se"}).assign(creativity_rate=0.0)
    t = by_size(summarize(g))
    mean = d.groupby(["size", "step"])[key].mean()
    out = []
    for row in t.itertuples():
        best, peak = int(row[5]), int(mean.loc[row.size].idxmax())
        out.append({"size": row.size, "rate at lowest loss": row[6], "lowest-loss step": best,
                    "best step (mean of seeds)": peak,
                    "best step is": "earlier" if peak < best else "later" if peak > best else "same",
                    "held-out gap": row[9], "SE": row[10], "gap / SE": row[9] / row[10] if row[10] else float("nan"),
                    "gap as % of rate": 100 * row[9] / row[6] if row[6] else float("nan")})
    return pd.DataFrame(out)


def score_all() -> pd.DataFrame:
    scorer, student = default_scorer(), Student.load()
    rows = []
    for r, n, t, idx, rating, probs in judged_checkpoints(lstm_table(), scorer, student):
        nov = t["novelty"].to_numpy()
        q, y = student_p_plausible(probs, CUT), (rating >= CUT).astype(float)
        row = {"run": r.run, "step": int(r.step), "val_loss": r.val_loss, "params": r.params, "n": n,
               "temperature": 1.0, "top_p": 1.0, "copy_rate": r.copy_rate, "novel_rate": r.novel_rate}
        for b in BARS:
            row[key_of(b)], row[key_of(b) + "_se"] = domain_rate((nov >= float(b) - EPS).astype(float), q, y, idx, n)
        row["graded"], row["graded_se"] = domain_rate(nov, q, y, idx, n)
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / "originality.csv", index=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="rebuild the tables and figure from results/originality.csv")
    d = pd.read_csv(RESULTS / "originality.csv") if ap.parse_args().reuse else score_all()
    base = lstm_table()["creativity_rate_judge"].to_numpy()
    assert np.allclose(d[key_of(BARS[0])], base, atol=1e-9), "the lowest bar must reproduce the headline"
    d[["arch", "size", "run_seed"]] = pd.DataFrame(d["run"].map(split_run).tolist(), index=d.index)
    nearest = NearestName(load_split("train"))
    val_nov = np.array([nearest.novelty(n) for n in load_split("val")])
    real_share = {b: float((val_nov >= float(b) - EPS).mean()) for b in BARS}

    out = []
    for b in BARS:
        g = gaps(d, key_of(b))
        g.insert(0, "real held-out names that clear it", [f"{100 * real_share[b]:.0f}%"] * len(g))
        g.insert(0, "distance at least", [str(b)] * len(g))
        out.append(g)
    dial = pd.concat(out, ignore_index=True)
    graded = gaps(d, "graded")

    md = ["# Originality as a dial", "",
          "Regenerate with `python scripts/originality_dial.py`. Written after the LSTM results were seen, so on",
          "these runs it describes; it does not test. Judge cut is provisional (3.0) until the scorer freezes.", "",
          "A name counts if it is creative under the base definition and its spelling distance from the nearest",
          "training name is at least the bar. 1/6 is the base score. Gap = rate at the step the other seeds say",
          "is best, minus the rate at this run's own lowest-loss checkpoint. Mean of 3 seeds.", "",
          "## 1. Raising the bar", "", md_table(dial), "",
          "## 2. No bar: each creative name counts by its distance", "",
          "Units are distance-weighted names per sample, so compare the last column with the tables above.", "",
          md_table(graded), ""]
    (RESULTS / "originality.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    figure(dial)


def figure(dial: pd.DataFrame) -> None:
    sizes = [s for s in SIZE_ORDER if s in set(dial["size"])]
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.3 * len(sizes) + 0.8, 5.2), sharex=True, sharey="row",
                             squeeze=False, gridspec_kw={"hspace": 0.14, "wspace": 0.08})
    x = np.arange(len(BARS))
    for j, size in enumerate(sizes):
        g = dial[dial["size"] == size].reset_index(drop=True)
        ax = axes[0, j]
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.errorbar(x, g["held-out gap"], yerr=g["SE"], color=SERIES["blue"], linewidth=2, marker="o", markersize=5, capsize=3,
                    elinewidth=1)
        ax.set_title(f"{size} · lowest loss at step {int(g['lowest-loss step'].iloc[0]):,}", loc="left", fontsize=10,
                     color=INK)
        if j == 0:
            ax.set_ylabel("creative names per sample gained\nby stopping at the best step")
        ax = axes[1, j]
        ax.set_yscale("log")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        best = int(g["lowest-loss step"].iloc[0])
        ax.axhline(best, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.plot(x, g["best step (mean of seeds)"], color=SERIES["blue"], linewidth=2, marker="o", markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n{s}" for b, s in zip(g["distance at least"], g["real held-out names that clear it"])],
                           fontsize=8)
        if j == 0:
            ax.set_ylabel("most creative step (log scale)\ndashed: lowest-loss step")
        if j == len(sizes) // 2:
            ax.set_xlabel("required distance from the nearest training name\n(below: share of real held-out names that clear it)")
    fig.savefig(RESULTS / "figures" / "originality_dial_lstm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
