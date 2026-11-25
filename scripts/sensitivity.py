"""Does the finding depend on how creativity is defined? And is the score circular with loss?

    python scripts/sensitivity.py                # the LSTM grid; reads results/samples/ and judge_cache.csv, no API calls
    python scripts/sensitivity.py transformer    # the same definitions on the transformer grid

Written after the LSTM grid showed creativity flat around the loss minimum. Two worries:
the score might just be validation loss in disguise, or the definition might be hiding
something a different reasonable definition would show. This script checks both.

The alternative definitions are fixed here, before looking at what they give, and ALL of
them are reported. Picking the one that produces the nicest story afterwards would be
its own kind of circularity.

    base         novel, and judge rating >= 3.0 (the provisional headline)
    strict_3.5   novel, and judge rating >= 3.5
    strict_4     novel, and judge rating >= 4.0 (a name the judge calls good, not passable)
    far          base, and at least a third of the letters differ from the nearest training
                 name (the base novelty rule asks for a sixth)
    no_splice    base, and not the front of one training name joined to the back of another
    atypical     base, and less typical under the 3-gram reference than the median real
                 held-out name: plausible to the judge without being the obvious string
    ngram        novel, and 3-gram percentile >= 0.10 (no judge at all)
    heldout      the sample is exactly a real name from the validation or test split

Every judge-based rate uses the same 200 judged names per checkpoint as the headline and
the same correction (cvl/estimate.py), restricted to the names the definition keeps, so
each is unbiased and has a standard error.

Outputs: results/sensitivity.csv (per checkpoint), results/sensitivity.md,
results/figures/sensitivity_lstm.png, results/figures/loss_vs_creativity_lstm.png.
For another architecture the files are sensitivity_{arch}.csv / .md and the figures end in
_{arch}.png. The definitions are the same objects either way; nothing is retuned per architecture.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT  # noqa: E402
from cvl.estimate import pick_sample, student_p_plausible  # noqa: E402
from cvl.plots import INK, MUTED, SERIES, SIZE_ORDER, plt, split_run  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402
from cvl.student import Student  # noqa: E402
from scripts.summarize_runs import by_size, md_table, summarize  # noqa: E402

RESULTS = ROOT / "results"
M = 200
# name -> (judge cut, which novel names the definition keeps)
JUDGED = {
    "base": (3.0, lambda t: np.ones(len(t), bool)),
    "strict_3.5": (3.5, lambda t: np.ones(len(t), bool)),
    "strict_4": (4.0, lambda t: np.ones(len(t), bool)),
    "far": (3.0, lambda t: t["novelty"].to_numpy() >= 1 / 3 - 1e-9),
    "no_splice": (3.0, lambda t: ~t["splice"].to_numpy()),
    "atypical": (3.0, lambda t: t["plausibility"].to_numpy() < 0.5),
}
FROM_TABLE = {"ngram": "creativity_rate_ngram", "heldout": "heldout_hit_rate"}
LABELS = {"base": "base (judge ≥ 3)", "strict_3.5": "judge ≥ 3.5", "strict_4": "judge ≥ 4", "far": "far from training",
          "no_splice": "not a splice", "atypical": "plausible but atypical", "ngram": "3-gram plausibility",
          "heldout": "real held-out names"}


def domain_rate(keep: np.ndarray, q: np.ndarray, y_s: np.ndarray, idx_s: np.ndarray, n_samples: int) -> tuple[float, float]:
    """Creative names per sample for one definition: student on every kept name, corrected
    by the judged sample. keep, q: over all novel names. y_s: judge says plausible, on the sample.
    keep may be 0/1 or a weight per name (scripts/originality_dial.py counts a name by its distance)."""
    n, m = len(keep), len(idx_s)
    z = q * keep
    if m == n:
        return float((y_s * keep).sum() / n_samples), 0.0
    diff = y_s * keep[idx_s] - z[idx_s]
    count = z.sum() + n * diff.mean()
    se = n * math.sqrt((1 - m / n) * diff.var(ddof=1) / m)
    return float(max(0.0, count) / n_samples), float(se / n_samples)


def runs_table(arch: str) -> pd.DataFrame:
    table = pd.read_csv(RESULTS / "checkpoints.csv")
    keys = table["run"].map(split_run)
    return table[(table["temperature"] == 1.0) & (table["top_p"] == 1.0) & keys.map(lambda k: bool(k) and k[0] == arch)]


def lstm_table() -> pd.DataFrame:
    return runs_table("lstm")


def judged_checkpoints(table: pd.DataFrame, scorer, student):
    """For each checkpoint: its row, the number of samples, the novel names (one row each), where the
    200 judged names sit in that table, their judge ratings, and the student's probabilities for all."""
    for r in table.itertuples():
        path = RESULTS / "samples" / r.run / f"step{int(r.step)}_t1_p1_s0.txt"
        names = path.read_text(encoding="utf-8").split("\n")[:-1]
        t = scorer.table(names)
        t = t[t["novel"] & t["gate_ok"] & t["first_occurrence"]].reset_index(drop=True)
        pos = {x: i for i, x in enumerate(t["name"])}
        sample = t["name"].tolist() if len(t) <= M else pick_sample(t["name"].tolist(), M)
        idx = np.array([pos[x] for x in sample])
        rating = np.array([scorer.judge.rating(x) for x in sample], dtype=float)
        yield r, len(names), t, idx, rating, student.predict(t["name"].tolist())
        print(f"{r.run} step {int(r.step)}", flush=True)


def main(arch: str = "lstm") -> None:
    scorer, student = default_scorer(), Student.load()
    table = runs_table(arch)
    if table.empty:
        sys.exit(f"no {arch} runs in results/checkpoints.csv")
    tag = "" if arch == "lstm" else f"_{arch}"
    rows, per_name = [], []
    for r, n, t, idx, rating, probs in judged_checkpoints(table, scorer, student):
        row = {"run": r.run, "step": int(r.step), "val_loss": r.val_loss, "params": r.params, "n": n,
               "temperature": 1.0, "top_p": 1.0, "copy_rate": r.copy_rate, "novel_rate": r.novel_rate}
        for key, (cut, rule) in JUDGED.items():
            row[key], row[key + "_se"] = domain_rate(rule(t), student_p_plausible(probs, cut), (rating >= cut).astype(float),
                                                     idx, n)
        for key, col in FROM_TABLE.items():
            p = getattr(r, col)
            row[key], row[key + "_se"] = p, math.sqrt(p * (1 - p) / n)
        rows.append(row)
        s = t.iloc[idx]
        per_name.append(pd.DataFrame({"name": s["name"].to_numpy(), "judge": rating, "ngram_pct": s["plausibility"].to_numpy(),
                                      "novelty": s["novelty"].to_numpy(), "length": s["name"].str.len().to_numpy(),
                                      "splice": s["splice"].to_numpy()}))
    d = pd.DataFrame(rows)
    d.to_csv(RESULTS / f"sensitivity{tag}.csv", index=False)
    assert np.allclose(d["base"], table["creativity_rate_judge"].to_numpy(), atol=1e-9), "base must reproduce the headline"

    keys = list(JUDGED) + list(FROM_TABLE)
    d[["arch", "size", "run_seed"]] = pd.DataFrame(d["run"].map(split_run).tolist(), index=d.index)
    md = ["# Does the result depend on the definition of creativity?" + ("" if arch == "lstm" else f" ({arch})"), "",
          f"Regenerate with `python scripts/sensitivity.py{'' if arch == 'lstm' else ' ' + arch}`. The definitions were fixed before the numbers were",
          "computed and all of them are shown. Judge cut is provisional (3.0) until the scorer freezes.", "",
          "## 1. Eight definitions, same checkpoints", "",
          "Gap = rate at the step the other seeds say is best under that definition, minus the rate at this",
          "run's own lowest-loss checkpoint. Mean of 3 seeds, standard error from sampling.", ""]
    out = []
    for key in keys:
        g = d.rename(columns={key: "creativity_rate_judge", key + "_se": "creativity_rate_se"}).assign(creativity_rate=0.0)
        t = by_size(summarize(g))
        mean = d.groupby(["size", "step"])[key].mean()
        for row in t.itertuples():
            peak = mean.loc[row.size].idxmax()
            best = int(row[5])
            gap, se = row[9], row[10]
            out.append({"definition": LABELS[key], "size": row.size, "rate at lowest loss": row[6],
                        "lowest-loss step": best, "best step (mean of seeds)": int(peak),
                        "best step is": "earlier" if peak < best else "later" if peak > best else "same",
                        "held-out gap": gap, "SE": se, "gap / SE": gap / se if se else float("nan"),
                        "gap as % of rate": 100 * gap / row[6] if row[6] else float("nan")})
    sens = pd.DataFrame(out)
    md += [md_table(sens), ""]

    # 2. Per name: is the judge the 3-gram in disguise?
    pn = pd.concat(per_name).drop_duplicates("name")
    rho_ng = spearmanr(pn["judge"], pn["ngram_pct"]).statistic
    rho_len = spearmanr(pn["judge"], pn["length"]).statistic
    rho_nov = spearmanr(pn["judge"], pn["novelty"]).statistic
    j, g3 = pn["judge"] >= 3.0, pn["ngram_pct"] >= 0.10
    md += ["## 2. Is the judge just the 3-gram model again?", "",
           f"{len(pn):,} distinct novel names from the grid that the judge rated.", "",
           f"* Spearman, judge rating vs. 3-gram percentile: {rho_ng:.2f}; vs. name length: {rho_len:.2f}; "
           f"vs. distance from the training set: {rho_nov:.2f}",
           f"* Judge >= 3 and 3-gram >= 0.10 agree on {100 * (j == g3).mean():.1f}% of names. "
           f"Judge passes {100 * j.mean():.1f}%, 3-gram passes {100 * g3.mean():.1f}%.",
           f"* Of names the 3-gram passes, the judge rejects {100 * (~j[g3]).mean():.1f}%. "
           f"Of names the 3-gram rejects, the judge passes {100 * j[~g3].mean():.1f}%.", ""]

    # 3. Same loss, two sides of the minimum.
    md += ["## 3. Does loss determine creativity?", "",
           "Each run passes through the same validation loss twice: once undertrained, once overfit. If the",
           "score were loss in disguise the two visits would score the same. Mean of 3 seeds, base definition,",
           "values interpolated along each side of the curve.", ""]
    same = []
    for size in [s for s in SIZE_ORDER if s in set(d["size"])]:
        m_ = d[d["size"] == size].groupby("step")[["val_loss", "base", "copy_rate", "novel_rate"]].mean()
        k = int(np.argmin(m_["val_loss"].to_numpy()))
        under, over = m_.iloc[:k + 1].sort_values("val_loss"), m_.iloc[k:].sort_values("val_loss")
        for extra in (0.05, 0.10, 0.20):
            level = m_["val_loss"].min() + extra
            if level > under["val_loss"].max() or level > over["val_loss"].max():
                continue
            f = lambda side, col: float(np.interp(level, side["val_loss"], side[col]))  # noqa: E731
            same.append({"size": size, "validation loss": level,
                         "creativity, undertrained side": f(under, "base"), "creativity, overfit side": f(over, "base"),
                         "copies, undertrained side": f(under, "copy_rate"), "copies, overfit side": f(over, "copy_rate")})
    rho_all = spearmanr(d["val_loss"], d["base"]).statistic
    md += [md_table(pd.DataFrame(same)), "",
           f"Across all {len(d)} checkpoints, Spearman between validation loss and the base creativity rate is {rho_all:.2f}.", ""]
    (RESULTS / f"sensitivity{tag}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    figures(d, keys, arch)


def figures(d: pd.DataFrame, keys: list[str], arch: str = "lstm") -> None:
    sizes = [s for s in SIZE_ORDER if s in set(d["size"])]
    fig, axes = plt.subplots(len(keys), len(sizes), figsize=(3.1 * len(sizes) + 0.8, 1.55 * len(keys) + 0.6),
                             sharex=True, sharey="row", squeeze=False, gridspec_kw={"hspace": 0.2, "wspace": 0.08})
    for j, size in enumerate(sizes):
        g = d[d["size"] == size]
        mean = g.groupby("step")[keys + ["val_loss"]].mean()
        best = mean["val_loss"].idxmin()
        for i, key in enumerate(keys):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.axvline(best, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            for _, run in g.groupby("run_seed"):
                run = run.sort_values("step")
                ax.plot(run["step"], run[key], color=SERIES["blue"], linewidth=0.8, alpha=0.35)
            ax.plot(mean.index, mean[key], color=SERIES["blue"], linewidth=1.8)
            ax.plot(mean[key].idxmax(), mean[key].max(), "o", color=SERIES["blue"], markersize=4.5)
            ax.set_ylim(0, None)
            if i == 0:
                ax.set_title(f"{size} · dashed: lowest loss, step {int(best):,}", loc="left", fontsize=9, color=INK)
            if j == 0:
                ax.set_ylabel(LABELS[key], fontsize=8, rotation=0, ha="right", va="center")
            if i == len(keys) - 1:
                ax.set_xlabel("training step (log scale)")
    fig.savefig(RESULTS / "figures" / f"sensitivity_{arch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(sizes), figsize=(3.3 * len(sizes) + 0.6, 3.4), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.08})
    for j, size in enumerate(sizes):
        ax = axes[0, j]
        m_ = d[d["size"] == size].groupby("step")[["val_loss", "base"]].mean()
        k = int(np.argmin(m_["val_loss"].to_numpy()))
        ax.grid(color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.plot(m_["val_loss"].iloc[:k + 1], m_["base"].iloc[:k + 1], "-o", color=SERIES["blue"], linewidth=2, markersize=3.5,
                label="undertrained side")
        ax.plot(m_["val_loss"].iloc[k:], m_["base"].iloc[k:], "-o", color=SERIES["orange"], linewidth=2, markersize=3.5,
                label="overfit side")
        ax.plot(m_["val_loss"].iloc[k], m_["base"].iloc[k], "o", color=INK, markersize=7)
        ax.annotate("lowest loss", (m_["val_loss"].iloc[k], m_["base"].iloc[k]), textcoords="offset points",
                    xytext=(8, 8), fontsize=8.5, color=INK)
        ax.set_title(size, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("validation loss (nats per symbol)")
        ax.set_ylim(0, 1)
        if j == 0:
            ax.set_ylabel("creativity rate")
            ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.savefig(RESULTS / "figures" / f"loss_vs_creativity_{arch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lstm")
