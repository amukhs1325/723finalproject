"""Report figures. Each function saves a PNG and returns its path.

Fig. 1  plot_run: three stacked panels sharing the step axis (no second y-axis):
        validation loss; creativity rate (judge-based, with its error band, and the
        3-gram and student-only versions for comparison); where the samples went
        (novel, respelling, near copy, derived, copy).
        plot_training_curve is the older single-panel version.
Fig. 2  plot_plane: checkpoints (or any labeled batches) on the
        novelty vs. plausibility plane.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LOSS_COLOR = "#4C6A92"
CREATIVE_COLOR = "#C0582E"
INK = "#2B2B2B"
MUTED = "#8A8A8A"

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "figure.dpi": 150, "savefig.bbox": "tight",
})


def plot_training_curve(df: pd.DataFrame, out: str | Path, title: str = "",
                        step_col: str = "step", loss_col: str = "val_loss",
                        creativity_col: str = "creativity_rate",
                        x_label: str = "training step") -> Path:
    """df: one row per checkpoint with step, val_loss, creativity_rate."""
    df = df.sort_values(step_col)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)

    ax.plot(df[step_col], df[loss_col], color=LOSS_COLOR, lw=1.8)
    ax2.plot(df[step_col], df[creativity_col], color=CREATIVE_COLOR, lw=1.8)

    lmin = df.loc[df[loss_col].idxmin()]
    cpk = df.loc[df[creativity_col].idxmax()]
    ax.axvline(lmin[step_col], color=LOSS_COLOR, ls="--", lw=1, alpha=.7)
    ax2.axvline(cpk[step_col], color=CREATIVE_COLOR, ls="--", lw=1, alpha=.7)
    ax.annotate(f"loss min\nstep {int(lmin[step_col])}", (lmin[step_col], lmin[loss_col]),
                xytext=(6, 10), textcoords="offset points", color=LOSS_COLOR, fontsize=8.5)
    ax2.annotate(f"creativity peak\nstep {int(cpk[step_col])}", (cpk[step_col], cpk[creativity_col]),
                 xytext=(6, -26), textcoords="offset points", color=CREATIVE_COLOR, fontsize=8.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel("validation loss (nats/char)", color=LOSS_COLOR)
    ax2.set_ylabel("creativity rate", color=CREATIVE_COLOR)
    gap = int(lmin[step_col] - cpk[step_col])
    ax.set_title(title or f"Creativity peaks {gap} steps before the loss minimum"
                 if gap > 0 else title or "Creativity vs. loss over training",
                 loc="left", fontsize=10.5, color=INK)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_plane(df: pd.DataFrame, out: str | Path, label_col: str = "label",
               color_col: str | None = None, novelty_threshold: float | None = None,
               plausibility_threshold: float | None = None, title: str = "",
               x_col: str = "novel_rate", y_col: str = "plausible_rate") -> Path:
    """df: one row per batch (checkpoint, decoding setting, sanity set).

    Default axes are the share of samples clearing each threshold; pass
    x_col="novelty", y_col="plausibility" to plot the means instead.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if color_col:
        sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap="viridis", s=42,
                        edgecolor="white", lw=.6, zorder=3)
        cb = fig.colorbar(sc, ax=ax, fraction=.046, pad=.03)
        cb.set_label(color_col, color=INK)
        cb.outline.set_visible(False)
        if len(df) > 1:
            d = df.sort_values(color_col)
            ax.plot(d[x_col], d[y_col], color=MUTED, lw=.8, zorder=2)
    else:
        # Point area shows diversity, so a model stuck on a few names is visibly small.
        sizes = 8 + 110 * df["diversity"] if "diversity" in df else 42
        ax.scatter(df[x_col], df[y_col], color=CREATIVE_COLOR, s=sizes, edgecolor="white",
                   lw=.6, zorder=3)
        if "diversity" in df:
            ax.text(1.0, -0.2, "point size = diversity (unique / total)", transform=ax.transAxes,
                    ha="right", fontsize=8, color=MUTED)
    if label_col in df:
        for _, r in df.iterrows():
            ax.annotate(str(r[label_col]), (r[x_col], r[y_col]), xytext=(5, 4),
                        textcoords="offset points", fontsize=8, color=INK)
    if novelty_threshold is not None:
        ax.axvline(novelty_threshold, color=MUTED, ls=":", lw=1)
    if plausibility_threshold is not None:
        ax.axhline(plausibility_threshold, color=MUTED, ls=":", lw=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    nice = {"novel_rate": "share novel", "plausible_rate": "share plausible",
            "novelty": "mean novelty", "plausibility": "mean plausibility"}
    ax.set_xlabel(nice.get(x_col, x_col) + "  → newer")
    ax.set_ylabel(nice.get(y_col, y_col) + "  → more name-like")
    ax.set_title(title or "Novelty vs. plausibility", loc="left", fontsize=10.5, color=INK)
    ax.grid(color="#EEEEEE", lw=.6, zorder=0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


# Categorical colors, fixed order, validated for colorblind separation on a light surface.
SERIES = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4"}
TIER_PARTS = [("novel_rate", "novel", "blue"), ("respelling_rate", "respelling", "orange"),
              ("near_copy_rate", "near copy", "aqua"), ("derived_rate", "derived or famous", "yellow"),
              ("copy_rate", "exact copy", "magenta")]


def plot_run(df: pd.DataFrame, out: str | Path, title: str = "") -> Path:
    """One training run: loss, creativity and novelty tiers against training step.

    df: rows of results/checkpoints.csv for a single run (one row per checkpoint).
    """
    d = df.sort_values("step").reset_index(drop=True)
    x = d["step"].to_numpy(dtype=float)
    has_judge = "creativity_rate_judge" in d and d["creativity_rate_judge"].notna().all()
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(6.6, 7.4), sharex=True,
                                  gridspec_kw={"height_ratios": [1, 1.25, 1.1], "hspace": 0.16})
    for ax in (a, b, c):
        ax.set_xscale("log")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
        ax.set_axisbelow(True)
    best = int(d["val_loss"].idxmin())
    for ax in (a, b, c):
        ax.axvline(x[best], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))

    a.plot(x, d["val_loss"], color=INK, linewidth=2)
    a.plot(x[best], d["val_loss"][best], "o", color=INK, markersize=6)
    a.annotate(f"lowest loss, step {int(x[best]):,}", (x[best], d["val_loss"][best]), textcoords="offset points",
               xytext=(10, 16), fontsize=9, color=INK)
    a.set_ylabel("validation loss\n(nats per symbol)")
    if title:
        a.set_title(title, loc="left", fontsize=11, color=INK)

    head = d["creativity_rate_judge"] if has_judge else d["creativity_rate"]
    if has_judge:
        se = d["creativity_rate_se"].fillna(0)
        b.fill_between(x, head - 2 * se, head + 2 * se, color=SERIES["blue"], alpha=0.18, linewidth=0)
        b.plot(x, d["creativity_rate_ngram"], color=SERIES["orange"], linewidth=1.5, label="3-gram plausibility")
        if "creativity_rate_student" in d:
            b.plot(x, d["creativity_rate_student"], color=SERIES["aqua"], linewidth=1.5, linestyle=(0, (4, 2)),
                   label="student alone (uncorrected)")
    b.plot(x, head, color=SERIES["blue"], linewidth=2,
           label="judge (student + judged sample, ±2 SE)" if has_judge else "creativity rate")
    peak = int(head.idxmax())
    b.plot(x[peak], head[peak], "o", color=SERIES["blue"], markersize=6)
    b.annotate(f"peak, step {int(x[peak]):,}", (x[peak], head[peak]), textcoords="offset points",
               xytext=(-10, -34), fontsize=9, color=INK, ha="center")
    b.set_ylabel("creativity rate")
    b.set_ylim(0, 1)
    b.legend(frameon=False, fontsize=8.5, loc="lower left")

    parts = [(d[col] + (d["famous_rate"] if col == "derived_rate" else 0)).to_numpy() for col, _, _ in TIER_PARTS]
    c.stackplot(x, parts, colors=[SERIES[k] for _, _, k in TIER_PARTS], labels=[n for _, n, _ in TIER_PARTS],
                edgecolor="white", linewidth=0.8)
    c.set_ylim(0, 1)
    c.set_ylabel("share of samples")
    c.set_xlabel("training step (log scale)")
    c.legend(frameon=False, fontsize=8.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out)


SIZE_ORDER = ["small", "medium", "large"]
GRID_ROWS = [("val_loss", "validation loss\n(nats per symbol)", INK),
             ("creativity", "creativity rate", SERIES["blue"]),
             ("copy_rate", "exact copies\n(share of samples)", SERIES["magenta"])]


def split_run(run: str) -> tuple[str, str, int] | None:
    """'lstm_medium_s0' -> ('lstm', 'medium', 0). None for runs not named that way."""
    parts = run.rsplit("_", 2)
    if len(parts) != 3 or not parts[2].startswith("s") or not parts[2][1:].isdigit():
        return None
    return parts[0], parts[1], int(parts[2][1:])


def plot_grid(df: pd.DataFrame, out: str | Path, arch: str = "lstm") -> Path:
    """One architecture, all sizes and seeds: a column per size, a row per measure.

    Thin lines are single seeds, the heavy line is their mean. The dashed line marks the
    step where the mean validation loss is lowest. Rows share a y-axis across sizes.
    df: rows of results/checkpoints.csv (temperature 1, top-p 1).
    """
    d = df.copy()
    keys = d["run"].map(split_run)
    d = d[keys.notna()]
    d[["arch", "size", "run_seed"]] = pd.DataFrame(keys[keys.notna()].tolist(), index=d.index)
    d = d[d["arch"] == arch]
    judged = d["creativity_rate_judge"].notna().all()
    d["creativity"] = d["creativity_rate_judge"] if judged else d["creativity_rate"]
    sizes = [s for s in SIZE_ORDER if s in set(d["size"])] + sorted(set(d["size"]) - set(SIZE_ORDER))
    fig, axes = plt.subplots(len(GRID_ROWS), len(sizes), figsize=(3.1 * len(sizes) + 0.6, 6.6),
                             sharex=True, sharey="row", squeeze=False,
                             gridspec_kw={"hspace": 0.14, "wspace": 0.08})
    for j, size in enumerate(sizes):
        g = d[d["size"] == size]
        mean = g.groupby("step")[[c for c, _, _ in GRID_ROWS]].mean()
        best = mean["val_loss"].idxmin()
        for i, (col, label, color) in enumerate(GRID_ROWS):
            ax = axes[i, j]
            ax.set_xscale("log")
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)
            ax.set_axisbelow(True)
            ax.axvline(best, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            for _, run in g.groupby("run_seed"):
                run = run.sort_values("step")
                ax.plot(run["step"], run[col], color=color, linewidth=0.9, alpha=0.4)
            ax.plot(mean.index, mean[col], color=color, linewidth=2)
            if i == 0:
                params = g["params"].iloc[0] if "params" in g and g["params"].notna().any() else None
                head = f"{size}" + (f" · {int(params):,} parameters" if params else "")
                ax.set_title(head, loc="left", fontsize=10, color=INK)
                ax.annotate(f"lowest loss\nstep {int(best):,}", (best, 1), xycoords=("data", "axes fraction"),
                            textcoords="offset points", xytext=(6, -4), fontsize=8, color=INK, va="top")
            if j == 0:
                ax.set_ylabel(label)
            if col != "val_loss":
                ax.set_ylim(0, 1)
            if i == len(GRID_ROWS) - 1:
                ax.set_xlabel("training step (log scale)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(out)
