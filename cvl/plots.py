"""Report figures. Each function saves a PNG and returns its path.

Fig. 1  plot_training_curve: validation loss and creativity rate vs. step,
        with the loss minimum and the creativity peak marked.
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
