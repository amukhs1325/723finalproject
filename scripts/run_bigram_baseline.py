"""E1: bigram baseline, plus its decoding sweep (the E4 setup on the baseline).

    python scripts/run_bigram_baseline.py

Writes:
  checkpoints/bigram.json                the baseline model
  outputs/baseline/bigram_sweep.csv      one row per (setting, seed)
  outputs/baseline/bigram_summary.csv    mean and std over seeds per setting
  outputs/baseline/bigram_plane.png      temperature and top-p curves on the plane
  outputs/baseline/bigram_examples.txt   example names per setting
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, load_split  # noqa: E402
from cvl.ngram import CharNGram  # noqa: E402
from cvl.plots import CREATIVE_COLOR, INK, LOSS_COLOR, MUTED  # noqa: E402
from cvl.sampling import sample  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402

N = 1000
SEEDS = [0, 1, 2]
TEMPS = [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
TOP_PS = [0.5, 0.7, 0.8, 0.9, 0.95]
OUT = ROOT / "outputs" / "baseline"
CKPT = ROOT / "checkpoints" / "bigram.json"


def main():
    train, val = load_split("train"), load_split("val")
    model = CharNGram(2).fit(train)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    CKPT.write_text(json.dumps(model.to_dict()))
    val_loss = model.val_loss(val)
    print(f"bigram val loss {val_loss:.4f} nats/char (perplexity {model.perplexity(val):.2f})")

    settings = [("temperature", t, t, 1.0) for t in TEMPS] + \
               [("top_p", p, 1.0, p) for p in TOP_PS]
    scorer = default_scorer()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for sweep, value, temp, top_p in settings:
        for seed in SEEDS:
            names = sample(CKPT, N, temperature=temp, top_p=top_p, seed=seed)
            rows.append({"sweep": sweep, "value": value, "temperature": temp, "top_p": top_p,
                         "seed": seed, "val_loss": val_loss, **scorer.score(names)})
            if seed == 0:
                examples.append(f"{sweep}={value}: " + ", ".join(names[:12]))
        print(f"{sweep}={value} done")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bigram_sweep.csv", index=False)
    metrics = ["creativity_rate", "diversity", "novel_rate", "plausible_rate", "novelty",
               "plausibility", "copy_rate", "heldout_hit_rate", "mean_length"]
    summary = df.groupby(["sweep", "value"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "bigram_summary.csv", index=False)
    (OUT / "bigram_examples.txt").write_text("\n".join(examples) + "\n")

    show = ["sweep", "value", "creativity_rate_mean", "creativity_rate_std", "novel_rate_mean",
            "plausible_rate_mean", "copy_rate_mean", "diversity_mean"]
    print()
    print(summary[show].round(3).to_string(index=False))
    plot_sweeps(summary)


def plot_sweeps(summary: pd.DataFrame) -> None:
    """Plane: each sweep as a connected curve, labelled by its value."""
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for sweep, color in [("temperature", CREATIVE_COLOR), ("top_p", LOSS_COLOR)]:
        d = summary[summary.sweep == sweep].sort_values("value")
        ax.plot(d.novel_rate_mean, d.plausible_rate_mean, "-o", color=color, ms=5, lw=1.4,
                label="temperature (top-p 1.0)" if sweep == "temperature" else "top-p (temperature 1.0)")
        # top-p labels above-right; temperature labels to the left, clear of its falling line
        offset, ha = ((4, 6), "left") if sweep == "top_p" else ((-7, -3), "right")
        for _, r in d.iterrows():
            ax.annotate(f"{r.value:g}", (r.novel_rate_mean, r.plausible_rate_mean), ha=ha,
                        xytext=offset, textcoords="offset points", fontsize=7.5, color=color)
    ax.set_xlabel("share novel  → newer")
    ax.set_ylabel("share plausible  → more name-like")
    ax.set_title("Bigram baseline: top-p stays plausible longer than temperature", loc="left",
                 fontsize=10.5, color=INK)
    ax.grid(color="#EEEEEE", lw=.6, zorder=0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.tick_params(colors=MUTED)
    fig.savefig(OUT / "bigram_plane.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved summary, no sampling
        plot_sweeps(pd.read_csv(OUT / "bigram_summary.csv"))
    else:
        main()
