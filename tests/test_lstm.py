"""Tests for the LSTM grid: sizes, the checkpoint layout, the run summary and the grid figure."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_sizes_hit_the_agreed_parameter_counts():
    from models.lstm import SIZES, CharLSTM
    counts = {k: CharLSTM.of_size(k).n_params() for k in SIZES}
    assert counts == {"small": 25321, "medium": 99967, "large": 400325}
    for k, target in {"small": 25_000, "medium": 100_000, "large": 400_000}.items():
        assert abs(counts[k] - target) / target < 0.02


def test_a_run_writes_checkpoints_the_scorer_can_open(tmp_path):
    from cvl.sampling import load_checkpoint, sample
    from training.train_lstm import save_steps, train_run
    names = ["thorin", "mira", "eldan", "kaelith", "bryn", "sorrel"] * 30
    out = train_run("small", seed=3, steps=60, out_root=tmp_path, train=names, val=names[:6], quiet=True)
    assert out.name == "lstm_small_s3"
    steps = save_steps(60)
    assert sorted(int(f.stem[4:]) for f in out.glob("step*.pt")) == steps
    log = pd.read_csv(out / "log.csv")
    assert list(log.columns) == ["step", "train_loss", "val_loss"] and log["step"].tolist() == steps
    model = load_checkpoint(out / f"step{steps[-1]}.pt")
    assert type(model).__name__ == "CharLSTM" and model.meta["size"] == "small" and model.meta["params"] == 25321
    assert abs(model.meta["val_loss"] - model.loss(names[:6])) < 1e-6
    assert sample(model, 20, seed=1) == sample(out / f"step{steps[-1]}.pt", 20, seed=1)
    # same seed, same run
    again = train_run("small", seed=3, steps=60, out_root=tmp_path / "b", train=names, val=names[:6], quiet=True)
    assert pd.read_csv(again / "log.csv").equals(log)


def test_checkpoint_steps_are_shared_and_log_spaced():
    from training.train_lstm import save_steps
    s = save_steps()
    assert s[0] == 50 and s[-1] == 40000 and len(s) == 28 and s == sorted(set(s))
    ratios = np.diff(np.log(s))
    assert ratios.max() - ratios.min() < 0.02


def _fake_table():
    rows = []
    steps = [50, 100, 200, 400, 800]
    for size in ["small", "large"]:
        for seed in range(3):
            for k, step in enumerate(steps):
                rows.append({"run": f"lstm_{size}_s{seed}", "step": step, "temperature": 1.0, "top_p": 1.0, "n": 1000,
                             "val_loss": [2.6, 2.4, 2.3, 2.35, 2.6][k] + 0.001 * seed,
                             "creativity_rate": 0.5, "creativity_rate_judge": [0.60, 0.70, 0.62, 0.50, 0.30][k] + 0.01 * seed,
                             "creativity_rate_se": 0.02, "copy_rate": 0.1 * k, "params": 25321 if size == "small" else 400325})
    rows.append({"run": "bigram", "step": None, "temperature": 1.0, "top_p": 1.0, "n": 1000, "val_loss": 2.5,
                 "creativity_rate": 0.4, "creativity_rate_judge": 0.4, "creativity_rate_se": 0.02, "copy_rate": 0.0})
    return pd.DataFrame(rows)


def test_summary_picks_the_creative_step_from_the_other_seeds():
    from scripts.summarize_runs import by_size, md_table, summarize
    s = summarize(_fake_table())
    assert len(s) == 6 and set(s["size"]) == {"small", "large"}
    assert (s["best_step"] == 200).all() and (s["peak_step"] == 100).all() and (s["heldout_step"] == 100).all()
    assert np.allclose(s["heldout_gap"], 0.08) and np.allclose(s["heldout_gap_se"], np.hypot(0.02, 0.02))
    t = by_size(s)
    assert t["size"].tolist() == ["small", "large"] and np.allclose(t["held-out gap"], 0.08)
    assert np.allclose(t["gap SE"], np.hypot(0.02, 0.02) / np.sqrt(3))
    assert md_table(t).count("\n") == 3


def test_summary_ignores_its_own_seed_when_choosing():
    from scripts.summarize_runs import summarize
    d = _fake_table()
    lucky = (d["run"] == "lstm_small_s0") & (d["step"] == 800)
    d.loc[lucky, "creativity_rate_judge"] = 0.99          # one run's fluke must not pick its own step
    s = summarize(d).set_index("run")
    assert s.loc["lstm_small_s0", "peak_step"] == 800 and s.loc["lstm_small_s0", "heldout_step"] == 100


def test_grid_figure_draws(tmp_path):
    from cvl.plots import plot_grid, split_run
    assert split_run("lstm_medium_s2") == ("lstm", "medium", 2) and split_run("bigram") is None
    assert split_run("reference_lstm") is None
    out = plot_grid(_fake_table(), tmp_path / "g.png")
    assert out.exists() and out.stat().st_size > 10_000
