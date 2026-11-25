"""Tests for the transformer: sizes matched to the LSTM, causal attention, the shared interface."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_sizes_match_the_lstm_within_two_percent():
    from models.lstm import CharLSTM
    from models.transformer import SIZES, CharTransformer
    counts = {k: CharTransformer.of_size(k).n_params() for k in SIZES}
    assert counts == {"small": 24989, "medium": 99029, "large": 401249}
    for k, n in counts.items():
        assert abs(n - CharLSTM.of_size(k).n_params()) / CharLSTM.of_size(k).n_params() < 0.02


def test_a_position_never_sees_later_characters_or_padding():
    from cvl.reference_lm import encode_batch
    from models.transformer import CharTransformer
    torch.manual_seed(0)
    m = CharTransformer.of_size("small").eval()
    x, _ = encode_batch(["mira"])
    changed = x.clone()
    changed[0, -1] = 7                                    # change the last input only
    assert torch.equal(m(x)[0][0, :-1], m(changed)[0][0, :-1])
    batch, _ = encode_batch(["mira", "averyverylongna"])  # the same name, padded out in a batch
    assert torch.allclose(m(x)[0][0], m(batch)[0][0, :x.shape[1]], atol=1e-5)
    assert abs(m.loss(["mira"]) - m.loss(["mira", "mira"])) < 1e-6


def test_follows_the_shared_interface(tmp_path):
    from cvl.reference_lm import SYMBOLS
    from cvl.sampling import load_checkpoint, sample
    from models.transformer import CharTransformer
    torch.manual_seed(0)
    m = CharTransformer.of_size("small")
    m.meta = {"step": 7, "val_loss": 3.2}
    assert m.symbols == SYMBOLS and abs(m.next_probs("ab").sum() - 1) < 1e-9
    names = sample(m, 30, seed=5)
    assert names == sample(m, 30, seed=5) and all("$" not in n and len(n) <= 20 for n in names)
    assert 3.0 < m.loss(["thorin", "mira"]) < 3.9          # untrained: close to ln(29) = 3.37
    # the batched sampler and next_probs describe the same model
    first = np.array([SYMBOLS.index(n[0]) if n else 0 for n in sample(m, 4000, seed=1)])
    assert np.abs(np.bincount(first, minlength=29) / 4000 - m.next_probs("")).max() < 0.03
    torch.save(m.checkpoint(), tmp_path / "step7.pt")
    again = load_checkpoint(tmp_path / "step7.pt")
    assert type(again).__name__ == "CharTransformer" and again.meta == {"step": 7, "val_loss": 3.2}
    assert sample(again, 30, seed=5) == names


def test_training_loop_is_shared_with_the_lstm(tmp_path):
    from training.train_lstm import save_steps, train_run
    names = ["thorin", "mira", "eldan", "kaelith", "bryn", "sorrel"] * 30
    out = train_run("small", seed=3, steps=60, out_root=tmp_path, train=names, val=names[:6], quiet=True,
                    arch="transformer", lr=1e-3)
    assert out.name == "transformer_small_s3"
    log = pd.read_csv(out / "log.csv")
    assert log["step"].tolist() == save_steps(60) and log["val_loss"].iloc[-1] < 3.3    # it learns something
    from cvl.sampling import load_checkpoint
    model = load_checkpoint(out / "step60.pt")
    assert model.meta["params"] == 24989 and model.meta["size"] == "small"
