"""Creativity scorer: the one function every experiment calls.

    from cvl.scorer import score, score_table
    result = score(samples)          # dict of batch-level numbers
    table = score_table(samples)     # pandas DataFrame, one row per sample

Per name:
    novelty       1 - max normalized Levenshtein similarity to any training name
    plausibility  percentile of the name's mean log-prob (reference n-gram)
                  among real validation names, 0 to 1
    creative      novelty >= novelty_threshold and plausibility >= plausibility_threshold

Per batch:
    creativity_rate   unique creative names / total samples   (headline number)
    diversity         unique names / total samples
    novelty, plausibility   means over all samples
    copy_rate         share of samples that exactly match a training name
    heldout_hit_rate  share of samples that exactly match a val or test name

Thresholds and the n-gram order live in scorer_config.json, written once by
scripts/calibrate.py and then frozen for the whole project.
"""
from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import PROCESSED_DIR, ROOT, load_split
from .ngram import CharNGram
from .novelty import NearestName

CONFIG_PATH = ROOT / "scorer_config.json"
EPS = 1e-9


def split_fingerprint(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()[:12]


@dataclass
class ScorerConfig:
    ngram_order: int
    novelty_threshold: float
    plausibility_threshold: float
    train_fingerprint: str

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ScorerConfig":
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/calibrate.py once first")
        d = json.loads(path.read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


class Scorer:
    def __init__(self, train: list[str], val: list[str], test: list[str],
                 ngram_order: int = 3, novelty_threshold: float = 0.0,
                 plausibility_threshold: float = 0.10):
        self.train = train
        self.train_set = set(train)
        self.heldout_set = set(val) | set(test)
        self.novelty_threshold = novelty_threshold
        self.plausibility_threshold = plausibility_threshold
        self.lm = CharNGram(ngram_order).fit(train)
        self.nearest_name = NearestName(train)
        # Plausibility is a percentile against real names the reference never saw.
        self._ref_scores = sorted(self.lm.avg_logprob(n) for n in val)
        self._cache: dict[str, dict] = {}

    @classmethod
    def from_config(cls, path: Path = CONFIG_PATH, data_dir: Path = PROCESSED_DIR) -> "Scorer":
        cfg = ScorerConfig.load(path)
        train = load_split("train", data_dir)
        if split_fingerprint(train) != cfg.train_fingerprint:
            raise RuntimeError("train split changed since calibration; rerun prepare_data.py "
                               "with the original seed, or recalibrate before any experiments")
        return cls(train, load_split("val", data_dir), load_split("test", data_dir),
                   cfg.ngram_order, cfg.novelty_threshold, cfg.plausibility_threshold)

    # ---- per name -------------------------------------------------------
    def plausibility_of_logprob(self, lp: float) -> float:
        return bisect.bisect_right(self._ref_scores, lp) / len(self._ref_scores)

    def score_name(self, name: str) -> dict:
        if name in self._cache:
            return self._cache[name]
        if not name:
            row = dict(novelty=0.0, nearest="", logprob=float("-inf"), plausibility=0.0)
        else:
            sim, nearest = self.nearest_name.nearest(name)
            lp = self.lm.avg_logprob(name)
            row = dict(novelty=1.0 - sim, nearest=nearest, logprob=lp,
                       plausibility=self.plausibility_of_logprob(lp))
        row["in_train"] = name in self.train_set
        row["in_heldout"] = name in self.heldout_set
        # Tolerance: novelty takes values like 1/6 and the threshold sits on one of them.
        row["novel"] = row["novelty"] >= self.novelty_threshold - EPS and not row["in_train"]
        row["plausible"] = row["plausibility"] >= self.plausibility_threshold - EPS
        row["creative"] = row["novel"] and row["plausible"]
        self._cache[name] = row
        return row

    # ---- per batch ------------------------------------------------------
    def table(self, names: list[str]) -> pd.DataFrame:
        names = [clean_sample(n) for n in names]
        rows = [{"name": n, **self.score_name(n)} for n in names]
        df = pd.DataFrame(rows)
        df["first_occurrence"] = ~df["name"].duplicated()
        return df

    def score(self, names: list[str]) -> dict:
        if not names:
            raise ValueError("no samples to score")
        df = self.table(names)
        n = len(df)
        uniq = df[df["first_occurrence"]]
        return {
            "n": n,
            "creativity_rate": float(uniq["creative"].sum() / n),
            "diversity": float(len(uniq) / n),
            "novelty": float(df["novelty"].mean()),
            "plausibility": float(df["plausibility"].mean()),
            "novel_rate": float(df["novel"].mean()),
            "plausible_rate": float(df["plausible"].mean()),
            "copy_rate": float(df["in_train"].mean()),
            "heldout_hit_rate": float(df["in_heldout"].mean()),
            "mean_length": float(df["name"].str.len().mean()),
        }


def clean_sample(name: str) -> str:
    """Strip whitespace and end markers a sampler may leave in."""
    return name.replace("\n", "").replace("$", "").strip().lower()


_default: Scorer | None = None


def default_scorer() -> Scorer:
    global _default
    if _default is None:
        _default = Scorer.from_config()
    return _default


def score(names: list[str]) -> dict:
    """Batch-level creativity numbers using the frozen project scorer."""
    return default_scorer().score(names)


def score_table(names: list[str]) -> pd.DataFrame:
    """One row per sample, for error analysis."""
    return default_scorer().table(names)
