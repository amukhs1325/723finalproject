"""Creativity scorer (v2): the one function every experiment calls.

    from cvl.scorer import score, score_table
    result = score(samples)          # dict of batch-level numbers
    table = score_table(samples)     # pandas DataFrame, one row per sample

A sample is creative if it is novel and plausible (docs/evaluation_plan.md).

Novelty, deterministic. `tier` is the first of these that applies:
    copy        exact training name
    respelling  same sound key as a training name            (khris -> chris)
    near_copy   spelling similarity above the threshold      (magnuss -> magnus)
    derived     embedding cosine above the threshold         (eldrath -> eldra)
    famous      on the outside famous-names list             (aragorn)
    novel       none of the above
`novelty` stays the continuous spelling distance, 1 - max normalized Levenshtein similarity.

Plausibility, for novel names only:
    gate        length min_len..max_len (3..15), otherwise not plausible
    mode "ngram"  percentile under the reference n-gram >= plausibility_threshold
    mode "judge"  cached LLM judge rating (expected value, 1.0 to 5.0) >= judge_cut (cvl/judge.py)
The mode is set in scorer_config.json. It is "ngram" until the judge has passed its
acceptance test against human ratings; the n-gram numbers are always reported too.

Per batch:
    creativity_rate   unique creative names / total samples   (headline number)
    diversity         unique names / total samples
    *_rate            share of samples in each novelty tier
Thresholds live in scorer_config.json, written by scripts/calibrate.py and then frozen.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import PROCESSED_DIR, ROOT, clean_name, load_split
from .judge import Judge, MissingJudgments
from .ngram import CharNGram
from .novelty import MeaningIndex, NearestName, SoundIndex, SpliceIndex

CONFIG_PATH = ROOT / "scorer_config.json"
FAMOUS_PATH = ROOT / "data" / "external" / "famous_fantasy_names.txt"
EPS = 1e-9
TIERS = ["copy", "respelling", "near_copy", "derived", "famous", "novel"]


def split_fingerprint(names: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()[:12]


def load_famous(path: Path = FAMOUS_PATH) -> set[str]:
    if not path.exists():
        return set()
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines()]
    return {c for c in (clean_name(x) for x in lines if x and not x.startswith("#")) if c}


@dataclass
class ScorerConfig:
    ngram_order: int
    novelty_threshold: float
    plausibility_threshold: float
    train_fingerprint: str
    version: int = 1
    meaning_model: str | None = None
    meaning_revision: str | None = None
    meaning_threshold: float | None = None
    min_len: int = 3
    max_len: int = 15
    plausibility_mode: str = "ngram"
    judge_model: str | None = None
    judge_prompt_hash: str | None = None
    judge_cut: float | None = None

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ScorerConfig":
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run scripts/calibrate.py once first")
        d = json.loads(path.read_text())
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


class Scorer:
    def __init__(self, train: list[str], val: list[str], test: list[str],
                 ngram_order: int = 3, novelty_threshold: float = 0.0,
                 plausibility_threshold: float = 0.10, *,
                 meaning: MeaningIndex | None = None, meaning_threshold: float | None = None,
                 famous: set[str] | None = None, min_len: int = 3, max_len: int = 15,
                 plausibility_mode: str = "ngram", judge: Judge | None = None,
                 judge_cut: float | None = None):
        if plausibility_mode not in ("ngram", "judge"):
            raise ValueError(f"unknown plausibility_mode: {plausibility_mode}")
        if plausibility_mode == "judge" and (judge is None or judge_cut is None):
            raise ValueError("judge mode needs a Judge and a judge_cut")
        self.train = train
        self.train_set = set(train)
        self.heldout_set = set(val) | set(test)
        self.novelty_threshold = novelty_threshold
        self.plausibility_threshold = plausibility_threshold
        self.meaning, self.meaning_threshold = meaning, meaning_threshold
        self.famous = set(famous or ()) - self.train_set
        self.min_len, self.max_len = min_len, max_len
        self.plausibility_mode, self.judge, self.judge_cut = plausibility_mode, judge, judge_cut
        self.lm = CharNGram(ngram_order).fit(train)
        self.nearest_name = NearestName(train)
        self.sound = SoundIndex(train)
        self.splice = SpliceIndex(train)
        # n-gram plausibility is a percentile against real names the reference never saw.
        self._ref_scores = sorted(self.lm.avg_logprob(n) for n in val)
        self._cache: dict[str, dict] = {}

    @classmethod
    def from_config(cls, path: Path = CONFIG_PATH, data_dir: Path = PROCESSED_DIR) -> "Scorer":
        cfg = ScorerConfig.load(path)
        train = load_split("train", data_dir)
        if split_fingerprint(train) != cfg.train_fingerprint:
            raise RuntimeError("train split changed since calibration; rerun prepare_data.py "
                               "with the original seed, or recalibrate before any experiments")
        meaning = None
        if cfg.meaning_threshold is not None:
            meaning = MeaningIndex(train, cache_dir=data_dir / ".cache",
                                   model=cfg.meaning_model, revision=cfg.meaning_revision)
        judge = None
        if cfg.judge_model:
            judge = Judge(model=cfg.judge_model)
            if cfg.judge_prompt_hash and judge.prompt_hash != cfg.judge_prompt_hash:
                raise RuntimeError("judge prompt changed since calibration (cvl/judge.py); "
                                   "restore it or recalibrate the judge")
        return cls(train, load_split("val", data_dir), load_split("test", data_dir),
                   cfg.ngram_order, cfg.novelty_threshold, cfg.plausibility_threshold,
                   meaning=meaning, meaning_threshold=cfg.meaning_threshold,
                   famous=load_famous(), min_len=cfg.min_len, max_len=cfg.max_len,
                   plausibility_mode=cfg.plausibility_mode, judge=judge, judge_cut=cfg.judge_cut)

    # ---- per name: everything that does not depend on the judge ---------------
    def plausibility_of_logprob(self, lp: float) -> float:
        return bisect.bisect_right(self._ref_scores, lp) / len(self._ref_scores)

    def _prepare(self, names) -> None:
        """Compute and cache the deterministic part for names not seen yet."""
        new = [n for n in dict.fromkeys(names) if n not in self._cache]
        if not new:
            return
        meaning = self.meaning.nearest_many(new) if self.meaning else [(float("nan"), "")] * len(new)
        for name, (m_sim, m_near) in zip(new, meaning):
            in_train = name in self.train_set
            if not name:
                sim, nearest, lp, s_match = 1.0, "", float("-inf"), None
            else:
                sim, nearest = self.nearest_name.nearest(name)
                lp = self.lm.avg_logprob(name)
                s_match = None if in_train else self.sound.match(name)
            novelty = 1.0 - sim
            if not name:
                tier = "empty"
            elif in_train:
                tier = "copy"
            elif s_match:
                tier = "respelling"
            # Tolerance: novelty takes values like 1/6 and the threshold sits on one of them.
            elif novelty < self.novelty_threshold - EPS:
                tier = "near_copy"
            elif self.meaning_threshold is not None and m_sim >= self.meaning_threshold - 1e-6:
                tier = "derived"
            elif name in self.famous:
                tier = "famous"
            else:
                tier = "novel"
            gate_ok = self.min_len <= len(name) <= self.max_len
            pct = self.plausibility_of_logprob(lp) if name else 0.0
            self._cache[name] = dict(
                tier=tier, novel=tier == "novel", novelty=novelty, nearest=nearest,
                sound_match=s_match or "", meaning_sim=m_sim, meaning_nearest=m_near,
                splice=bool(name) and self.splice.is_splice(name),
                in_train=in_train, in_heldout=name in self.heldout_set,
                gate_ok=gate_ok, logprob=lp, plausibility=pct,
                plausible_ngram=gate_ok and pct >= self.plausibility_threshold - EPS)

    def _finish(self, name: str) -> dict:
        """Add the judge-dependent columns (looked up fresh, the cache can grow)."""
        row = dict(self._cache[name])
        needs = row["novel"] and row["gate_ok"]
        rating = self.judge.rating(name) if (self.judge and needs) else None
        row["judge_rating"] = float("nan") if rating is None else float(rating)
        if self.plausibility_mode == "judge":
            row["plausible"] = bool(needs and rating is not None and rating >= self.judge_cut)
        else:
            row["plausible"] = row["plausible_ngram"]
        row["creative"] = row["novel"] and row["plausible"]
        row["creative_ngram"] = row["novel"] and row["plausible_ngram"]
        return row

    def score_name(self, name: str) -> dict:
        self._prepare([name])
        return self._finish(name)

    def needs_judging(self, names: list[str]) -> list[str]:
        """Novel, gate-passing names with no judge rating yet (the judge sees nothing else)."""
        if self.judge is None:
            return []
        names = [clean_sample(n) for n in names]
        self._prepare(names)
        cand = [n for n in dict.fromkeys(names) if self._cache[n]["novel"] and self._cache[n]["gate_ok"]]
        return self.judge.missing(cand)

    # ---- per batch ------------------------------------------------------
    def table(self, names: list[str]) -> pd.DataFrame:
        names = [clean_sample(n) for n in names]
        self._prepare(names)
        if self.plausibility_mode == "judge":
            missing = self.needs_judging(names)
            if missing:
                raise MissingJudgments(missing)
        df = pd.DataFrame([{"name": n, **self._finish(n)} for n in names])
        df["first_occurrence"] = ~df["name"].duplicated()
        return df

    def score(self, names: list[str]) -> dict:
        if not names:
            raise ValueError("no samples to score")
        df = self.table(names)
        n = len(df)
        uniq = df[df["first_occurrence"]]
        novel = df[df["novel"]]
        rated = df["judge_rating"].dropna()
        out = {
            "n": n,
            "creativity_rate": float(uniq["creative"].sum() / n),
            "diversity": float(len(uniq) / n),
            "novelty": float(df["novelty"].mean()),
            "plausibility": float(df["plausibility"].mean()),
            "novel_rate": float(df["novel"].mean()),
            "plausible_rate": float(df["plausible_ngram"].mean()),
            "copy_rate": float(df["in_train"].mean()),
            "heldout_hit_rate": float(df["in_heldout"].mean()),
            "mean_length": float(df["name"].str.len().mean()),
        }
        for tier in TIERS[1:-1]:
            out[f"{tier}_rate"] = float((df["tier"] == tier).mean())
        out.update({
            "gate_fail_rate": float((~df["gate_ok"]).mean()),
            "splice_rate_among_novel": float(novel["splice"].mean()) if len(novel) else 0.0,
            "plausible_rate_given_novel": float(novel["plausible"].mean()) if len(novel) else 0.0,
            "mean_judge_rating": float(rated.mean()) if len(rated) else math.nan,
            "creativity_rate_ngram": float(uniq["creative_ngram"].sum() / n),
            "plausibility_mode": self.plausibility_mode,
        })
        return out


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
