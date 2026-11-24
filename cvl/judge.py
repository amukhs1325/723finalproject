"""LLM judge for plausibility, with an on-disk cache.

The judge answers one question per name, on the same 1 to 5 scale the human
raters use (RUBRIC below is shown to both). It only ever sees names that passed
novelty and the length gate. Ratings are stored in judge_cache.csv and committed,
so scoring is a table lookup: nobody needs an API key to reproduce our numbers,
and training code never calls the API.

    judge = Judge()                       # model and prompt are pinned here
    judge.rating("torvus")                # float 1..5, or None if not judged yet
    judge.probs("torvus")                 # [p1, p2, p3, p4, p5]
    judge.fill(["torvus", "elmar"])       # calls the API for names not in the cache

Soft ratings. The judge answers with one digit and we read the model's probability
for each of the five digits (token log-probabilities). The rating is the expected
value, so a name the judge is torn on between 3 and 4 gets 3.4 instead of a coin
flip. The five probabilities are also what the local student model is trained on
(cvl/student.py).

One name per call, on purpose. In a pilot of 63 names, rating 25 names per call
gave a different rating for 30% of names when their batch-mates changed. That
would let the quality of a checkpoint's other samples leak into each rating.

API key: environment variable OPENAI_API_KEY (only needed for fill()).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = ROOT / "judge_cache.csv"

JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"   # dated snapshot, temperature 0
WORKERS = 12                              # parallel API calls in fill()
SEED = 723
FIELDS = ["name", "expected", "p1", "p2", "p3", "p4", "p5", "model", "prompt_hash"]

QUESTION = ("Could this be the given name of a character in a fantasy role-playing game "
            "such as Dungeons & Dragons?")

# Shown word for word to human raters (scripts/human_tasks.py) and to the judge.
RUBRIC = f"""{QUESTION}

5 = Yes. Reads as a finished name. You would not look twice at it on a character sheet.
4 = Probably. Sounds like a name, with something slightly off.
3 = Borderline. Pronounceable, but it is unclear whether it is a name.
2 = Probably not. Awkward letter sequences, or it reads as a fragment.
1 = No. Unpronounceable or random letters.

Names are lowercase and may contain apostrophes or hyphens. Judge only whether the name
works as a name. Do not reward a name for being familiar, and do not penalize it for
being unusual."""

SYSTEM_PROMPT = f"""You are rating a candidate character name. The user message is the name. Answer this
question about it:

{RUBRIC}

Answer with a single digit from 1 to 5 and nothing else."""

PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]


def on_disk(probs: list[float]) -> list[float]:
    """Probabilities exactly as the cache file stores them (5 decimals).

    A rating is used the same way whether it was fetched a second ago or read back from
    judge_cache.csv. Without this, a name rated 2.99996 counts as below a cut of 3 in the
    run that fetched it and as 3.0000 in every later run.
    """
    return [float(f"{p:.5f}") for p in probs]


def expected_rating(probs: list[float]) -> float:
    return sum((i + 1) * p for i, p in enumerate(probs))


class MissingJudgments(RuntimeError):
    """Raised when a score needs judge ratings that are not in the cache yet."""

    def __init__(self, names: list[str]):
        self.names = names
        super().__init__(f"{len(names)} names have no judge rating yet (e.g. {names[:5]}); "
                         "run scripts/eval_checkpoint.py with --fill-judge, or Judge().fill(names)")


class Judge:
    def __init__(self, model: str = JUDGE_MODEL, cache_path: Path = CACHE_PATH,
                 workers: int = WORKERS):
        self.model, self.cache_path, self.workers = model, Path(cache_path), workers
        self.prompt_hash = PROMPT_HASH
        self._probs: dict[str, list[float]] = {}
        self._other_rows: list[dict] = []   # ratings from other models or prompts, kept on disk
        if self.cache_path.exists():
            with open(self.cache_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["model"] == model and row["prompt_hash"] == self.prompt_hash:
                        self._probs[row["name"]] = [float(row[f"p{k}"]) for k in range(1, 6)]
                    else:
                        self._other_rows.append(row)

    # ---- lookups (no network) ---------------------------------------------
    def probs(self, name: str) -> list[float] | None:
        return self._probs.get(name)

    def rating(self, name: str) -> float | None:
        """Expected rating, 1.0 to 5.0."""
        p = self._probs.get(name)
        return None if p is None else expected_rating(p)

    def missing(self, names) -> list[str]:
        return [n for n in dict.fromkeys(names) if n and n not in self._probs]

    def __len__(self) -> int:
        return len(self._probs)

    def put(self, name: str, probs: list[float]) -> None:
        """Store a rating by hand (tests, or importing ratings made elsewhere)."""
        total = sum(probs)
        self._probs[name] = on_disk([p / total for p in probs])

    # ---- filling the cache (network) ----------------------------------------
    def fill(self, names, verbose: bool = True) -> int:
        """Judge every name not yet in the cache. Returns the number of API calls made."""
        todo = sorted(self.missing(names))
        random.Random(SEED).shuffle(todo)
        done = 0
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for name, probs in zip(todo, pool.map(self._judge_one, todo)):
                    self._probs[name] = on_disk(probs)
                    done += 1
                    if done % 250 == 0:
                        self.save()
                        if verbose:
                            print(f"judged {done}/{len(todo)}", flush=True)
        finally:
            self.save()                        # keep whatever finished, even after an error
        if verbose and todo:
            print(f"judged {done}/{len(todo)} names", flush=True)
        return done

    def _judge_one(self, name: str) -> list[float]:
        data = self._post({
            "model": self.model, "temperature": 0, "seed": SEED, "max_tokens": 1,
            "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": name}]})
        top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        mass = [0.0] * 5
        for t in top:
            tok = t["token"].strip()
            if tok in ("1", "2", "3", "4", "5"):
                mass[int(tok) - 1] += math.exp(t["logprob"])
        total = sum(mass)
        if total < 0.5:                        # the model did not answer with a digit
            raise RuntimeError(f"judge gave no usable rating for {name!r}")
        return [m / total for m in mass]

    @staticmethod
    def _post(body: dict) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; it is only needed to judge new names")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(7):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode()[:300]
                if e.code in (429, 500, 502, 503, 504) and "credit" not in detail and attempt < 6:
                    time.sleep(min(30, 2 ** attempt) * (0.5 + random.random()))
                    continue
                raise RuntimeError(f"judge API error {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == 6:
                    raise
                time.sleep(min(30, 2 ** attempt))
        raise RuntimeError("unreachable")

    def save(self) -> None:
        rows = list(self._other_rows)
        for n, p in self._probs.items():
            rows.append({"name": n, "expected": f"{expected_rating(p):.4f}",
                         **{f"p{k + 1}": f"{p[k]:.5f}" for k in range(5)},
                         "model": self.model, "prompt_hash": self.prompt_hash})
        rows.sort(key=lambda r: (r["model"], r["prompt_hash"], r["name"]))
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(self.cache_path)
