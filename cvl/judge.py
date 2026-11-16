"""LLM judge for plausibility, with an on-disk cache.

The judge answers one question per name, on the same 1 to 5 scale the human
raters use (RUBRIC below is shown to both). It only ever sees names that passed
novelty and the length gate. Ratings are stored in judge_cache.csv and committed,
so scoring is a table lookup: nobody needs an API key to reproduce our numbers,
and training code never calls the API.

One name per call, on purpose. In a pilot of 63 names, rating 25 names per call
gave a different rating for 30% of names when their batch-mates changed. That
would let the quality of a checkpoint's other samples leak into each rating.
Alone, every name got the same rating on a repeated call (63 of 63).

    judge = Judge()                       # model and prompt are pinned here
    judge.rating("torvus")                # int 1..5, or None if not judged yet
    judge.fill(["torvus", "elmar"])       # calls the API for names not in the cache

API key: environment variable OPENAI_API_KEY (only needed for fill()).
"""
from __future__ import annotations

import csv
import hashlib
import json
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
BATCH_SIZE = 1                            # see the note above before raising this
WORKERS = 8                               # parallel API calls in fill()
SEED = 723

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

Return the name exactly as given, with its rating."""

PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12]

_SCHEMA = {
    "name": "ratings", "strict": True,
    "schema": {
        "type": "object", "additionalProperties": False, "required": ["ratings"],
        "properties": {"ratings": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "rating"],
            "properties": {"name": {"type": "string"},
                           "rating": {"type": "integer", "enum": [1, 2, 3, 4, 5]}}}}},
    },
}


class MissingJudgments(RuntimeError):
    """Raised when a score needs judge ratings that are not in the cache yet."""

    def __init__(self, names: list[str]):
        self.names = names
        super().__init__(f"{len(names)} names have no judge rating yet (e.g. {names[:5]}); "
                         "run scripts/eval_checkpoint.py with --fill-judge, or Judge().fill(names)")


class Judge:
    def __init__(self, model: str = JUDGE_MODEL, cache_path: Path = CACHE_PATH,
                 batch_size: int = BATCH_SIZE, workers: int = WORKERS):
        self.model, self.cache_path, self.batch_size = model, Path(cache_path), batch_size
        self.workers = workers
        self.prompt_hash = PROMPT_HASH
        self._ratings: dict[str, int] = {}
        self._other_rows: list[dict] = []   # ratings from other models or prompts, kept on disk
        if self.cache_path.exists():
            with open(self.cache_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["model"] == model and row["prompt_hash"] == self.prompt_hash:
                        self._ratings[row["name"]] = int(row["rating"])
                    else:
                        self._other_rows.append(row)

    # ---- lookups (no network) ---------------------------------------------
    def rating(self, name: str) -> int | None:
        return self._ratings.get(name)

    def missing(self, names) -> list[str]:
        return [n for n in dict.fromkeys(names) if n and n not in self._ratings]

    def __len__(self) -> int:
        return len(self._ratings)

    # ---- filling the cache (network) ----------------------------------------
    def fill(self, names, verbose: bool = True) -> int:
        """Judge every name not yet in the cache. Returns the number of API calls made."""
        todo = sorted(self.missing(names))
        random.Random(SEED).shuffle(todo)
        batches = [todo[i:i + self.batch_size] for i in range(0, len(todo), self.batch_size)]
        calls = done = 0

        def run(batch: list[str]) -> tuple[dict[str, int], int]:
            got, n_calls = self._judge_batch(batch), 1
            for n in [x for x in batch if x not in got]:   # skipped or altered name: ask again alone
                got.update(self._judge_batch([n]))
                n_calls += 1
            still = [x for x in batch if x not in got]
            if still:
                raise RuntimeError(f"judge returned no rating for {still}")
            return got, n_calls

        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for i, (got, n_calls) in enumerate(pool.map(run, batches), 1):
                    self._ratings.update(got)
                    calls += n_calls
                    done += len(got)
                    if i % 200 == 0:
                        self._save()
                        if verbose:
                            print(f"judged {done}/{len(todo)}", flush=True)
        finally:
            self._save()                       # keep whatever finished, even after an error
        if verbose and todo:
            print(f"judged {done}/{len(todo)} names in {calls} calls", flush=True)
        return calls

    def _judge_batch(self, batch: list[str]) -> dict[str, int]:
        body = {
            "model": self.model, "temperature": 0, "seed": SEED,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": "\n".join(batch)}],
            "response_format": {"type": "json_schema", "json_schema": _SCHEMA},
        }
        data = self._post(body)
        ratings = json.loads(data["choices"][0]["message"]["content"])["ratings"]
        if len(batch) == 1 and len(ratings) == 1:      # one name in, one rating out
            return {batch[0]: int(ratings[0]["rating"])}
        wanted = set(batch)
        return {r["name"]: int(r["rating"]) for r in ratings if r["name"] in wanted}

    @staticmethod
    def _post(body: dict) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set; it is only needed to judge new names")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"judge API error {e.code}: {e.read().decode()[:300]}") from e
            except urllib.error.URLError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")

    def _save(self) -> None:
        rows = self._other_rows + [
            {"name": n, "rating": r, "model": self.model, "prompt_hash": self.prompt_hash}
            for n, r in self._ratings.items()]
        rows.sort(key=lambda r: (r["model"], r["prompt_hash"], r["name"]))
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["name", "rating", "model", "prompt_hash"])
            w.writeheader()
            w.writerows(rows)
        tmp.replace(self.cache_path)
