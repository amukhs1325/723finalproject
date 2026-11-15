"""Load, clean and split the D&D character name dataset.

Unit of data: one name per line, lowercased, characters a-z plus ' and -.
By default the unit is the character's given name (first word, titles removed).
Use unit="full" to keep the full name (spaces allowed).
"""
from __future__ import annotations

import csv
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "DnD_characters_May2018.txt"
PROCESSED_DIR = ROOT / "data" / "processed"

# Titles and filler words people typed before the real name.
TITLES = {
    "sir", "lord", "lady", "dame", "the", "captain", "capt", "king", "queen",
    "prince", "princess", "brother", "sister", "father", "mother", "mr", "mrs",
    "ms", "miss", "dr", "doctor", "fr", "master", "mistress", "madame", "saint",
    "st", "professor", "prof", "baron", "baroness", "count", "countess", "duke",
    "duchess", "general", "commander", "sergeant", "sgt", "old", "young", "big",
    "little", "lil", "uncle", "aunt", "grandpa", "granny", "elder",
}

MIN_LEN, MAX_LEN = 2, 15
EOS = "\n"  # end-of-name marker for models; never part of a name


def to_ascii(text: str) -> str:
    """Turn accented letters into plain ones and lowercase."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()


def clean_name(raw: str, unit: str = "given") -> str | None:
    """Return a cleaned name, or None if the entry should be dropped."""
    text = to_ascii(raw).strip()
    if unit == "given":
        words = [w.strip(".,!?;:\"()") for w in text.split()]
        words = [w for w in words if w and w not in TITLES]
        if not words:
            return None
        name = words[0].strip("'-")
        pattern = r"[a-z][a-z'\-]*[a-z]|[a-z]"
    elif unit == "full":
        name = re.sub(r"\s+", " ", re.sub(r"[.,!?;:\"()]", "", text)).strip()
        pattern = r"[a-z][a-z' \-]*[a-z]|[a-z]"
        max_len = 30
        if not re.fullmatch(pattern, name) or not (MIN_LEN <= len(name) <= max_len):
            return None
        return name
    else:
        raise ValueError(f"unknown unit: {unit}")
    if not re.fullmatch(pattern, name):
        return None
    if not (MIN_LEN <= len(name) <= MAX_LEN):
        return None
    return name


def read_raw(path: Path = RAW_FILE) -> list[str]:
    """Name column of the raw CSV (header skipped)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.reader(f))
    return [r[0] for r in rows[1:] if r and r[0].strip()]


def prepare(unit: str = "given", seed: int = 723, out_dir: Path = PROCESSED_DIR,
            fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> dict:
    """Clean, dedupe and split the dataset; write train/val/test.txt and meta.json."""
    raw = read_raw()
    cleaned = [clean_name(r, unit) for r in raw]
    kept = [c for c in cleaned if c]
    counts = Counter(kept)
    unique = sorted(counts)

    rng = random.Random(seed)
    order = unique[:]
    rng.shuffle(order)
    n = len(order)
    n_train = int(fractions[0] * n)
    n_val = int(fractions[1] * n)
    splits = {
        "train": sorted(order[:n_train]),
        "val": sorted(order[n_train:n_train + n_val]),
        "test": sorted(order[n_train + n_val:]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        (out_dir / f"{split}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")

    vocab = sorted(set("".join(unique)))
    lengths = [len(x) for x in unique]
    meta = {
        "source": RAW_FILE.name,
        "unit": unit,
        "seed": seed,
        "raw_entries": len(raw),
        "kept_entries": len(kept),
        "dropped_entries": len(raw) - len(kept),
        "unique_names": n,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "vocab": vocab,
        "eos": EOS,
        "length": {
            "min": min(lengths), "max": max(lengths),
            "mean": round(sum(lengths) / n, 2),
        },
        "most_common": counts.most_common(25),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_split(split: str, data_dir: Path = PROCESSED_DIR) -> list[str]:
    """Names in a split: 'train', 'val' or 'test'."""
    path = data_dir / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/prepare_data.py first")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_meta(data_dir: Path = PROCESSED_DIR) -> dict:
    return json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
