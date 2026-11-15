"""Clean and split the D&D names.

    python scripts/prepare_data.py              # given names (default)
    python scripts/prepare_data.py --unit full  # full names
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import prepare  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", choices=["given", "full"], default="given")
    ap.add_argument("--seed", type=int, default=723)
    args = ap.parse_args()
    meta = prepare(unit=args.unit, seed=args.seed)
    print(f"raw entries:    {meta['raw_entries']}")
    print(f"kept entries:   {meta['kept_entries']}  (dropped {meta['dropped_entries']})")
    print(f"unique names:   {meta['unique_names']}")
    print(f"splits:         {meta['split_sizes']}")
    print(f"vocab ({len(meta['vocab'])}): {''.join(meta['vocab'])!r}")
    print(f"length:         {meta['length']}")
    print(f"most common:    {meta['most_common'][:10]}")
