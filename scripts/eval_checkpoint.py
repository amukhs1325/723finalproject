"""Score checkpoints: sample names, score them, write one row per checkpoint.

    python scripts/eval_checkpoint.py checkpoints/lstm_medium_s0/            # every step*.pt in a run
    python scripts/eval_checkpoint.py checkpoints/bigram.json                # or single files
    python scripts/eval_checkpoint.py <run dir> --fill-judge                 # also judge new names

Each checkpoint: 1,000 names at temperature 1, top-p 1, seed 0 (override with flags for
the decoding sweep). Rows go to results/checkpoints.csv, keyed by checkpoint and sampling
settings, so rerunning replaces a row instead of duplicating it. Samples are saved under
results/samples/ for error analysis.

Any checkpoint that cvl.sampling.load_checkpoint can open works. Validation loss is read
from the model if it carries one (`model.meta["val_loss"]`, nats per symbol with the end
marker counted), or computed for n-gram models.

Judge ratings: scoring never calls the API by itself. Names the judge has not seen are
listed, and --fill-judge sends them (needs OPENAI_API_KEY). In "ngram" mode missing
ratings do not block scoring; in "judge" mode they do.
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, load_split  # noqa: E402
from cvl.sampling import load_checkpoint, sample  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402

RESULTS = ROOT / "results" / "checkpoints.csv"
KEY = ["checkpoint", "n", "temperature", "top_p", "seed"]


def find_checkpoints(paths: list[str]) -> list[Path]:
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            files = [f for f in p.iterdir() if f.suffix in (".pt", ".json")]
            found += sorted(files, key=lambda f: (step_of(f) if step_of(f) is not None else -1, f.name))
        else:
            found.append(p)
    return found


def step_of(path: Path) -> int | None:
    m = re.search(r"step(\d+)", path.stem)
    return int(m.group(1)) if m else None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="checkpoint files or run directories")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=RESULTS)
    ap.add_argument("--fill-judge", action="store_true", help="send unseen names to the judge")
    args = ap.parse_args()

    scorer = default_scorer()
    val = load_split("val")
    ckpts = find_checkpoints(args.paths)
    if not ckpts:
        sys.exit("no checkpoints found")
    sample_dir = args.out.parent / "samples"

    # Phase 1: sample everything (no API calls).
    batches = []
    for ck in ckpts:
        model = load_checkpoint(ck)
        names = sample(model, n=args.n, temperature=args.temperature, top_p=args.top_p, seed=args.seed)
        meta = getattr(model, "meta", {}) or {}
        val_loss = meta.get("val_loss")
        if val_loss is None and hasattr(model, "val_loss"):
            val_loss = model.val_loss(val)
        run = ck.parent.name if step_of(ck) is not None else ck.stem
        out = sample_dir / run / f"{ck.stem}_t{args.temperature:g}_p{args.top_p:g}_s{args.seed}.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(names) + "\n", encoding="utf-8")
        batches.append(dict(ck=ck, run=run, names=names, val_loss=val_loss,
                            step=meta.get("step", step_of(ck))))
        print(f"sampled {rel(ck)}", flush=True)

    # Phase 2: judge names nobody has judged yet (only novel, gate-passing names are sent).
    unseen = scorer.needs_judging([n for b in batches for n in b["names"]])
    if unseen and args.fill_judge:
        scorer.judge.fill(unseen)
    elif unseen:
        print(f"{len(unseen)} novel names have no judge rating yet; pass --fill-judge to send them")

    # Phase 3: score and write.
    rows = []
    for b in batches:
        r = scorer.score(b["names"])
        rows.append({"run": b["run"], "checkpoint": rel(b["ck"]), "step": b["step"],
                     "val_loss": b["val_loss"], "temperature": args.temperature,
                     "top_p": args.top_p, "seed": args.seed, **r})
    new = pd.DataFrame(rows)
    if args.out.exists():
        old = pd.read_csv(args.out)
        merged = old.merge(new[KEY], on=KEY, how="left", indicator=True)
        new = pd.concat([old[(merged["_merge"] == "left_only").to_numpy()], new], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["run", "step", "temperature", "top_p", "seed"], na_position="first").to_csv(args.out, index=False)
    show = ["run", "step", "val_loss", "creativity_rate", "creativity_rate_ngram", "novel_rate",
            "copy_rate", "respelling_rate", "near_copy_rate", "derived_rate", "gate_fail_rate", "diversity"]
    print()
    print(pd.DataFrame(rows)[show].round(3).to_string(index=False))
    print(f"\nwrote {len(rows)} rows to {rel(args.out)}")


if __name__ == "__main__":
    main()
