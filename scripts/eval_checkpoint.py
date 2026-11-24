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
listed, and --fill-judge sends them (needs OPENAI_API_KEY).

How many names the judge sees per checkpoint is --judge-sample (default 200). The student
model (student.pt) scores every novel name for free, the judge rates a random 200, and the
two are combined into an unbiased estimate with a standard error (cvl/estimate.py). Use
--judge-sample all to judge every novel name instead. Columns written:
    creativity_rate_judge   the judge-based creativity rate (estimate, or exact with "all")
    creativity_rate_se      its standard error (0 when exact)
    creativity_rate_student the student alone, uncorrected. Biased; shown for comparison only.
    creativity_rate_ngram   the 3-gram version, as before
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.data import ROOT, load_split  # noqa: E402
from cvl.sampling import load_checkpoint, sample  # noqa: E402
from cvl.estimate import PROVISIONAL_CUT, plausible_share  # noqa: E402
from cvl.scorer import default_scorer  # noqa: E402
from cvl.student import STUDENT_PATH, Student  # noqa: E402

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
    ap.add_argument("--judge-sample", default="200",
                    help="novel names per checkpoint to judge: a number, or 'all' (default 200)")
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
        batches.append(dict(ck=ck, run=run, names=names, val_loss=val_loss, params=meta.get("params"),
                            step=meta.get("step", step_of(ck))))
        print(f"sampled {rel(ck)}", flush=True)

    # Phase 2: decide which names the judge needs to see, and send the ones it has not seen.
    m = None if args.judge_sample == "all" else int(args.judge_sample)
    student = Student.load() if STUDENT_PATH.exists() else None
    cut = scorer.judge_cut if scorer.judge_cut is not None else PROVISIONAL_CUT

    def novel_names(names):
        tab = scorer.table(names) if scorer.plausibility_mode != "judge" else None
        if tab is None:                          # judge mode: avoid the table's completeness check
            scorer._prepare([n for n in names])
            return [n for n in dict.fromkeys(names) if n and scorer._cache[n]["novel"] and scorer._cache[n]["gate_ok"]]
        keep = tab[tab["novel"] & tab["gate_ok"] & tab["first_occurrence"]]
        return keep["name"].tolist()

    from cvl.scorer import clean_sample
    for b in batches:
        b["clean"] = [clean_sample(x) for x in b["names"]]
        b["novel"] = novel_names(b["clean"])
        b["share"] = plausible_share(b["novel"], scorer.judge, student, cut, m)
    unseen = list(dict.fromkeys(x for b in batches for x in b["share"]["missing"]))
    if unseen and args.fill_judge:
        scorer.judge.fill(unseen)
        for b in batches:
            b["share"] = plausible_share(b["novel"], scorer.judge, student, cut, m)
    elif unseen:
        print(f"{len(unseen)} names need a judge rating; pass --fill-judge to send them")

    # Phase 3: score and write.
    rows = []
    for b in batches:
        r = scorer.score(b["names"]) if scorer.plausibility_mode != "judge" or not b["share"]["missing"] else None
        if r is None:
            continue
        sh, n = b["share"], len(b["names"])
        scale = sh["n"] / n                      # unique novel gate-passing names per sample
        r.update({
            "creativity_rate_judge": None if sh["estimate"] is None else sh["estimate"] * scale,
            "creativity_rate_se": None if sh["se"] is None else sh["se"] * scale,
            "creativity_rate_student": sh["student_only"] * scale,
            "judged_names": sh["judged"], "judge_method": sh["method"], "judge_cut_used": cut,
        })
        rows.append({"run": b["run"], "checkpoint": rel(b["ck"]), "step": b["step"],
                     "val_loss": b["val_loss"], "params": b["params"], "temperature": args.temperature,
                     "top_p": args.top_p, "seed": args.seed, **r})
    new = pd.DataFrame(rows)
    if args.out.exists():
        old = pd.read_csv(args.out)
        merged = old.merge(new[KEY], on=KEY, how="left", indicator=True)
        new = pd.concat([old[(merged["_merge"] == "left_only").to_numpy()], new], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["run", "step", "temperature", "top_p", "seed"], na_position="first").to_csv(args.out, index=False)
    show = ["run", "step", "val_loss", "creativity_rate_judge", "creativity_rate_se", "creativity_rate_student",
            "creativity_rate_ngram", "novel_rate", "copy_rate", "judged_names"]
    print()
    print(pd.DataFrame(rows)[show].round(3).to_string(index=False))
    print(f"\nwrote {len(rows)} rows to {rel(args.out)}")


if __name__ == "__main__":
    main()
