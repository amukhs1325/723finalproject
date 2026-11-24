"""Distill the LLM judge into the local student model, and measure how good the copy is.

    python scripts/train_student.py select          # 1. train a first student, pick on-policy names to label
    python scripts/train_student.py select --fill   #    ... and send them to the judge (needs OPENAI_API_KEY)
    python scripts/train_student.py train           # 2. train all variants, evaluate, save student.pt

Vocabulary used below:
  off-policy names   the 5,000-name pool from scripts/judge_pool.py: real names, n-gram samples,
                     random strings. None comes from a neural model.
  on-policy names    samples from checkpoints of the reference LSTM (cvl/reference_lm.py), i.e. the
                     kind of names the student will actually have to score.
  test checkpoints   five checkpoints (undertrained to badly overfit) whose names are NEVER used
                     for training. Every "on-policy test" number comes from them.

Experiments (results/student/eval.md):
  A  soft vs. hard targets: train on the judge's five probabilities, or only on its top rating?
  B  how many judge labels are needed (learning curve)?
  C  off-policy vs. on-policy: does a student trained without neural samples still work on them?
  D  with a budget of 1,000 extra labels, is it better to label random on-policy names or the
     ones the first student is least sure about (uncertainty sampling)?
The student that ships (student.pt) is trained on every labeled name except the test checkpoints.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cvl.agreement import cohen_kappa  # noqa: E402
from cvl.data import ROOT  # noqa: E402
from cvl.judge import Judge  # noqa: E402
from cvl.scorer import clean_sample, default_scorer  # noqa: E402
from cvl.student import STUDENT_PATH, Student  # noqa: E402
from judge_pool import POOL, study_names  # noqa: E402

OUT = ROOT / "results" / "student"
SAMPLES = ROOT / "results" / "reference" / "samples" / "reference_lstm"
TEST_STEPS = [50, 209, 542, 1405, 12000]      # undertrained ... lowest loss (542) ... badly overfit
BUDGET = 1000
SEED = 723
CUT = 3.0     # provisional yes/no cut for the agreement numbers; the real cut comes from human ratings
P = ["p1", "p2", "p3", "p4", "p5"]


def onpolicy_names() -> pd.DataFrame:
    """Novel, gate-passing sample names from every reference checkpoint, with the step they came from."""
    scorer = default_scorer()
    rows = []
    for f in sorted(SAMPLES.glob("step*_t1_p1_s0.txt")):
        step = int(f.stem.split("_")[0][4:])
        names = list(dict.fromkeys(clean_sample(x) for x in f.read_text().splitlines()))
        scorer._prepare(names)
        rows += [{"name": n, "step": step} for n in names
                 if n and scorer.score_name(n)["novel"] and scorer.score_name(n)["gate_ok"]]
    if not rows:
        sys.exit(f"no samples in {SAMPLES}; run scripts/train_reference_lstm.py and "
                 "scripts/eval_checkpoint.py checkpoints/reference_lstm --out results/reference/checkpoints.csv")
    return pd.DataFrame(rows)


def offpolicy_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool = pd.read_csv(POOL).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return pool[:4000], pool[4000:4500], pool[4500:]


def labels(judge: Judge, names: list[str], hard: bool = False) -> np.ndarray:
    y = np.array([judge.probs(n) for n in names])
    if hard:                                   # keep only the judge's single most likely rating
        y = np.eye(5)[y.argmax(1)]
    return y


def select(fill: bool) -> None:
    judge = Judge()
    tr, va, _ = offpolicy_split()
    OUT.mkdir(parents=True, exist_ok=True)
    s0 = Student.fit(tr["name"].tolist(), labels(judge, tr["name"].tolist()),
                     va["name"].tolist(), labels(judge, va["name"].tolist()))
    s0.save(OUT / "s0.pt")
    on = onpolicy_names()
    test = on[on["step"].isin(TEST_STEPS)]
    blocked = set(test["name"]) | set(pd.read_csv(POOL)["name"]) | study_names()
    cand = sorted(set(on[~on["step"].isin(TEST_STEPS)]["name"]) - blocked)
    rng = np.random.default_rng(SEED)
    random_pick = list(rng.choice(cand, BUDGET, replace=False))
    unc = s0.uncertainty(cand)
    active_pick = [cand[i] for i in np.argsort(-unc)[:BUDGET]]
    test.to_csv(OUT / "onpolicy_test.csv", index=False)
    pd.DataFrame({"name": random_pick}).to_csv(OUT / "onpolicy_random.csv", index=False)
    pd.DataFrame({"name": active_pick, "uncertainty": np.sort(unc)[::-1][:BUDGET]}).to_csv(
        OUT / "onpolicy_active.csv", index=False)
    need = judge.missing(list(test["name"]) + random_pick + active_pick)
    print(f"on-policy: {on['name'].nunique()} novel names from {on['step'].nunique()} checkpoints; "
          f"test set {test['name'].nunique()} names from steps {TEST_STEPS}; {len(cand)} candidates; "
          f"random and uncertainty picks share {len(set(random_pick) & set(active_pick))} names")
    print(f"{len(need)} names to judge")
    if need and fill:
        judge.fill(need)


def metrics(student: Student, judge: Judge, names: list[str]) -> dict:
    r = student.rating(names)
    j = np.array([judge.rating(n) for n in names])
    return {"n": len(names),
            "spearman": float(pd.Series(r).corr(pd.Series(j), method="spearman")),
            "mae": float(np.abs(r - j).mean()),
            "agreement": float(((r >= CUT) == (j >= CUT)).mean()),
            "kappa": float(cohen_kappa(list(r >= CUT), list(j >= CUT))),
            "mean_student": float(r.mean()), "mean_judge": float(j.mean())}


def per_checkpoint(student: Student, judge: Judge, test: pd.DataFrame) -> list[dict]:
    """Share of each test checkpoint's novel names rated plausible, by the judge and by the student."""
    rows = []
    for step, g in test.groupby("step"):
        names = g["name"].tolist()
        j = np.array([judge.rating(n) for n in names]) >= CUT
        s = student.rating(names) >= CUT
        rows.append({"step": int(step), "n": len(names), "judge": float(j.mean()), "student": float(s.mean())})
    return rows


def train() -> None:
    judge = Judge()
    tr, va, te = offpolicy_split()
    test = pd.read_csv(OUT / "onpolicy_test.csv")
    rnd = pd.read_csv(OUT / "onpolicy_random.csv")["name"].tolist()
    act = pd.read_csv(OUT / "onpolicy_active.csv")["name"].tolist()
    missing = judge.missing(list(test["name"]) + rnd + act)
    if missing:
        sys.exit(f"{len(missing)} on-policy names are not judged yet; run: train_student.py select --fill")
    off_te = te["name"].tolist()
    on_te = sorted(set(test["name"]))
    trn, van = tr["name"].tolist(), va["name"].tolist()

    def fit(train_names, val_names, hard=False, k=5):
        return Student.fit(train_names, labels(judge, train_names, hard), val_names, labels(judge, val_names))

    def with_extra(extra):                    # 90% of the extra names to train, 10% to validation
        cut = int(0.9 * len(extra))
        return trn + extra[:cut], van + extra[cut:]

    results, students = {}, {}
    plans = {
        "off-policy, soft targets": (trn, van, False),
        "off-policy, hard targets": (trn, van, True),
        "+1,000 random on-policy": (*with_extra(rnd), False),
        "+1,000 least-certain on-policy": (*with_extra(act), False),
        "everything (shipped)": (*with_extra(list(dict.fromkeys(rnd + act))), False),
    }
    for label, (a, b, hard) in plans.items():
        s = fit(a, b, hard)
        students[label] = s
        results[label] = {"train_names": len(a), "off_policy_test": metrics(s, judge, off_te),
                          "on_policy_test": metrics(s, judge, on_te),
                          "per_checkpoint": per_checkpoint(s, judge, test)}
        o, n = results[label]["off_policy_test"], results[label]["on_policy_test"]
        print(f"{label:32s} off-policy Spearman {o['spearman']:.3f} agree {o['agreement']:.3f} | "
              f"on-policy Spearman {n['spearman']:.3f} agree {n['agreement']:.3f}", flush=True)

    curve = []
    for size in (500, 1000, 2000, 4000):
        s = students["off-policy, soft targets"] if size == 4000 else fit(trn[:size], van)
        curve.append({"labels": size, **{k: v for k, v in metrics(s, judge, off_te).items()
                                         if k in ("spearman", "mae", "agreement")}})
        print(f"learning curve {size:>5} labels: Spearman {curve[-1]['spearman']:.3f}", flush=True)

    final = students["everything (shipped)"]
    final.meta = {"judge_model": judge.model, "judge_prompt_hash": judge.prompt_hash,
                  "train_names": results["everything (shipped)"]["train_names"],
                  "on_policy_test": results["everything (shipped)"]["on_policy_test"]}
    final.save(STUDENT_PATH)
    report = {"cut_for_agreement": CUT, "test_steps": TEST_STEPS, "variants": results, "learning_curve": curve}
    (OUT / "eval.json").write_text(json.dumps(report, indent=2))
    write_markdown(report)
    print(f"\nsaved {STUDENT_PATH.name} and {OUT.relative_to(ROOT)}/eval.md")


def write_markdown(rep: dict) -> None:
    lines = ["# Student vs. judge", "",
             "Generated by `scripts/train_student.py train`. The judge is the reference; every number is "
             "the student compared with the judge on names the student never trained on.",
             f"Yes/no agreement uses a provisional cut of rating >= {rep['cut_for_agreement']}. The real cut "
             "comes from the human ratings.", "",
             "## Variants", "",
             "| Student trained on | Labels | Off-policy test: Spearman | agreement | On-policy test: Spearman | MAE | agreement | kappa |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for label, r in rep["variants"].items():
        o, n = r["off_policy_test"], r["on_policy_test"]
        lines.append(f"| {label} | {r['train_names']:,} | {o['spearman']:.3f} | {o['agreement']:.3f} | "
                     f"{n['spearman']:.3f} | {n['mae']:.3f} | {n['agreement']:.3f} | {n['kappa']:.3f} |")
    lines += ["", "## Does the student reproduce the judge per checkpoint?", "",
              "Share of each test checkpoint's novel names rated plausible. This is the number that feeds "
              "the creativity rate, so the gap in the last column is what matters.", "",
              "| Step | Novel names | Judge | Shipped student | Gap | Off-policy-only student | Gap |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    ship = rep["variants"]["everything (shipped)"]["per_checkpoint"]
    off = rep["variants"]["off-policy, soft targets"]["per_checkpoint"]
    for a, b in zip(ship, off):
        lines.append(f"| {a['step']} | {a['n']} | {a['judge']:.3f} | {a['student']:.3f} | "
                     f"{a['student'] - a['judge']:+.3f} | {b['student']:.3f} | {b['student'] - b['judge']:+.3f} |")
    lines += ["", "## Learning curve (off-policy test)", "", "| Judge labels | Spearman | MAE | agreement |",
              "| --- | --- | --- | --- |"]
    lines += [f"| {c['labels']:,} | {c['spearman']:.3f} | {c['mae']:.3f} | {c['agreement']:.3f} |"
              for c in rep["learning_curve"]]
    (OUT / "eval.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--fill", action="store_true")
    sub.add_parser("train")
    args = ap.parse_args()
    select(args.fill) if args.cmd == "select" else train()
