"""Human grounding for the creativity score (docs/evaluation_plan.md, section 3).

Make the two rating sheets (deterministic, needs no trained model):
    python scripts/human_tasks.py make

Each of us rates in the rating app (same items as the sheets, one at a time, in a random
order per rater), then sends the block of text it shows at the end. Turn that into files:
    python scripts/human_tasks.py import ratings_from_vraj.txt     # or paste on stdin
Filling the CSV sheets by hand still works: save them next to the sheets as
rater_<name>_A.csv and rater_<name>_B.csv, without looking at the key files. Then:
    python scripts/human_tasks.py analyze                 # agreement, judge vs. humans
    python scripts/human_tasks.py analyze --fill-judge    # also judge any Task A names not cached
    python scripts/human_tasks.py analyze --freeze        # write judge_cut into scorer_config.json

Task A (120 names): rate plausibility 1 to 5 with the judge's exact rubric. Half the
names calibrate the judge's cut, the other half test it.
Task B (60 pairs): is the left name just a variant of the right? Checks that the
novelty views flag what a person would call the same name.
"""
import argparse
import json
import random
import string
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cvl.agreement import cohen_kappa, krippendorff_alpha  # noqa: E402
from cvl.data import ROOT, clean_name, load_split  # noqa: E402
from cvl.judge import RUBRIC  # noqa: E402
from cvl.ngram import CharNGram  # noqa: E402
from cvl.sampling import sample  # noqa: E402
from cvl.scorer import CONFIG_PATH, default_scorer  # noqa: E402

OUT = ROOT / "outputs" / "human_study"
RNN_FILE = ROOT / "data" / "raw" / "DnD_character_raw_output.txt"
SEED = 723

QUOTA_A = {"real held-out": 30, "char-RNN": 25, "5-gram": 15, "3-gram": 15, "bigram": 15,
           "shuffled letters": 10, "random strings": 10}
QUOTA_B = {"near_copy": 10, "respelling": 12, "derived": 12, "under spelling": 8,
           "under meaning": 8, "distant": 10}
QUESTION_B = ("Is the LEFT name basically the RIGHT name: a respelling, or the same name with "
              "a small addition or change? 1 = yes, 0 = no.")
APP_URL = "https://bloohunnits.github.io/723finalproject/"   # plain web page, no account needed


def one_edit(name: str, rng: random.Random) -> str:
    letters = string.ascii_lowercase
    i = rng.randrange(len(name))
    op = rng.choice(["sub", "ins", "del"]) if len(name) > 3 else "sub"
    if op == "sub":
        return name[:i] + rng.choice(letters.replace(name[i], "")) + name[i + 1:]
    if op == "ins":
        return name[:i] + rng.choice(letters) + name[i:]
    return name[:i] + name[i + 1:]


def pools(rng: random.Random) -> dict[str, list[str]]:
    """Candidate names by source. Sources span the plausibility range on purpose."""
    train, test = load_split("train"), load_split("test")
    blocks = RNN_FILE.read_text(encoding="utf-8", errors="replace").split("\n\n")
    rnn = [clean_name(b.strip().splitlines()[0]) for b in blocks if b.strip()]
    lengths = [len(n) for n in train]
    out = {
        "real held-out": list(test),
        "char-RNN": [n for n in rnn if n],
        "shuffled letters": ["".join(rng.sample(n, len(n))) for n in rng.sample(test, 600)],
        "random strings": ["".join(rng.choices(string.ascii_lowercase, k=rng.choice(lengths)))
                           for _ in range(600)],
        "one-letter edits": [one_edit(n, rng) for n in rng.sample(train, 1500)],
    }
    for label, order in (("bigram", 2), ("3-gram", 3), ("5-gram", 5)):
        out[label] = sample(CharNGram(order).fit(train), n=1500, seed=SEED + order)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def make() -> None:
    rng = random.Random(SEED)
    scorer = default_scorer()
    src = pools(rng)

    # ---- Task A: novel, gate-passing names only (that is all the judge ever sees) ----
    picked, used = [], set()
    for source, quota in QUOTA_A.items():
        ok = [n for n in src[source] if n not in used
              and (r := scorer.score_name(n))["novel"] and r["gate_ok"]]
        if len(ok) < quota:
            sys.exit(f"only {len(ok)} usable names from {source}, need {quota}")
        chosen = rng.sample(ok, quota)
        halves = ["calibration"] * (quota // 2) + ["test"] * (quota - quota // 2)
        rng.shuffle(halves)
        picked += [(n, source, h) for n, h in zip(chosen, halves)]
        used.update(chosen)
    rng.shuffle(picked)
    key_a = pd.DataFrame([{"id": i + 1, "name": n, "source": s, "half": h,
                           "ngram_plausibility": scorer.score_name(n)["plausibility"]}
                          for i, (n, s, h) in enumerate(picked)])

    # ---- Task B: pairs (candidate, the training name a view matched it to) ----
    tau = scorer.meaning_threshold
    cand = [n for k in ("real held-out", "char-RNN", "5-gram", "one-letter edits") for n in src[k]]
    cand = [n for n in dict.fromkeys(cand) if n not in used and n not in scorer.train_set]
    scorer._prepare(cand)
    strata = {k: [] for k in QUOTA_B}
    for n in cand:
        r = scorer.score_name(n)
        if not r["gate_ok"]:
            continue
        if r["tier"] == "near_copy":
            strata["near_copy"].append((n, r["nearest"]))
        elif r["tier"] == "respelling":
            strata["respelling"].append((n, r["sound_match"]))
        elif r["tier"] == "derived":
            strata["derived"].append((n, r["meaning_nearest"]))
        elif r["tier"] == "novel":
            if len(n) >= 5 and r["novelty"] <= 0.25:
                strata["under spelling"].append((n, r["nearest"]))
            elif tau - 0.05 <= r["meaning_sim"] < tau:
                strata["under meaning"].append((n, r["meaning_nearest"]))
            elif r["novelty"] >= 0.4 and r["meaning_sim"] < tau - 0.1:
                strata["distant"].append((n, r["nearest"]))
    pairs = []
    for stratum, quota in QUOTA_B.items():
        if len(strata[stratum]) < quota:
            sys.exit(f"only {len(strata[stratum])} pairs for {stratum}, need {quota}")
        pairs += [(left, right, stratum) for left, right in rng.sample(strata[stratum], quota)]
    rng.shuffle(pairs)
    key_b = pd.DataFrame([{"id": i + 1, "left": a, "right": b, "stratum": s}
                          for i, (a, b, s) in enumerate(pairs)])

    OUT.mkdir(parents=True, exist_ok=True)
    key_a.to_csv(OUT / "key_A.csv", index=False)
    key_b.to_csv(OUT / "key_B.csv", index=False)
    key_a[["id", "name"]].assign(rating="").to_csv(OUT / "sheet_A.csv", index=False)
    key_b[["id", "left", "right"]].assign(same="").to_csv(OUT / "sheet_B.csv", index=False)
    (OUT / "INSTRUCTIONS.md").write_text(f"""# Rating instructions

About 15 minutes. Do it alone, in one sitting, without looking up any name, and without
talking about the names with the others until all three of us are done. Do not open the
`key_*.csv` files (they say where each name came from).

## The easy way: the rating app

Open **{APP_URL}** on your phone or laptop. It is an ordinary web page: no account, no
install. (The same page is `rating_app.html` in this folder, if you would rather open the file.)

1. Type your first name and press Start.
2. Part 1: {len(key_a)} names, one at a time. Tap 1 to 5 (or press the number keys). Ten of them
   come back a second time later on. That is on purpose, just answer again.
3. Part 2: {len(key_b)} pairs. Tap Same name or Different name (or press Y / N).
4. On the last screen press **Send my ratings** (phones) or **Copy my ratings**, and send that
   whole block of text to Andre.

You can go back to change an answer, and you can close the tab and resume later on the same
device. Nothing is sent anywhere automatically.

### What you are answering

**Part 1.** {RUBRIC}

**Part 2.** {QUESTION_B.replace("LEFT", "first").replace("RIGHT", "second").replace(" 1 = yes, 0 = no.", "")}

Examples: `jonn` / `jon` is the same name. `kaius` / `caius` is the same name. `torvus` /
`corvus` is different, a different first letter makes a different name.

## The manual way: CSV sheets

Only if the app does not work for you. Fill the `rating` column of `sheet_A.csv` with 1 to 5
and the `same` column of `sheet_B.csv` with 1 or 0, using the two questions above. Save your
copies in this folder as `rater_<yourname>_A.csv` and `rater_<yourname>_B.csv`, and keep the
`id` column as it is.

## For Andre: turning app results into files

Save each person's pasted text to a file and run
`python scripts/human_tasks.py import <file>`. It writes `rater_<name>_A.csv`,
`rater_<name>_B.csv` and `rater_<name>_repeats.csv` here and refuses incomplete results.
""", encoding="utf-8")
    print(f"wrote {OUT}/sheet_A.csv ({len(key_a)} names), sheet_B.csv ({len(key_b)} pairs), "
          "INSTRUCTIONS.md and the hidden key files")


def import_app(path: str | None) -> None:
    """Turn the rating app's result text into rater_<name>_A.csv / _B.csv / _repeats.csv."""
    text = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    lines = [x.strip() for x in text.strip().splitlines() if x.strip()]
    if not lines or lines[0] != "CVL-RATINGS v1" or lines[-1] != "END":
        sys.exit("this is not a complete result: it must start with 'CVL-RATINGS v1' and end with 'END'")
    rater = lines[1].split(":", 1)[1].strip()
    rows = {"A": {}, "A2": {}, "B": {}}
    for line in lines[2:-1]:
        kind, i, v, ms = line.split(",")
        rows[kind][int(i)] = (int(v), int(ms))
    sa, sb = pd.read_csv(OUT / "sheet_A.csv"), pd.read_csv(OUT / "sheet_B.csv")
    missing = [f"A:{i}" for i in sa["id"] if i not in rows["A"]] + [f"B:{i}" for i in sb["id"] if i not in rows["B"]]
    if missing:
        sys.exit(f"incomplete: {len(missing)} items unanswered (e.g. {missing[:5]}); ask {rater} to finish and resend")
    if any(not 1 <= v <= 5 for v, _ in rows["A"].values()) or any(v not in (0, 1) for v, _ in rows["B"].values()):
        sys.exit("a rating is out of range")
    sa.assign(rating=[rows["A"][i][0] for i in sa["id"]], ms=[rows["A"][i][1] for i in sa["id"]]) \
        .to_csv(OUT / f"rater_{rater}_A.csv", index=False)
    sb.assign(same=[rows["B"][i][0] for i in sb["id"]], ms=[rows["B"][i][1] for i in sb["id"]]) \
        .to_csv(OUT / f"rater_{rater}_B.csv", index=False)
    pd.DataFrame([{"id": i, "first": rows["A"][i][0], "second": v} for i, (v, _) in sorted(rows["A2"].items())]) \
        .to_csv(OUT / f"rater_{rater}_repeats.csv", index=False)
    med_a = sa["id"].map(lambda i: rows["A"][i][1]).median() / 1000
    print(f"imported {rater}: {len(sa)} names, {len(sb)} pairs, {len(rows['A2'])} repeats; "
          f"median {med_a:.1f} s per name")


def _raters(task: str, column: str) -> pd.DataFrame:
    files = sorted(OUT.glob(f"rater_*_{task}.csv"))
    if not files:
        sys.exit(f"no rater_*_{task}.csv files in {OUT}")
    cols = {}
    for f in files:
        d = pd.read_csv(f).set_index("id")[column]
        cols[f.stem[len("rater_"):-2]] = pd.to_numeric(d, errors="coerce")
    return pd.DataFrame(cols)


def _units(df: pd.DataFrame) -> list[list]:
    return [[None if pd.isna(v) else int(v) for v in row] for row in df.to_numpy()]


def analyze(fill_judge: bool, freeze: bool) -> None:
    scorer = default_scorer()
    report: dict = {}

    # ---- Task A ----
    key = pd.read_csv(OUT / "key_A.csv").set_index("id")
    ra = _raters("A", "rating")
    print(f"Task A: {ra.shape[1]} raters ({', '.join(ra.columns)}), {len(ra)} names")
    alpha_a = krippendorff_alpha(_units(ra), "ordinal")
    pairwise = ra.corr(method="spearman")
    human_pair = float(pd.Series([pairwise.iloc[i, j] for i in range(len(pairwise))
                                  for j in range(i)]).mean()) if ra.shape[1] > 1 else float("nan")
    print(f"  Krippendorff alpha (ordinal): {alpha_a:.3f}   mean rater-to-rater Spearman: {human_pair:.3f}")
    for f in sorted(OUT.glob("rater_*_repeats.csv")):       # same name shown twice, 40+ items apart
        r = pd.read_csv(f)
        if len(r):
            print(f"  {f.stem[6:-8]} rated repeated names the same {(r['first'] == r['second']).mean():.0%} of the time, "
                  f"within one point {((r['first'] - r['second']).abs() <= 1).mean():.0%}")
    df = key.join(ra.mean(axis=1).rename("human"))
    df["human_plausible"] = df["human"] >= 3

    judge = scorer.judge
    if judge is None:
        sys.exit("scorer_config.json has no judge_model")
    if fill_judge:
        judge.fill(df["name"].tolist())
    if judge.missing(df["name"].tolist()):
        sys.exit("some Task A names have no judge rating; rerun with --fill-judge")
    df["judge"] = [judge.rating(n) for n in df["name"]]

    calib, test = df[df["half"] == "calibration"], df[df["half"] == "test"]
    best = None
    for cut in (2, 3, 4, 5):       # chosen on the calibration half only
        k = cohen_kappa(list(calib["judge"] >= cut), list(calib["human_plausible"]))
        if best is None or k > best[1] + 1e-12 or (abs(k - best[1]) <= 1e-12 and abs(cut - 3) < abs(best[0] - 3)):
            best = (cut, k)
    cut = best[0]
    judge_rho = float(test["judge"].corr(test["human"], method="spearman"))
    ngram_rho = float(test["ngram_plausibility"].corr(test["human"], method="spearman"))
    pred = list(test["judge"] >= cut)
    acc = sum(p == h for p, h in zip(pred, test["human_plausible"])) / len(test)
    kappa = cohen_kappa(pred, list(test["human_plausible"]))
    passed = judge_rho > ngram_rho and (ra.shape[1] < 2 or judge_rho >= human_pair - 0.10)
    print(f"  judge cut chosen on calibration half: rating >= {cut} (kappa {best[1]:.3f})")
    print(f"  test half: Spearman with mean human rating  judge {judge_rho:.3f}   3-gram {ngram_rho:.3f}")
    print(f"  test half: plausible yes/no vs. humans  accuracy {acc:.3f}   Cohen kappa {kappa:.3f}")
    print(f"  acceptance test: {'PASS' if passed else 'FAIL'} "
          "(judge beats the 3-gram and is within 0.10 of rater-to-rater agreement)")
    print("\n  mean rating by source (human, judge):")
    print(df.groupby("source")[["human", "judge"]].mean().round(2).to_string())
    report["task_A"] = dict(raters=list(ra.columns), alpha_ordinal=alpha_a, human_pair_spearman=human_pair,
                            judge_cut=cut, judge_spearman_test=judge_rho, ngram_spearman_test=ngram_rho,
                            accuracy_test=acc, kappa_test=kappa, acceptance=passed)
    df.to_csv(OUT / "analysis_A.csv")

    # ---- Task B ----
    if list(OUT.glob("rater_*_B.csv")):
        kb = pd.read_csv(OUT / "key_B.csv").set_index("id")
        rb = _raters("B", "same")
        alpha_b = krippendorff_alpha(_units(rb), "nominal")
        kb["same"] = rb.mean(axis=1) >= 0.5
        by = kb.groupby("stratum")["same"].agg(["mean", "size"]).round(3)
        print(f"\nTask B: {rb.shape[1]} raters, Krippendorff alpha (nominal): {alpha_b:.3f}")
        print("  share of pairs raters call the same name, by stratum")
        print("  (flagged views should be high; 'under' and 'distant' should be low)")
        print(by.to_string())
        report["task_B"] = dict(alpha_nominal=alpha_b, same_rate_by_stratum=by["mean"].to_dict())
        kb.to_csv(OUT / "analysis_B.csv")

    (OUT / "analysis.json").write_text(json.dumps(report, indent=2))
    if freeze:
        cfg = json.loads(CONFIG_PATH.read_text())
        cfg["judge_cut"] = cut
        cfg["plausibility_mode"] = "judge" if passed else "ngram"
        cfg["notes"]["judge_rule"] = ("cut chosen on the calibration half of Task A; judge adopted only "
                                      "if it passed the acceptance test on the test half")
        cfg["notes"]["human_study"] = report["task_A"]
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        print(f"\nfroze {CONFIG_PATH.name}: plausibility_mode={cfg['plausibility_mode']}, judge_cut={cut}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    imp = sub.add_parser("import", help="read the rating app's result text (file, or stdin)")
    imp.add_argument("file", nargs="?")
    a = sub.add_parser("analyze")
    a.add_argument("--fill-judge", action="store_true")
    a.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    if args.cmd == "make":
        make()
    elif args.cmd == "import":
        import_app(args.file)
    else:
        analyze(args.fill_judge, args.freeze)
