"""
evaluate.py - compares automatic counts with human (expected) counts for every row in tests.csv.

For each test it reports the count of the final state-machine counter (2D image angle), the same
counter without smoothing, the same counter with 3D world landmarks (real videos only) and a naive
single-threshold counter, so the effect of each design choice is visible.
Writes results/evaluation_results.md and updates the results section of README.md.
Run:  python evaluate.py
"""
import csv
import shutil
from datetime import date
from pathlib import Path

from counter import process_landmarks, process_video
from rep_counter import get_exercise

TESTS = Path("tests.csv")
RESULTS = Path("results")
RUNS = RESULTS / "runs"
PLOTS = RESULTS / "plots"
RECORDED = Path("data/recorded")
README = Path("README.md")
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def run_test(row: dict) -> dict | None:
    path = Path(row["file"])
    if not path.is_file():
        return None
    ex = get_exercise(row["exercise"])
    use_3d = row["coords"] == "world"
    name = f"{path.stem}_{row['coords']}"
    if path.suffix.lower() in VIDEO_EXT:
        landmarks = RECORDED / f"{path.stem}.csv"
        session, _ = process_video(str(path), ex, row["side"], use_3d=use_3d, save_video=False,
                                   out_dir=RUNS / name, landmarks_csv=landmarks)
    else:
        landmarks = path
        session, _ = process_landmarks(path, ex, row["side"], use_3d=use_3d, out_dir=RUNS / name)
    # method comparison: 3D world landmarks (real videos only; needs the video itself)
    world = None
    if path.suffix.lower() in VIDEO_EXT and not use_3d:
        ws, _ = process_video(str(path), ex, row["side"], use_3d=True, save_video=False,
                              out_dir=RUNS / f"{path.stem}_world")
        world = ws.count
    # ablation: same landmarks without smoothing (2D only)
    if use_3d:
        no_smooth = None
    else:
        ns, _ = process_landmarks(landmarks, ex, row["side"], smoothing=1.0, out_dir=RUNS / f"{name}_nosmooth")
        no_smooth = ns.count
    PLOTS.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUNS / name / "plot.png", PLOTS / f"{name}.png")
    s = session.summary()
    return {"counted": s["total_reps"], "partial": s["partial_reps"], "rejected": s["rejected_too_fast"],
            "no_smooth": no_smooth, "world": world, "naive": s["naive_baseline_reps"],
            "missing": s["missing_or_low_confidence_pct"], "plot": f"plots/{name}.png"}


def stats(rows, key):
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return "-", "-", "-", "-"
    errors = [r[key] - r["expected"] for r in rows]
    exact = sum(e == 0 for e in errors)
    mae = sum(abs(e) for e in errors) / len(errors)
    over = sum(e for e in errors if e > 0)
    under = -sum(e for e in errors if e < 0)
    return f"{exact}/{len(rows)}", f"{mae:.2f}", str(over), str(under)


def main() -> None:
    if not TESTS.is_file():
        raise SystemExit("tests.csv not found.")
    done, skipped = [], []
    with TESTS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["expected"] = int(row["expected"])
            row["expected_partial"] = int(row["expected_partial"] or 0)
            print(f"Testing {row['file']} ({row['coords']}) ...", flush=True)
            result = run_test(row)
            if result is None:
                print("  skipped: file not found")
                skipped.append(row)
                continue
            row.update(result)
            done.append(row)
            print(f"  expected {row['expected']}, counted {row['counted']}, partial {row['partial']}")

    summary = ["| Method | Exact count | Mean abs. error | Extra reps (false positives) | Missed reps (false negatives) |",
               "|---|---|---|---|---|"]
    for label, key in (("State machine + smoothing (final)", "counted"),
                       ("State machine without smoothing", "no_smooth"),
                       ("State machine with 3D world landmarks (real videos only)", "world"),
                       ("Naive single threshold (baseline)", "naive")):
        summary.append(f"| {label} | " + " | ".join(stats(done, key)) + " |")

    cats = list(dict.fromkeys(r["category"] for r in done))
    cat_table = ["| Category | Tests | Exact count | Mean abs. error |", "|---|---|---|---|"]
    for c in cats:
        rows = [r for r in done if r["category"] == c]
        exact, mae, _, _ = stats(rows, "counted")
        cat_table.append(f"| {c} | {len(rows)} | {exact} | {mae} |")

    detail = ["| Test | Exercise | Angle | Expected | Counted | Partial (exp./found) | No smoothing | 3D world | Naive | Missing frames | Plot |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in done:
        mark = "✅" if r["counted"] == r["expected"] else "❌"
        ns = "-" if r["no_smooth"] is None else r["no_smooth"]
        wd = "-" if r["world"] is None else r["world"]
        detail.append(f"| {r['description']} | {r['exercise']} | {r['coords']} | {r['expected']} | "
                      f"{r['counted']} {mark} | {r['expected_partial']}/{r['partial']} | {ns} | {wd} | {r['naive']} | "
                      f"{r['missing']}% | [plot]({r['plot']}) |")

    header = [f"_Generated by `python evaluate.py` on {date.today()}: {len(done)} tests run"
              + (f", {len(skipped)} skipped (file not found)" if skipped else "") + "._", "",
              "**Overall** (automatic count compared with the human count)", "", *summary, "",
              "**Per category** (final counter)", "", *cat_table, ""]
    report = ["# Evaluation results", "", *header, "## All tests", "", *detail, ""]
    if skipped:
        report += ["## Skipped tests (file not found)", ""] + [f"- `{r['file']}`: {r['description']}" for r in skipped]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "evaluation_results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nWrote {RESULTS / 'evaluation_results.md'}")

    start, end = "<!-- EVAL_START -->", "<!-- EVAL_END -->"
    if README.is_file():
        text = README.read_text(encoding="utf-8")
        if start in text and end in text:
            before, rest = text.split(start, 1)
            after = rest.split(end, 1)[1]
            block = "\n".join(header + ["All tests with plots: [results/evaluation_results.md](results/evaluation_results.md)"])
            README.write_text(f"{before}{start}\n{block}\n{end}{after}", encoding="utf-8")
            print("Updated the results section in README.md")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
