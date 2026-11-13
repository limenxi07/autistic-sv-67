"""Assemble qualification inputs and summaries from the frozen V3 receipts.

Run from the repository root after copying completed Omarchy reports. This
does not execute the candidate, tune a model, or rewrite frozen sources.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from tools.qualify_v3 import qualify
from tools.v3_data import canonical


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def summary(rows):
    confusion = Counter()
    for row in rows:
        for expected, predicted, count in row.get("initial_confusion", []):
            confusion[expected, predicted] += count
    classes = sorted({expected for expected, _ in confusion})
    return {
        "episodes": len(rows),
        "wins": sum(r.get("status") == "win" and r.get("win") is True for r in rows),
        "initial_boards_exact": sum(r.get("initial_board_exact") is True for r in rows),
        "median_evaluation_seconds": median(r["evaluation_seconds"] for r in rows),
        "max_evaluation_seconds": max(r["evaluation_seconds"] for r in rows),
        "max_strict_cold_seconds": max(r["strict_cold_seconds"] for r in rows),
        "max_peak_rss_bytes": max(r["peak_rss_bytes"] for r in rows),
        "per_class": {
            name: {
                "support": sum(n for (a, _), n in confusion.items() if a == name),
                "correct": confusion[name, name],
                "recall": confusion[name, name]
                / sum(n for (a, _), n in confusion.items() if a == name),
            }
            for name in classes
        },
        "confusion": [[a, b, n] for (a, b), n in sorted(confusion.items())],
        "failures": [
            r for r in rows if r.get("status") != "win" or r.get("win") is not True
        ],
    }


def main():
    out = Path("artifacts/v3/qualification-reviewed")
    frozen = Path("artifacts/v3/frozen-reviewed")
    manifest = read(frozen / "manifest.json")
    for name, expected in manifest["runtime_sources"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    raw_source = (frozen / "task2_agent.py").read_bytes()
    assert hashlib.sha256(raw_source).hexdigest() == manifest["source_sha256"]
    public, heldout, structured, oracle = (
        read(out / filename)
        for filename in (
            "public-66.json",
            "final-300.json",
            "structured-266.json",
            "oracle-22.json",
        )
    )
    assert [len(r["results"]) for r in (public, heldout, structured, oracle)] == [
        66,
        300,
        266,
        22,
    ]
    assert (out / "pipeline.exit").read_text().strip() == "0"
    baseline = read("artifacts/v3/qualification/structured-baseline.json")
    jobs = read("artifacts/v3/qualification/regression-jobs.json")
    final_jobs = read(out / "final-jobs.json")
    dataset = read("artifacts/v3/data/manifest-v5.json")
    audit = read("artifacts/v3/data/final-300-v5/audit.json")
    records = [
        json.loads(line)
        for line in Path("artifacts/v3/data/final-300-v5/boards.jsonl")
        .read_text()
        .splitlines()
    ]
    manifest.update(
        artifact_path=str(frozen / "task2_agent.py"),
        dataset=dataset,
        dataset_sha256=hashlib.sha256(canonical(dataset)).hexdigest(),
        heldout_records=records,
        heldout_audit=audit,
        heldout_manifest_path="artifacts/v3/data/manifest-v5.json",
        heldout_asset_root="artifacts/v3/data/final-300-v5/asset-root",
        heldout_class_counts=audit["class_counts"],
        heldout_family_counts=audit["family_counts"],
        expected_heldout=[(j["label"], j["seed"]) for j in final_jobs],
        expected_regression=[(j["label"], j["seed"]) for j in jobs],
    )
    regression = {
        "results": structured["results"],
        "baseline": baseline["results"],
        "oracle": oracle["results"],
    }
    write(out / "acceptance-manifest.json", manifest)
    write(out / "regression-paired.json", regression)
    result = qualify(public, heldout, regression, manifest)
    write(out / "qualification.json", result)
    paired = {(r["level"], r["seed"]): r for r in baseline["results"]}
    summaries = {
        "source_sha256": manifest["source_sha256"],
        "suites": {
            name: summary(report["results"])
            for name, report in (
                ("public-image", public),
                ("final-image", heldout),
                ("structured", structured),
                ("oracle", oracle),
            )
        },
        "final_by_family": {
            family: summary(
                [r for r in heldout["results"] if r["label"].split("/")[0] == family]
            )
            for family in sorted(audit["family_counts"])
        },
        "structured_exact_paired_rewards": sum(
            r["reward"] == paired[r["label"], r["seed"]]["reward"]
            for r in structured["results"]
        ),
        "public_min_effective_performance": min(
            r["effective_performance"] for r in public["results"]
        ),
        "final_axis": "new-layout/unseen-art",
        "evidence_boundary": result["evidence_boundary"],
    }
    write(out / "summary.json", summaries)
    print(json.dumps(result, indent=2))
    if not result["local_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
