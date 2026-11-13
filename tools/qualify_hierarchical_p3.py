"""Fail-closed exact-source P3 evidence checks and paired raw-reward comparison."""

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from tools.benchmark_planner_p1 import THREAD_VARS
from tools.qualify_compact_p2 import pair_reports


def validate_report(report, sha, count, watchdog, target):
    errors = []
    rows = report.get("results", [])
    if len(rows) != count:
        errors.append("incomplete case count")
    if report.get("instrumented") is not False:
        errors.append("instrumented or unspecified receipt")
    keys = [(r.get("level"), r.get("repetition"), r.get("seed")) for r in rows]
    if len(set(keys)) != len(keys):
        errors.append("duplicate cases")
    for r in rows:
        label = f"{r.get('level')}:{r.get('seed')}"
        checks = {
            "source": r.get("source_sha256") == sha,
            "watchdog": r.get("evaluation_limit_seconds") == watchdog,
            "power-profile-stability": r.get("power_profile_changed", False) is False,
            "turn-limit": r.get("turn_limit") == 150,
            "setup-watchdog": r.get("setup_limit_seconds") == 30,
            "provenance": all(
                r.get(k) for k in ("host", "cpu", "python", "platform", "packages")
            ),
            "profile": r.get("profiled") is False,
            "win": r.get("status") == "win"
            and r.get("win") is True
            and r.get("lose") is False,
            "replay": r.get("replay", {}).get("matches") is True,
            "CPU": len(r.get("cpu_affinity", [])) == 1,
            "threads": all(
                r.get("thread_limits", {}).get(k) == "1" for k in THREAD_VARS
            ),
            "time": isinstance(r.get("evaluation_seconds"), (float, int))
            and 0 <= r["evaluation_seconds"] < target,
            "RSS": isinstance(r.get("peak_rss_bytes"), int)
            and 0 < r["peak_rss_bytes"] < 1_000_000_000,
        }
        errors.extend(label + ": " + name for name, ok in checks.items() if not ok)
    return errors


def paired_rewards(old, new):
    try:
        pairs = pair_reports(old["results"], new["results"])
    except (ValueError, KeyError) as e:
        return [str(e)]
    return [
        str((a["level"], a["seed"], a.get("reward"), b.get("reward")))
        for a, b in pairs
        if a["reward"] != b["reward"] or a["win"] != b["win"]
    ]


def distribution(rows):
    times = sorted(r["evaluation_seconds"] for r in rows)
    return {
        "cases": len(rows),
        "wins": sum(r["win"] for r in rows),
        "median_seconds": statistics.median(times),
        "maximum_seconds": max(times),
        "p95_seconds": times[math.ceil(0.95 * len(times)) - 1],
        "peak_rss_bytes": max(r["peak_rss_bytes"] for r in rows),
    }


def qualify(root):
    freeze = json.loads((root / "candidate/freeze.json").read_text())
    sha = freeze["source_sha256"]
    checks = {}
    reports = {}
    for name, count, watchdog, target, entry in [
        ("boss", 6, 15, 13.5, "Task3Agent"),
        ("public", 66, 5, 4.25, "Agent"),
        ("generated", 200, 5, 4.25, "Agent"),
        ("stress", 36, 15, 13.5, "Task3Agent"),
        ("holdout", 18, 15, 13.5, "Task3Agent"),
    ]:
        report = json.loads((root / (name + ".json")).read_text())
        reports[name] = report
        errors = validate_report(report, sha, count, watchdog, target)
        if report.get("entrypoint") != entry:
            errors.append("wrong entrypoint budget")
        checks[name] = errors
    for name, filename in [
        ("boss", "candidate-task3-boss-v3.json"),
        ("public", "candidate-public-v3.json"),
        ("generated", "generated-regression-v3.json"),
    ]:
        baseline = json.loads((Path("artifacts/p2-next") / filename).read_text())
        checks[name + "-paired-rewards"] = paired_rewards(baseline, reports[name])
    for name, count, repeats in [("stress", 12, 3), ("holdout", 18, 1)]:
        actual = {
            (r["level"], r["seed"], r["repetition"]) for r in reports[name]["results"]
        }
        expected = {
            (f"case-{i:03d}", 42 + rep, rep)
            for i in range(count)
            for rep in range(repeats)
        }
        checks[name + "-coverage"] = (
            [] if actual == expected else ["wrong case/seed coverage"]
        )
    checks["freeze-before-acceptance"] = []
    frozen_at = datetime.fromisoformat(freeze["frozen_at"])
    for name, report in reports.items():
        if (
            datetime.fromisoformat(
                report.get("started_at_utc", "1970-01-01T00:00:00+00:00")
            )
            <= frozen_at
        ):
            checks["freeze-before-acceptance"].append(name)
    fixture = json.loads((root / "holdout-format-only.json").read_text())
    checks["frozen-fixture"] = (
        []
        if hashlib.sha256(
            Path("tests/hierarchical_p3_fixtures.py").read_bytes()
        ).hexdigest()
        == fixture["formatted_sha256"]
        and fixture["normalized_ast_equal"] is True
        else ["holdout fixture drift"]
    )
    old = json.loads(Path("artifacts/p2-next/stress-v3.json").read_text())["results"]
    new = {r["level"]: r for r in reports["stress"]["results"] if r["repetition"] == 0}
    checks["stress-preserved-winners"] = [
        r["level"]
        for r in old
        if r["win"]
        and (r["level"] not in new or r["reward"] != new[r["level"]]["reward"])
    ]
    diff = json.loads((root / "differential.json").read_text())
    checks["differential"] = (
        []
        if diff.get("passed")
        and diff.get("source_sha256") == sha
        and diff.get("comparisons", 0) >= 100000
        and diff.get("many_compared", 0) >= 1000
        else ["differential gate failed"]
    )
    source = Path("agents/hierarchical_p3.py").read_bytes()
    export = (root / "candidate/task1_agent.py").read_bytes()
    checks["exact-source"] = (
        []
        if source == export
        and hashlib.sha256(source).hexdigest() == sha
        and len(source) < 2_000_000
        else ["source/export mismatch"]
    )
    protected = json.loads((root / "protected-before.json").read_text())
    checks["protected-P1-P2"] = []
    for name, digest in protected.items():
        if name.startswith(("agents/v3", "agents/p3")):
            continue
        from tools.historical_sources import historical_bytes

        try:
            actual = hashlib.sha256(historical_bytes(name)).hexdigest()
        except FileNotFoundError:
            actual = None
        if actual != digest:
            checks["protected-P1-P2"].append(name)
    return {
        "passed": all(not e for e in checks.values()),
        "source_sha256": sha,
        "checks": checks,
        "distributions": {
            name: distribution(r["results"]) for name, r in reports.items()
        },
        "evidence_scope": freeze.get(
            "evidence_scope",
            "Structured public API only; raw generated rewards; old stress/public/generated reused; frozen disjoint holdout opened after candidate freeze; no perception or platform submission.",
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("artifacts/p3/structured"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = qualify(a.root)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
