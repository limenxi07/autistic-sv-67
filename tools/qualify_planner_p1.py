"""Evidence gates for P1 checkpoint selection and exact-artifact qualification."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

PUBLIC = set(PUBLIC_GAMEPLAY_LEVEL_BUILDERS)
INTRO = {label for label in PUBLIC if label.startswith("A")}


def _mean(rows):
    return statistics.mean(r["effective_performance"] for r in rows) if rows else None


def compare(a, b):
    """Single-run public comparison. Tuning reliability is checked separately."""
    old = {r["level"]: r for r in a["results"]}
    new = {r["level"]: r for r in b["results"]}
    coverage = (
        len(a["results"]) == len(b["results"]) == 22 and set(old) == set(new) == PUBLIC
    )
    lost = [k for k in old.keys() & new.keys() if old[k]["win"] and not new[k]["win"]]
    new_failures = [
        k
        for k in old.keys() & new.keys()
        if new[k]["status"] in ("timeout", "error", "setup_error")
        and old[k]["status"] not in ("timeout", "error", "setup_error")
    ]
    unprofiled = all(not r.get("profiled", False) for r in a["results"] + b["results"])
    old_mean, new_mean = _mean(a["results"]), _mean(b["results"])
    return {
        "promote_p1b": bool(
            coverage
            and unprofiled
            and not lost
            and not new_failures
            and new_mean > old_mean
        ),
        "complete_coverage": coverage,
        "unprofiled": unprofiled,
        "lost_wins": lost,
        "new_failures": new_failures,
        "p1a_mean": old_mean,
        "p1b_mean": new_mean,
    }


def qualify_public(report, expected_sha256):
    rows = report["results"]
    expected = {(label, rep) for label in PUBLIC for rep in range(3)}
    coverage = (
        len(rows) == 66 and {(r["level"], r["repetition"]) for r in rows} == expected
    )
    intro = [r for r in rows if r["level"] in INTRO]
    checks = {
        "all_22_cases_three_repeats": coverage,
        "exact_source_hash": bool(rows)
        and all(r["source_sha256"] == expected_sha256 for r in rows),
        "unprofiled": bool(rows) and all(r.get("profiled") is False for r in rows),
        "single_cpu_affinity": bool(rows)
        and all(
            isinstance(r.get("cpu_affinity"), list)
            and len(r["cpu_affinity"]) == 1
            and isinstance(r["cpu_affinity"][0], int)
            for r in rows
        ),
        "single_thread_libraries": bool(rows)
        and all(
            all(
                (r.get("thread_limits") or {}).get(name) == "1"
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            )
            for r in rows
        ),
        "turn_limit_150": bool(rows) and all(r.get("turn_limit") == 150 for r in rows),
        "evaluation_watchdog_5s": bool(rows)
        and all(r.get("evaluation_limit_seconds") == 5.0 for r in rows),
        "setup_watchdog_30s": bool(rows)
        and all(r.get("setup_limit_seconds") == 30.0 for r in rows),
        "host_and_environment_recorded": bool(rows)
        and all(
            all(r.get(name) for name in ("host", "cpu", "python", "platform"))
            and all(
                ((r.get("packages") or {}).get(name) or {}).get("version")
                for name in ("grid-adventure", "grid-universe")
            )
            for r in rows
        ),
        "no_errors_or_timeouts": bool(rows)
        and all(r["status"] in ("win", "loss") for r in rows),
        "all_11_intro_wins_each_repeat": len(intro) == 33
        and all(r["win"] and r["status"] == "win" for r in intro),
        "every_episode_under_4_25_seconds": bool(rows)
        and all(
            r["evaluation_seconds"] is not None and r["evaluation_seconds"] < 4.25
            for r in rows
        ),
        "every_peak_rss_under_1_GB": bool(rows)
        and all(
            r["peak_rss_bytes"] is not None and r["peak_rss_bytes"] < 1_000_000_000
            for r in rows
        ),
        "intro_mean_at_least_0_85": bool(intro) and _mean(intro) >= 0.85,
    }
    return {
        "local_gates_passed": all(checks.values()),
        "checks": checks,
        "source_sha256": expected_sha256,
        "intro_mean_effective_performance": _mean(intro),
        "public_mean_effective_performance": _mean(rows),
        "per_level": {
            label: {
                "wins": sum(r["win"] is True for r in rows if r["level"] == label),
                "mean_effective_performance": _mean(
                    [r for r in rows if r["level"] == label]
                ),
            }
            for label in sorted(PUBLIC)
        },
        "platform_debugging": "unverified by local benchmark",
        "graded_submission": "external user handoff",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("compare")
    selection.add_argument("--p1a", type=Path, required=True)
    selection.add_argument("--p1b", type=Path, required=True)
    selection.add_argument("--p1a-tuning", type=Path, required=True)
    selection.add_argument("--p1b-tuning", type=Path, required=True)
    qualification = sub.add_parser("qualify")
    qualification.add_argument("--public", type=Path, required=True)
    qualification.add_argument("--manifest", type=Path, required=True)
    for p in (selection, qualification):
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_text())

    if args.command == "compare":
        result = compare(read(args.p1a), read(args.p1b))
        a, b = read(args.p1a_tuning), read(args.p1b_tuning)
        old = {r["level"]: r for r in a["results"]}
        new = {r["level"]: r for r in b["results"]}
        result["tuning_coverage"] = (
            len(old) == len(new) == 50 and old.keys() == new.keys()
        )
        result["tuning_lost_wins"] = [
            label
            for label in old.keys() & new.keys()
            if old[label]["win"] and not new[label]["win"]
        ]
        result["tuning_new_failures"] = [
            label
            for label in new
            if new[label]["status"] in ("timeout", "error", "setup_error")
        ]
        result["promote_p1b"] &= (
            result["tuning_coverage"]
            and not result["tuning_lost_wins"]
            and not result["tuning_new_failures"]
        )
    else:
        result = qualify_public(read(args.public), read(args.manifest)["source_sha256"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
