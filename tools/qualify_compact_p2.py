"""Compare P2 candidate results with their recorded baselines."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pair_reports(old, new):
    def index(rows):
        indexed = {(r["level"], r["repetition"]): r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate case/repetition")
        return indexed

    a, b = index(old), index(new)
    if not a or a.keys() != b.keys():
        raise ValueError("Incomplete paired coverage")
    if any(a[k]["seed"] != b[k]["seed"] for k in a):
        raise ValueError("Mismatched paired seeds")
    return [(a[k], b[k]) for k in sorted(a)]


def validate_rows(rows, sha256, *, require_win=True):
    from tools.benchmark_planner_p1 import THREAD_VARS

    errors = []
    if not rows:
        return ["empty report"]
    for r in rows:
        valid = (
            r.get("status") in (("win",) if require_win else ("win", "loss"))
            and (r.get("win") is True or not require_win)
            and r.get("source_sha256") == sha256
            and r.get("profiled") is False
            and isinstance(r.get("evaluation_seconds"), (int, float))
            and 0 <= r["evaluation_seconds"] < 4.25
            and isinstance(r.get("peak_rss_bytes"), int)
            and 0 < r["peak_rss_bytes"] < 1_000_000_000
            and isinstance(r.get("cpu_affinity"), list)
            and len(r["cpu_affinity"]) == 1
            and all(r.get(k) for k in ("host", "cpu", "python", "platform", "packages"))
            and r.get("evaluation_limit_seconds") == 5.0
            and r.get("turn_limit") == 150
            and r.get("setup_limit_seconds") == 30.0
            and all((r.get("thread_limits") or {}).get(k) == "1" for k in THREAD_VARS)
            and all(
                ((r.get("packages") or {}).get(k) or {}).get("version")
                for k in ("grid-adventure", "grid-universe")
            )
        )
        if not valid:
            errors.append(r.get("level", "missing level"))
    return errors


def paired_regressions(baseline, candidate):
    try:
        pairs = pair_reports(baseline["results"], candidate["results"])
    except (KeyError, ValueError) as error:
        return {
            "paired_coverage": False,
            "error": str(error),
            "lost_wins": [],
            "reward_regressions": [],
        }
    return {
        "paired_coverage": True,
        "lost_wins": [a["level"] for a, b in pairs if a["win"] and not b["win"]],
        "reward_regressions": [
            a["level"]
            for a, b in pairs
            if a["win"] and b["win"] and b["reward"] < a["reward"]
        ],
    }


def qualify_public(candidate, baseline, candidate_sha256, baseline_sha256):
    from tools.qualify_planner_p1 import qualify_public as p1_checks

    base = p1_checks(candidate, candidate_sha256)
    rows = candidate["results"]
    paired = paired_regressions(baseline, candidate)
    base["checks"].update(
        no_simulator_disagreements=all(
            (r.get("agent_statistics") or {}).get("stop_reason") != "invalid"
            for r in rows
        ),
        all_22_wins_each_repeat=len(rows) == 66
        and all(r.get("win") is True for r in rows),
        each_public_performance_at_least_0_85=bool(rows)
        and all(
            isinstance(r.get("effective_performance"), (int, float))
            and r["effective_performance"] >= 0.85
            for r in rows
        ),
        baseline_exact_hash=bool(baseline["results"])
        and all(r.get("source_sha256") == baseline_sha256 for r in baseline["results"]),
        baseline_acceptance_valid=p1_checks(baseline, baseline_sha256)[
            "local_gates_passed"
        ],
        complete_paired_coverage=paired["paired_coverage"],
        no_lost_paired_wins=not paired["lost_wins"],
        no_paired_reward_regressions=not paired["reward_regressions"],
    )
    base["paired"] = paired
    base["local_gates_passed"] = all(base["checks"].values())
    return base


def qualify_generated(candidate, baseline, expected, candidate_sha256, baseline_sha256):
    rows = candidate["results"]
    labels = {r["level"] for r in rows}
    paired = paired_regressions(baseline, candidate)
    checks = {
        "coverage": len(rows) == len(expected) and labels == set(expected),
        "no_simulator_disagreements": all(
            (r.get("agent_statistics") or {}).get("stop_reason") != "invalid"
            for r in rows
        ),
        "candidate_hash": bool(rows)
        and all(r.get("source_sha256") == candidate_sha256 for r in rows),
        "baseline_hash": bool(baseline["results"])
        and all(r.get("source_sha256") == baseline_sha256 for r in baseline["results"]),
        "wins_at_least_95_percent": bool(rows)
        and sum(r.get("win") is True for r in rows) >= 0.95 * len(rows),
        "valid_resources": not validate_rows(rows, candidate_sha256, require_win=False),
        "baseline_valid": not validate_rows(
            baseline["results"], baseline_sha256, require_win=False
        ),
        "paired_coverage": paired["paired_coverage"],
        "no_lost_wins": not paired["lost_wins"],
        "no_reward_regressions": not paired["reward_regressions"],
    }
    return {
        "local_gates_passed": all(checks.values()),
        "checks": checks,
        "paired": paired,
        "wins": sum(r.get("win") is True for r in rows),
        "cases": len(rows),
        "optimum": "unknown; raw reward only",
    }


def differential_gate(report, sha256):
    categories = (
        "movement",
        "pickup",
        "key-door",
        "box",
        "overlap",
        "damage",
        "shield",
        "stacked",
        "boots",
        "phasing",
        "wait",
        "terminal",
        "final-turn",
    )
    return (
        report.get("passed") is True
        and report.get("comparisons", 0) >= 100000
        and report.get("layouts", 0) >= 1000
        and report.get("public_trace_cases") == 22
        and report.get("source_hashes", {}).get("agents/compact_p2.py")
        == sha256
        and all(report.get("coverage", {}).get(k, 0) > 0 for k in categories)
    )


def main():
    import argparse
    import json
    from pathlib import Path

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--differential", type=Path, required=True)
    p.add_argument("--heldout", type=Path)
    p.add_argument("--baseline-heldout", type=Path)
    p.add_argument("--heldout-manifest", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    def read(path):
        return json.loads(path.read_text())

    sha = read(args.manifest)["source_sha256"]
    result = qualify_public(
        read(args.public), read(args.baseline), sha, args.baseline_sha256
    )
    differential = read(args.differential)
    result["checks"]["differential_gate"] = differential_gate(differential, sha)
    if args.heldout and args.baseline_heldout and args.heldout_manifest:
        manifest = read(args.heldout_manifest)
        result["heldout"] = qualify_generated(
            read(args.heldout),
            read(args.baseline_heldout),
            [f"p2-{r['index']:03d}" for r in manifest["cases"]],
            sha,
            args.baseline_sha256,
        )
        result["checks"]["heldout_gate"] = result["heldout"]["local_gates_passed"]
    else:
        result["checks"]["heldout_gate"] = False
    result["local_gates_passed"] = all(result["checks"].values())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["local_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
