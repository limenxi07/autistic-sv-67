"""Verify the final checkpoint's hashes, paired outcomes and resource limits."""

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    path = Path(path)
    remote_root = Path("/home/enxi/coding/p2-next-20261109")
    if path.is_relative_to(remote_root):
        path = path.relative_to(remote_root)
    from tools.historical_sources import historical_bytes

    return hashlib.sha256(historical_bytes(path)).hexdigest()


def distribution(rows):
    def metric(field):
        values = sorted(r[field] for r in rows)
        return dict(
            median=statistics.median(values),
            maximum=max(values),
            p95_nearest_rank=values[math.ceil(0.95 * len(values)) - 1],
        )

    return dict(
        cases=len(rows),
        statuses=dict(Counter(r["status"] for r in rows)),
        evaluation_seconds=metric("evaluation_seconds"),
        cold_seconds=metric("worker_wall_seconds"),
        setup_seconds=metric("setup_seconds"),
        peak_rss_bytes=max(r["peak_rss_bytes"] for r in rows),
    )


def verify(root):
    manifest = json.loads((root / "candidate-v3-manifest.json").read_text())
    expected = manifest["source_sha256"]
    assert sha("agents/boss_p2_next.py") == expected
    assert Path("agents/boss_p2_next.py").stat().st_size < 2_000_000
    for path, digest in manifest["hashes"].items():
        assert sha(path) == digest, path
    protected = json.loads((root / "protected-before.json").read_text())
    assert all(sha(path) == digest for path, digest in protected.items())
    differential = json.loads((root / "differential-v3.json").read_text())
    assert differential["source_sha256"] == expected and differential["passed"]
    assert differential["comparisons"] >= 100_000
    assert differential["many_initial_supported"] >= 1000
    assert differential["many_compared"] >= 1000
    suites = {}
    for suite, filename, count, limit in (
        ("boss", "candidate-task3-boss-v3.json", 6, 15),
        ("public", "candidate-public-v3.json", 66, 5),
        ("stress", "stress-v3.json", 12, 15),
        ("generated", "generated-regression-v3.json", 200, 5),
    ):
        report = json.loads((root / filename).read_text())
        rows = report["results"]
        assert len(rows) == count and not report["instrumented"]
        assert len({(r["level"], r["repetition"]) for r in rows}) == count
        assert all(
            r["source_sha256"] == expected and r["replay"]["matches"] for r in rows
        )
        assert all(
            not r["profiled"]
            and r["evaluation_limit_seconds"] == limit
            and r["seed"] == 42 + r["repetition"]
            for r in rows
        )
        assert all(
            r["status"] in ("win", "loss")
            and r["evaluation_seconds"] < limit
            and r["peak_rss_bytes"] < 1_000_000_000
            for r in rows
        )
        assert all(
            r["cpu_affinity"] == [0] and set(r["thread_limits"].values()) == {"1"}
            for r in rows
        )
        if suite != "stress":
            assert all(r["status"] == "win" for r in rows), suite
        target = 13.5 if limit == 15 else 4.25
        assert all(r["evaluation_seconds"] < target for r in rows)
        suites[suite] = rows
    for variant in ("p2a", "p2b"):
        baseline = json.loads(
            Path(f"artifacts/p2/{variant}-ready/public-repeat3.json").read_text()
        )["results"]
        pairs = {(r["level"], r["repetition"]): r for r in baseline}
        assert set(pairs) == {(r["level"], r["repetition"]) for r in suites["public"]}
        assert all(
            r["reward"] == pairs[r["level"], r["repetition"]]["reward"]
            for r in suites["public"]
        )
        baseline = json.loads(
            Path(f"artifacts/p2/{variant}-ready/heldout-v2.json").read_text()
        )["results"]
        pairs = {
            int(r["level"].split("-")[-1]): r for r in baseline if r["repetition"] == 0
        }
        assert all(
            r["reward"] == pairs[int(r["level"].split("-")[-1])]["reward"]
            for r in suites["generated"]
        )
    for filename in ("unchanged-v3-oracle.json", "candidate-oracle.json"):
        report = json.loads((root / filename).read_text())
        assert report["bridge_gate_passed"] and len(report["results"]) == 22
        assert all(r["paired_reward_match"] for r in report["results"])
        assert all(
            r["evaluation_seconds"] < 5 and r["peak_rss_bytes"] < 1_000_000_000
            for r in report["results"]
        )
        assert all(sha(p) == digest for p, digest in report["source_sha256"].items())
    assert "Ran 161 tests" in (root / "full-suite-final.log").read_text()
    assert (root / "full-suite-final.log").read_text().rstrip().endswith("OK")
    bosses = {}
    for label in ("C0", "C1"):
        rows = [r for r in suites["boss"] if r["level"].startswith(label)]
        assert len(rows) == 3
        assert all(r["agent_statistics"]["reference"] is None for r in rows)
        assert all(r["reward"] == (-68 if label == "C0" else -163) for r in rows)
        bosses[label] = dict(
            **distribution(rows),
            reward=rows[0]["reward"],
            first_verified_search_seconds=statistics.median(
                float(r["agent_statistics"]["time_to_first_win"]) for r in rows
            ),
            stop_reasons=dict(
                Counter(r["agent_statistics"]["stop_reason"] for r in rows)
            ),
        )
    coverage = defaultdict(Counter)
    for row in json.loads((root / "coverage-final.json").read_text())["traces"]:
        coverage[row["label"].split(":")[0]].update(row["counts"])
    return dict(
        source_sha256=expected,
        source_bytes=manifest["bytes"],
        suite_distributions={s: distribution(rows) for s, rows in suites.items()},
        bosses=bosses,
        differential={
            k: differential[k]
            for k in (
                "comparisons",
                "frames",
                "many_initial_supported",
                "many_compared",
                "coverage",
            )
        },
        coverage={k: dict(v) for k, v in coverage.items()},
        public_paired_rewards="66/66 equal P2a and P2b",
        generated_paired_rewards="200/200 equal P2a and P2b",
        oracle="unchanged V3 and separate candidate probe each 22/22 paired reward matches",
        protected_hashes_unchanged=len(protected),
        repository_tests=161,
        stress=[
            dict(
                level=r["level"],
                seed=r["seed"],
                status=r["status"],
                reward=r["reward"],
                seconds=r["evaluation_seconds"],
                searches=int(r["agent_statistics"]["search_calls"]),
                stop_reason=r["agent_statistics"]["stop_reason"],
            )
            for r in suites["stress"]
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/p2-next"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = verify(args.root)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in ("stress", "suite_distributions", "coverage", "differential")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
