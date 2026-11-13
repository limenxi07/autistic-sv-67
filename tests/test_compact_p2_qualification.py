import copy
import unittest

from tools.qualify_compact_p2 import pair_reports, validate_rows


def row(level="case", repetition=0):
    return {
        "level": level,
        "repetition": repetition,
        "seed": 42 + repetition,
        "source_sha256": "abc",
        "status": "win",
        "win": True,
        "reward": -3,
        "profiled": False,
        "evaluation_seconds": 0.1,
        "peak_rss_bytes": 100,
        "cpu_affinity": [0],
        "thread_limits": {
            k: "1"
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "host": "test",
        "cpu": "cpu",
        "python": "3.13",
        "platform": "linux",
        "packages": {
            "grid-adventure": {"version": "1"},
            "grid-universe": {"version": "1"},
        },
        "setup_limit_seconds": 30.0,
        "evaluation_limit_seconds": 5.0,
        "turn_limit": 150,
    }


class QualificationTests(unittest.TestCase):
    def test_pairs_match_case_repetition_and_seed(self):
        a = [row("a"), row("b")]
        self.assertEqual(len(pair_reports(a, list(reversed(a)))), 2)
        for bad in ([row("a")], [row("a"), row("a")]):
            with self.assertRaises(ValueError):
                pair_reports(a, bad)
        b = copy.deepcopy(a)
        b[0]["seed"] = 99
        with self.assertRaises(ValueError):
            pair_reports(a, b)

    def test_invalid_evidence_rejected(self):
        self.assertEqual(validate_rows([row()], "abc"), [])
        for field, value in [
            ("status", "timeout"),
            ("source_sha256", "wrong"),
            ("profiled", True),
            ("evaluation_seconds", 4.25),
            ("peak_rss_bytes", 1_000_000_000),
            ("cpu_affinity", None),
            ("host", None),
            ("evaluation_limit_seconds", 7.0),
        ]:
            bad = row()
            bad[field] = value
            self.assertTrue(validate_rows([bad], "abc"), field)

    def test_operation_probe_stops_at_requested_transitions(self):
        from tools.benchmark_compact_p2 import operation_probe

        for limit in (1, 3):
            r = operation_probe(
                "artifacts/p1/frozen/task1_agent.py", "A1 Maze Turns", transitions=limit
            )
            self.assertLessEqual(r["transitions"], limit)
            self.assertEqual(r["requested_transitions"], limit)
            self.assertIn("first_win_transitions", r)
            self.assertIn("provenance", r)


class PublicGateTests(unittest.TestCase):
    def reports(self):
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        rows = []
        for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS:
            for repeat in range(3):
                r = row(label, repeat)
                r.update(
                    effective_performance=1.0,
                    setup_limit_seconds=30.0,
                    packages={
                        "grid-adventure": {"version": "1"},
                        "grid-universe": {"version": "1"},
                    },
                    thread_limits={
                        name: "1"
                        for name in (
                            "OMP_NUM_THREADS",
                            "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "NUMEXPR_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS",
                        )
                    },
                )
                rows.append(r)
        return {"results": rows}

    def test_p2_public_gate_requires_every_win_and_reward_pair(self):
        from tools.qualify_compact_p2 import qualify_public

        baseline = self.reports()
        candidate = self.reports()
        self.assertTrue(
            qualify_public(candidate, baseline, "abc", "abc")["local_gates_passed"]
        )
        for field, value in [
            ("win", False),
            ("reward", -4),
            ("effective_performance", 0.84),
            ("thread_limits", {}),
            ("profiled", True),
            ("source_sha256", "bad"),
        ]:
            bad = copy.deepcopy(candidate)
            bad["results"][-1][field] = value
            self.assertFalse(
                qualify_public(bad, baseline, "abc", "abc")["local_gates_passed"], field
            )
        for rows in (
            candidate["results"][:-1],
            candidate["results"] + [candidate["results"][0]],
        ):
            self.assertFalse(
                qualify_public({"results": rows}, baseline, "abc", "abc")[
                    "local_gates_passed"
                ]
            )

    def test_generated_allows_five_percent_losses_without_baseline_regression(self):
        from tools.qualify_compact_p2 import qualify_generated

        rows = [row("g" + str(i)) for i in range(20)]
        rows[-1].update(status="loss", win=False)
        result = qualify_generated(
            {"results": rows},
            {"results": copy.deepcopy(rows)},
            [r["level"] for r in rows],
            "abc",
            "abc",
        )
        self.assertTrue(result["local_gates_passed"])

    def test_differential_gate_rejects_missing_categories_and_source_drift(self):
        from tools.qualify_compact_p2 import differential_gate

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
        report = {
            "passed": True,
            "comparisons": 100000,
            "layouts": 1000,
            "public_trace_cases": 22,
            "source_hashes": {"agents/compact_p2.py": "abc"},
            "coverage": dict.fromkeys(categories, 1),
        }
        self.assertTrue(differential_gate(report, "abc"))
        del report["coverage"]["boots"]
        self.assertFalse(differential_gate(report, "abc"))
        report["coverage"]["boots"] = 1
        self.assertFalse(differential_gate(report, "bad"))

    def test_differential_runs_the_selected_export_bytes(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path

        from tools.differential_compact_p2 import run

        source = Path("artifacts/p2/p2a-ready/task1_agent.py")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            run(layouts=0, comparisons=1, seed=4, output=output, source=source)
            report = json.loads(output.read_text())
            self.assertEqual(
                report["source_hashes"]["agents/compact_p2.py"],
                hashlib.sha256(source.read_bytes()).hexdigest(),
            )
