import unittest
from copy import deepcopy

from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tools.qualify_planner_p1 import compare, qualify_public


def report(repeats=1, performance=1.0):
    return {
        "results": [
            {
                "level": name,
                "repetition": r,
                "status": "win",
                "win": True,
                "effective_performance": performance,
                "evaluation_seconds": 3.1,
                "peak_rss_bytes": 300_000_000,
                "source_sha256": "abc",
                "profiled": False,
                "cpu_affinity": [0],
                "thread_limits": {
                    name: "1"
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
                "turn_limit": 150,
                "evaluation_limit_seconds": 5.0,
                "setup_limit_seconds": 30.0,
                "host": "test-host",
                "cpu": "test-cpu",
                "python": "3.13",
                "platform": "Linux",
                "packages": {
                    name: {"version": "test"}
                    for name in ("grid-adventure", "grid-universe")
                },
            }
            for r in range(repeats)
            for name in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
        ]
    }


class QualificationTests(unittest.TestCase):
    def test_exact_scope_and_resource_gates(self):
        good = report(3)
        self.assertTrue(qualify_public(good, "abc")["local_gates_passed"])
        for key, value in [
            ("status", "timeout"),
            ("evaluation_seconds", 4.25),
            ("peak_rss_bytes", 1_000_000_000),
            ("peak_rss_bytes", None),
            ("source_sha256", "wrong"),
            ("profiled", True),
            ("win", False),
        ]:
            bad = deepcopy(good)
            bad["results"][0][key] = value
            self.assertFalse(qualify_public(bad, "abc")["local_gates_passed"], key)
        self.assertFalse(qualify_public(report(1), "abc")["local_gates_passed"])
        duplicate = deepcopy(good)
        duplicate["results"][0] = duplicate["results"][1]
        self.assertFalse(qualify_public(duplicate, "abc")["local_gates_passed"])

    def test_promotion_requires_improvement_and_preserves_wins(self):
        a, b = report(performance=0.5), report()
        self.assertTrue(compare(a, b)["promote_p1b"])
        self.assertFalse(compare(b, b)["promote_p1b"])
        b["results"][0]["status"] = "loss"
        b["results"][0]["win"] = False
        self.assertFalse(compare(a, b)["promote_p1b"])

    def test_execution_provenance_required_and_enforced(self):
        invalid = {
            "cpu_affinity": [0, 1, 2, 3],
            "thread_limits": {"OMP_NUM_THREADS": "4"},
            "turn_limit": 999,
            "evaluation_limit_seconds": 60.0,
            "setup_limit_seconds": 300.0,
            "host": None,
            "cpu": None,
            "python": None,
            "platform": None,
            "packages": {},
            "profiled": None,
        }
        for name, value in invalid.items():
            with self.subTest(field=name):
                bad = report(3)
                for row in bad["results"]:
                    row[name] = value
                self.assertFalse(qualify_public(bad, "abc")["local_gates_passed"])
                for row in bad["results"]:
                    del row[name]
                self.assertFalse(qualify_public(bad, "abc")["local_gates_passed"])
