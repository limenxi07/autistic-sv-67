import copy
import json
import unittest
from pathlib import Path

from tools.qualify_hierarchical_p3 import paired_rewards, validate_report


class EvidenceTests(unittest.TestCase):
    def report(self):
        report = json.loads(
            Path("artifacts/p2-next/candidate-task3-boss-v3.json").read_text()
        )
        return report, report["results"][0]["source_sha256"]

    def test_accepts_exact_unprofiled_boss_receipt(self):
        r, sha = self.report()
        self.assertEqual(validate_report(r, sha, 6, 15, 13.5), [])

    def test_rejects_wrong_watchdog_source_profile_and_missing_pair(self):
        for field, value in [
            ("evaluation_limit_seconds", 5),
            ("source_sha256", "wrong"),
            ("profiled", True),
            ("cpu_affinity", [0, 1]),
            ("replay", {"matches": False}),
        ]:
            r, sha = self.report()
            r["results"][0][field] = value
            self.assertTrue(validate_report(r, sha, 6, 15, 13.5), field)
        r, sha = self.report()
        r["results"].pop()
        self.assertTrue(validate_report(r, sha, 6, 15, 13.5))

    def test_rejects_duplicate_and_reward_regression(self):
        r, sha = self.report()
        r["results"][1] = copy.deepcopy(r["results"][0])
        self.assertTrue(validate_report(r, sha, 6, 15, 13.5))
        old, _ = self.report()
        new = copy.deepcopy(old)
        new["results"][0]["reward"] -= 1
        self.assertTrue(paired_rewards(old, new))
        new = copy.deepcopy(old)
        new["results"][0]["seed"] += 1
        self.assertTrue(paired_rewards(old, new))
