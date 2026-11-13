import json
import unittest
from pathlib import Path

from tools.diagnose_vision_v4 import extract_regressions, verify_baseline


class DiagnosticTests(unittest.TestCase):
    def test_extracts_five_losses_and_twelve_wrong_boards(self):
        report = json.loads(
            Path("artifacts/v3/qualification-reviewed/final-300.json").read_text()
        )
        result = extract_regressions(report)
        self.assertEqual(len(result["lost_jobs"]), 5)
        self.assertEqual(len(result["incorrect_initial_jobs"]), 12)
        jobs = result["diagnostic_jobs"]
        self.assertEqual(len(jobs), 13)
        self.assertEqual(len({json.dumps(j, sort_keys=True) for j in jobs}), 13)
        for job in jobs:
            self.assertIn("expected_pixels_sha256", job)
            self.assertIn(job, [r["job"] for r in report["results"]])

    def test_baseline_hashes_and_distinct_p3_entrypoints(self):
        result = verify_baseline()
        self.assertEqual(result["p3_bytes"], 96112)
        self.assertEqual(result["p3_default_rows"], 266)
        self.assertEqual(result["p3_task3_rows"], 60)
        self.assertEqual(len(result["v3_runtime_sources"]), 8)
