import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

from tools.qualify_vision_v3 import qualify


class QualificationTests(unittest.TestCase):
    def fixture(self):
        row = {
            "label": "a",
            "seed": 2,
            "source_sha256": "candidate",
            "status": "win",
            "win": True,
            "reward": -3,
            "evaluation_seconds": 1.0,
            "cold_import_to_terminal_seconds": 1.2,
            "peak_rss_bytes": 300000000,
            "provenance": {
                "cpu_affinity": [0],
                "thread_limits": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
            },
            "torch_threads": [1, 1],
            "budget_reset": False,
            "budget_seconds": [3.0, 3.5],
            "effective_performance": 1.0,
            "runtime_overrides": {},
            "initial_board_exact": True,
            "observation": "image",
            "strict_cold_seconds": 1.3,
            "cold_watchdog_seconds": 5.0,
            "power_state_before": {"epp": "performance"},
            "power_state_after": {"epp": "performance"},
        }
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        public = {
            "results": [
                {**deepcopy(row), "label": label, "seed": seed}
                for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
                for seed in (2, 43, 20261109)
            ]
        }
        data = json.loads(Path("artifacts/v3/data/manifest-v5.json").read_text())
        records = [
            json.loads(line)
            for line in Path("artifacts/v3/data/final-300-v5/boards.jsonl")
            .read_text()
            .splitlines()
        ]
        audit = json.loads(
            Path("artifacts/v3/data/final-300-v5/audit.json").read_text()
        )
        jobs = json.loads(
            Path("artifacts/v3/qualification-reviewed/final-jobs.json").read_text()
        )
        dataset_hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for job in jobs:
            job["appearance_manifest_sha256"] = dataset_hash
        held = {
            "results": [
                {
                    **deepcopy(row),
                    "label": job["label"],
                    "seed": job["seed"],
                    "job": job,
                    "observation": "image",
                    "effective_performance": None,
                }
                for job in jobs
            ]
        }
        regression = {
            "results": [
                {**deepcopy(row), "label": f"r{i}", "observation": "gridstate"}
                for i in range(266)
            ],
            "baseline": [
                {**deepcopy(row), "label": f"r{i}", "source_sha256": "parent"}
                for i in range(266)
            ],
            "oracle": [
                {
                    **deepcopy(row),
                    "label": label,
                    "paired_reward_match": True,
                    "observation": "oracle",
                }
                for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
            ],
        }
        manifest = {
            "source_sha256": "candidate",
            "backend_sha256": "parent",
            "model_sha256": "model",
            "bytes": 600000,
            "configuration": {"initial_seconds": 3.0, "cutoff_seconds": 3.5},
            "export_parity_passed": True,
            "dataset": data,
            "expected_heldout": [(j["label"], j["seed"]) for j in jobs],
            "heldout_records": records,
            "heldout_audit": audit,
            "heldout_manifest_path": "artifacts/v3/data/manifest-v5.json",
            "heldout_asset_root": "artifacts/v3/data/final-300-v5/asset-root",
            "expected_regression": [(f"r{i}", 2) for i in range(266)],
            "heldout_class_counts": audit["class_counts"],
            "heldout_family_counts": audit["family_counts"],
            "dataset_sha256": hashlib.sha256(
                json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        return public, held, regression, manifest

    def test_accepts_complete_paired_evidence(self):
        self.assertTrue(qualify(*self.fixture())["local_gates_passed"])

    def reject(self, mutate):
        args = self.fixture()
        mutate(*args)
        self.assertFalse(qualify(*args)["local_gates_passed"])

    def test_timeout_marked_as_win_is_rejected(self):
        self.reject(lambda p, h, r, m: p["results"][0].update(status="timeout"))

    def test_hash_mismatch_is_rejected(self):
        self.reject(
            lambda p, h, r, m: h["results"][0].update(source_sha256="different")
        )

    def test_split_contamination_is_rejected(self):
        self.reject(
            lambda p, h, r, m: m["dataset"]["assets"].extend(
                [
                    {"group": "same", "pixel_sha256": "p", "split": "train"},
                    {"group": "same", "pixel_sha256": "p", "split": "final"},
                ]
            )
        )

    def test_missing_case_is_rejected(self):
        self.reject(lambda p, h, r, m: p["results"].clear())

    def test_class_omission_is_rejected(self):
        self.reject(lambda p, h, r, m: m["heldout_class_counts"].clear())

    def test_two_million_bytes_is_rejected(self):
        self.reject(lambda p, h, r, m: m.update(bytes=2000000))

    def test_budget_reset_is_rejected(self):
        self.reject(lambda p, h, r, m: p["results"][0].update(budget_reset=True))

    def test_reward_regression_is_rejected(self):
        self.reject(lambda p, h, r, m: r["results"][0].update(reward=-6))

    def test_generated_optimality_claim_is_rejected(self):
        self.reject(
            lambda p, h, r, m: h["results"][0].update(effective_performance=1.0)
        )

    def test_runtime_override_is_rejected(self):
        self.reject(
            lambda p, h, r, m: p["results"][0].update(
                runtime_overrides={"no_tracker": True}
            )
        )

    def test_power_profile_change_invalidates_timing(self):
        self.reject(
            lambda p, h, r, m: p["results"][0].update(
                power_state_before={"epp": "performance"},
                power_state_after={"epp": "power"},
            )
        )

    def test_parent_cold_watchdog_limit_is_enforced(self):
        self.reject(lambda p, h, r, m: p["results"][0].update(strict_cold_seconds=5.1))

    def test_training_layout_cannot_masquerade_as_heldout(self):
        self.reject(
            lambda p, h, r, m: h["results"][0]["job"].update(layout_split="train")
        )

    def test_training_artwork_cannot_masquerade_as_heldout(self):
        self.reject(
            lambda p, h, r, m: h["results"][0]["job"].update(appearance_split="train")
        )

    def test_stale_appearance_manifest_is_rejected(self):
        self.reject(
            lambda p, h, r, m: h["results"][0]["job"].update(
                appearance_manifest_sha256="stale"
            )
        )

    def test_wrong_pixel_hash_is_rejected(self):
        self.reject(
            lambda p, h, r, m: h["results"][0]["job"].update(
                expected_pixels_sha256="wrong"
            )
        )

    def test_structured_truth_cannot_masquerade_as_image_results(self):
        self.reject(lambda p, h, r, m: h["results"][0].update(observation="gridstate"))
