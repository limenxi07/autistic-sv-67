import copy
import hashlib
import unittest
from typing import Any

from agents.vision_v4_bridge import CLASSES
from tools.benchmark_planner_p1 import THREAD_VARS
from tools.vision_v3_data import FAMILIES
from tools.qualify_vision_v4 import assess_fresh, qualify


def fixture():
    import json
    from collections import Counter

    from tools.vision_v4_data import canonical, digest

    source = hashlib.sha256(b"candidate").hexdigest()
    assets = [
        {
            "path": k + ".png",
            "class": k,
            "group": k,
            "pixel_sha256": digest(k.encode()),
            "sha256": digest(k.encode()),
            "split": "final",
        }
        for k in CLASSES
    ]
    layouts: list[dict[str, Any]] = []
    for i in range(100):
        labels = [
            [[k] for k in CLASSES],
            [["wall" if i & (1 << x) else "floor"] for x in range(14)],
        ]
        payload = {"width": 14, "height": 2, "labels": labels}
        layouts.append(
            {
                "index": i,
                "family": FAMILIES[i // 10],
                "seed": i,
                "split": "final",
                "sha256": digest(canonical(payload)),
                "layout": payload,
                "appearance_seeds": [i * 3 + a for a in range(3)],
            }
        )
    contract = {"layouts": {"final": layouts}}
    appearance = {"assets": assets, "layouts": layouts}
    rows = []
    jobs = []
    records = []
    counts = Counter()
    for i in range(300):
        layout = layouts[i // 3]
        labels = layout["layout"]["labels"]
        label = f"{layout['family']}/{layout['sha256'][:12]}/{i}"
        record = {
            "image": f"{i}.png",
            "image_sha256": digest(f"png{i}".encode()),
            "pixels_sha256": digest(f"pixels{i}".encode()),
            "labels": labels,
            "visible_labels": [[c[0] for c in row] for row in labels],
            "width": 14,
            "height": 2,
            "turn": 0,
            "appearance_seed": i,
            "appearance_split": "final",
            "geometry_split": "final",
            "constituents": assets,
            "resolution": 640,
            "manifest_sha256": digest(canonical(appearance)),
            "layout_sha256": layout["sha256"],
            "family": layout["family"],
            "layout_seed": layout["seed"],
            "layout_index": layout["index"],
        }
        job = {
            "label": label,
            "seed": layout["seed"],
            "generated": layout["index"],
            "generator": "v4_optional_sides",
            "multiple_gems": True,
            "layout_split": "final",
            "expected_layout_sha256": layout["sha256"],
            "appearance_manifest": "appearance.json",
            "appearance_manifest_sha256": digest(canonical(appearance)),
            "appearance_split": "final",
            "appearance_seed": i,
            "appearance_sampling": "class_group_variant",
            "resolution": 640,
            "asset_root": "assets",
            "expected_pixels_sha256": record["pixels_sha256"],
            "expected_image_sha256": record["image_sha256"],
        }
        actual = Counter(c[0] for row in labels for c in row)
        counts.update(actual)
        row = {
            "label": label,
            "seed": layout["seed"],
            "job": job,
            "status": "win",
            "win": True,
            "lose": False,
            "reward": -10,
            "source_sha256": source,
            "initial_board_exact": True,
            "initial_confusion": [[k, k, n] for k, n in actual.items()],
            "observation": "image",
            "effective_performance": None,
            "diagnostic_only": False,
            "runtime_overrides": {},
            "budget_reset": False,
            "budget_seconds": [3.0, 3.5],
            "worker_pid": i + 1,
            "fresh_worker": True,
            "vision_library_preloaded": False,
            "vision_preloaded_at_construction": False,
            "watchdog_origin": "before_process_start",
            "cold_watchdog_seconds": 5.0,
            "evaluation_seconds": 1.0,
            "cold_import_to_terminal_seconds": 1.1,
            "strict_cold_seconds": 1.2,
            "peak_rss_bytes": 300000000,
            "torch_threads": [1, 1],
            "entrypoint": "Agent",
            "agent_info": {},
            "power_state_before": {"cpu": 0, "epp": "performance"},
            "power_state_after": {"cpu": 0, "epp": "performance"},
            "provenance": {
                "cpu_affinity": [0],
                "thread_limits": dict.fromkeys(THREAD_VARS, "1"),
            },
        }
        rows.append(row)
        jobs.append(job)
        records.append(record)
    records_sha = digest(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records).encode()
    )
    final_manifest = {"final": {"records_sha256": records_sha}}
    manifest = {
        "source_sha256": source,
        "fresh_jobs": jobs,
        "fresh_class_counts": dict(counts),
        "fresh_family_counts": {k: 30 for k in FAMILIES},
        "bytes": 700000,
        "unseen_art_status": "unproven",
        "training_groups": ["exposed"],
        "exposure": {"exposed": True},
        "fresh_records": records,
        "fresh_appearance_manifest": appearance,
        "fresh_appearance_manifest_path": "appearance.json",
        "fresh_asset_root": "assets",
        "corpus_contract": contract,
        "corpus_final_manifest": final_manifest,
        "data_audit": {
            "passed": True,
            "violations": [],
            "contract_sha256": digest((json.dumps(contract, indent=2) + "\n").encode()),
            "final_manifest_sha256": digest(
                (json.dumps(final_manifest, indent=2) + "\n").encode()
            ),
        },
    }
    return {
        "source_sha256": source,
        "results": rows,
        "diagnostic_only": False,
    }, manifest


class QualificationTests(unittest.TestCase):
    def test_strong_fresh_layout_metrics_never_imply_unseen_art(self):
        report, manifest = fixture()
        self.assertTrue(all(assess_fresh(report, manifest)["checks"].values()))
        result = qualify({"results": []}, {}, report, manifest)
        self.assertEqual(result["unseen_art_status"], "unproven")
        self.assertFalse(result["original_unseen_art_gate_passed"])

    def test_296_exact_boards_rejected(self):
        report, manifest = fixture()
        for row in report["results"][:4]:
            row["initial_board_exact"] = False
        self.assertFalse(
            assess_fresh(report, manifest)["checks"]["fresh_exact_boards_99pct"]
        )

    def test_284_wins_rejected_and_timeout_cannot_count_as_win(self):
        report, manifest = fixture()
        for row in report["results"][:16]:
            row["status"] = "cold_timeout"
        self.assertFalse(assess_fresh(report, manifest)["checks"]["fresh_wins_95pct"])

    def test_98_of_100_gems_and_99_instances_rejected(self):
        report, manifest = fixture()
        for row in report["results"][:6]:
            for cell in row["initial_confusion"]:
                if cell[0] == "gem":
                    cell[1] = "coin"
        self.assertFalse(
            assess_fresh(report, manifest)["checks"]["critical_class_recall_99pct"]
        )
        report, manifest = fixture()
        report["results"][0]["initial_confusion"] = [
            v for v in report["results"][0]["initial_confusion"] if v[0] != "gem"
        ]
        for row in report["results"][1:201]:
            row["initial_confusion"] = [
                v for v in row["initial_confusion"] if v[0] != "gem"
            ]
        manifest["fresh_class_counts"]["gem"] = 99
        self.assertFalse(
            assess_fresh(report, manifest)["checks"]["critical_class_support_100"]
        )

    def test_missing_duplicate_jobs_and_pixel_tampering_rejected(self):
        for mutation in ("missing", "duplicate", "pixel"):
            report, manifest = fixture()
            if mutation == "missing":
                report["results"].pop()
            elif mutation == "duplicate":
                report["results"][-1] = copy.deepcopy(report["results"][0])
            else:
                report["results"][0]["job"] = dict(report["results"][0]["job"])
                report["results"][0]["job"]["expected_pixels_sha256"] = "tampered"
            self.assertFalse(
                assess_fresh(report, manifest)["checks"]["fresh_jobs_bound"]
            )

    def test_diagnostics_overrides_budget_resets_and_2mb_rejected(self):
        for key, value in [
            ("diagnostic_only", True),
            ("runtime_overrides", {"mode": "x"}),
            ("budget_reset", True),
        ]:
            report, manifest = fixture()
            report["results"][0][key] = value
            result = assess_fresh(report, manifest)
            self.assertFalse(all(result["checks"].values()), key)
        report, manifest = fixture()
        manifest["bytes"] = 2000000
        result = qualify({"results": []}, {}, report, manifest)
        self.assertFalse(result["checks"]["source_under_decimal_2mb"])

    def test_exposed_art_cannot_be_declared_untouched(self):
        report, manifest = fixture()
        manifest["untouched_jobs"] = manifest["fresh_jobs"]
        manifest["untouched_groups"] = ["exposed"]
        result = qualify({"results": []}, {}, report, manifest, untouched_art=report)
        self.assertEqual(result["unseen_art_status"], "failed")
        self.assertFalse(result["original_unseen_art_gate_passed"])

    def test_matching_job_tampering_still_rejected_by_frozen_records(self):
        report, manifest = fixture()
        report["results"][0]["job"]["expected_pixels_sha256"] = "tampered-in-both"
        manifest["fresh_jobs"][0]["expected_pixels_sha256"] = "tampered-in-both"
        self.assertFalse(assess_fresh(report, manifest)["checks"]["fresh_jobs_bound"])
