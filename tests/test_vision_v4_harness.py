import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.benchmark_vision_v4 import replay_job, validate_timing_report


class HarnessTests(unittest.TestCase):
    def test_v4_trace_records_initial_perception_stage(self):
        from tests.vision_v3_fixtures import image_env
        from tests.planner_p1_fixtures import corridor
        from tools.benchmark_vision_v4 import trace_agent
        from tools.benchmark_planner_p1 import load_agent

        selected = json.loads(Path("artifacts/v4/models/candidate.json").read_text())
        env = image_env(corridor())
        observation, _ = env.reset()
        with tempfile.TemporaryDirectory() as d:
            agent = trace_agent(
                load_agent(selected["source"])(), env, {"trace_output": d}
            )
            agent.step(observation)
        stages = agent._v4_trace[0]["stages"]
        self.assertIn("classify_evidence", stages)
        self.assertGreater(stages["classify_evidence"], 0)

    def test_preserves_job_pixel_binding(self):
        report = json.loads(
            Path("artifacts/v3/qualification-reviewed/final-300.json").read_text()
        )
        job = dict(report["results"][0]["job"])
        job["expected_pixels_sha256"] = "tampered"
        from tests.vision_v3_fixtures import image_env
        from tools.benchmark_vision_v4 import recorded_appearance
        from tools.vision_v3_data import new_layout

        state = new_layout(
            job["generated"], split=job["layout_split"], multiple_gems=True
        )[0]
        with self.assertRaisesRegex(ValueError, "decoded pixels"):
            recorded_appearance(image_env(state), job).reset()

    def test_trace_cannot_qualify_as_timing(self):
        with self.assertRaisesRegex(ValueError, "diagnostic"):
            validate_timing_report({"diagnostic_only": True, "results": []})

    def test_replay_preserves_all_job_fields(self):
        job = {"label": "fixture", "seed": 7, "expected_pixels_sha256": "pixels"}
        with (
            tempfile.TemporaryDirectory() as d,
            patch(
                "tools.benchmark_vision_v4.run_case",
                return_value={"status": "loss"},
            ) as run,
        ):
            report = replay_job(
                job,
                Path("artifacts/v3/frozen-reviewed/task2_agent.py"),
                Path(d) / "report.json",
                trace=True,
            )
        self.assertEqual(run.call_args.kwargs["expected_pixels_sha256"], "pixels")
        self.assertEqual(report["results"][0]["job"], job)
        self.assertTrue(report["diagnostic_only"])

    def test_group_balanced_renderer_preserves_pixel_binding(self):
        import hashlib

        from tests.vision_v3_fixtures import image_env
        from tools.benchmark_vision_v4 import recorded_appearance
        from tools.vision_v4_data import (
            canonical,
            digest,
            new_layout,
            render_frame,
        )

        manifest = json.loads(Path("artifacts/v3/data/manifest-v5.json").read_text())
        manifest["assets"] = [{**a, "split": "train"} for a in manifest["assets"]]
        manifest["layouts"] = []
        state, *_ = new_layout(10000, split="train", multiple_gems=True)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            path = out / "manifest.json"
            path.write_text(json.dumps(manifest))
            record = render_frame(
                manifest, "train", state, 123, out / "corpus", resolution=480
            )
            job = {
                "appearance_manifest": str(path),
                "appearance_manifest_sha256": digest(canonical(manifest)),
                "appearance_seed": 123,
                "appearance_split": "train",
                "resolution": 480,
                "appearance_sampling": "class_group_variant",
                "asset_root": str(out / "corpus/asset-root"),
                "expected_pixels_sha256": record["pixels_sha256"],
            }
            obs, _ = recorded_appearance(image_env(state), job).reset()
            self.assertEqual(
                hashlib.sha256(obs["image"].tobytes()).hexdigest(),
                record["pixels_sha256"],
            )

    def test_rejects_reused_workers_and_changed_watchdog_origin(self):
        good = {
            "source_sha256": "source",
            "worker_pid": 1,
            "fresh_worker": True,
            "vision_library_preloaded": False,
            "vision_preloaded_at_construction": False,
            "watchdog_origin": "before_process_start",
            "cold_watchdog_seconds": 5.0,
        }
        for rows in ([good, good], [{**good, "watchdog_origin": "after_agent_init"}]):
            with self.assertRaises(ValueError):
                validate_timing_report({"source_sha256": "source", "results": rows})

    def test_mixed_source_report_rejected(self):
        with self.assertRaisesRegex(ValueError, "source"):
            validate_timing_report(
                {"source_sha256": "a", "results": [{"source_sha256": "b"}]}
            )

    def test_preloaded_vision_library_cannot_qualify(self):
        row = {
            "source_sha256": "source",
            "worker_pid": 1,
            "fresh_worker": True,
            "watchdog_origin": "before_process_start",
            "cold_watchdog_seconds": 5.0,
            "vision_library_preloaded": True,
            "vision_preloaded_at_construction": True,
        }
        with self.assertRaisesRegex(ValueError, "preloaded"):
            validate_timing_report({"source_sha256": "source", "results": [row]})

    def test_supervisor_imports_do_not_preload_torch(self):
        import subprocess
        import sys

        code = "import sys; import tools.vision_v4_selection; import tools.vision_v4_development; assert 'torch' not in sys.modules"
        worker = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(worker.returncode, 0, worker.stderr)
