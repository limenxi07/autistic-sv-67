import tempfile
import time
import unittest
from pathlib import Path
from typing import cast

from tools.benchmark_planner_p1 import (
    effective_performance,
    load_agent,
    run_case,
)


class HarnessTests(unittest.TestCase):
    def test_scoring(self):
        for reward, expected in [(-20, 1.0), (-25, 0.5), (-30, 0.0)]:
            self.assertEqual(
                effective_performance(reward, -20, -30, True, "win"), expected
            )
        for status in ("loss", "error", "timeout", "setup_error"):
            self.assertEqual(effective_performance(-20, -20, -30, True, status), 0.0)
        self.assertEqual(effective_performance(-20, -20, -30, False, "win"), 0.0)

    def test_starter_loader_and_evaluator_agree(self):
        from public_levels.catalog import (
            PUBLIC_LEVEL_BUILDERS,
        )
        from utils import evaluate_level

        label = "A0 Basic Movement"
        reference = evaluate_level(
            load_agent("starter"),
            PUBLIC_LEVEL_BUILDERS[label],
            -9,
            cast(int, -13.5),
            turn_limit=150,
            time_limit=5,
        )
        result = run_case("starter", label, 42)
        self.assertEqual(result["status"], "win")
        self.assertEqual(result["reward"], reference["total_reward"])
        self.assertEqual(result["effective_performance"], reference["performance"])
        self.assertEqual(result["turns"], 4)
        rss = result["peak_rss_bytes"]
        assert isinstance(rss, int)
        self.assertGreater(rss, 0)
        source_hash = result["source_sha256"]
        assert isinstance(source_hash, str)
        self.assertEqual(len(source_hash), 64)
        self.assertIsInstance(result["evaluation_seconds"], float)
        self.assertEqual(result["turn_limit"], 150)

    def test_starter_loads_without_a_notebook(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agents").mkdir()
            (root / "agents/starter.py").write_text("class Agent:\n    pass\n")
            with patch("tools.benchmark_planner_p1.ROOT", root):
                self.assertEqual(load_agent("starter").__name__, "Agent")

    def test_starter_rejects_top_level_execution(self):
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "agents").mkdir()
            (root / "agents/starter.py").write_text(
                "raise RuntimeError('must not execute')\nclass Agent:\n    pass\n"
            )
            with patch("tools.benchmark_planner_p1.ROOT", root):
                with self.assertRaises(ValueError):
                    load_agent("starter")

    def test_file_and_module_loaders(self):
        from grid_adventure.step import Action

        self.assertIs(
            load_agent("tests.planner_p1_fixtures:WaitAgent")().step(None),
            Action.WAIT,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.py"
            path.write_text("class Agent:\n    pass\n")
            self.assertEqual(load_agent(str(path)).__name__, "Agent")

    def test_hanging_evaluation_is_terminated(self):
        started = time.perf_counter()
        result = run_case(
            "tests.planner_p1_fixtures:HangAgent",
            "A0 Basic Movement",
            42,
            evaluation_limit=0.15,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["termination"], "local-watchdog")
        self.assertEqual(result["effective_performance"], 0.0)
        self.assertIsNone(result["reward"])
        elapsed = result["evaluation_seconds"]
        assert isinstance(elapsed, float)
        self.assertLess(elapsed, 1.0)
        self.assertLess(time.perf_counter() - started, 32.0)

    def test_hanging_import_is_setup_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hang.py"
            path.write_text("while True: pass\n")
            result = run_case(str(path), "A0 Basic Movement", 42, setup_limit=0.15)
        self.assertEqual(result["status"], "setup_error")
        self.assertIsNone(result["evaluation_seconds"])
        self.assertEqual(result["termination"], "local-watchdog")

    def test_optional_agent_statistics(self):
        result = run_case("agents.planner_p1:Agent", "A0 Basic Movement", 42)
        self.assertEqual(result["status"], "win")
        stats = result["agent_statistics"]
        assert isinstance(stats, dict)
        self.assertEqual(stats["search_calls"], "1")
        self.assertEqual(stats["cache_hits"], "3")
        self.assertFalse(result["profiled"])

    def test_error_and_loss(self):
        for spec, status in [
            ("tests.planner_p1_fixtures:ErrorAgent", "error"),
            ("tests.planner_p1_fixtures:WaitAgent", "loss"),
        ]:
            result = run_case(spec, "A0 Basic Movement", 42)
            self.assertEqual(result["status"], status)
            self.assertEqual(result["effective_performance"], 0.0)
        self.assertEqual(result["turns"], 150)

    def test_mechanic_witnesses(self):
        from grid_adventure.grid import step

        from tests.planner_p1_fixtures import fixture_cases

        for builder, actions in fixture_cases("mechanics").values():
            state = builder()
            for action in actions:
                state = step(state, action)
            self.assertTrue(state.win)


if __name__ == "__main__":
    unittest.main()
