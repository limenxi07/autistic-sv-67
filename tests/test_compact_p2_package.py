import ast
import tempfile
import unittest
from pathlib import Path
from typing import Any

from grid_adventure.step import Action

from tests.planner_p1_fixtures import corridor
from tools.package_planner_p1 import package


class PackageTests(unittest.TestCase):
    def test_standalone_grammar_and_unregistered_grader_prelude(self):
        raw = Path("agents/compact_p2.py").read_bytes()
        ast.parse(raw, feature_version=(3, 10))
        with tempfile.TemporaryDirectory() as d:
            source = package(Path("agents/compact_p2.py"), Path(d))
            self.assertEqual(source.read_bytes(), raw)
            namespace: dict[str, Any] = {"__name__": "unregistered_p2_submission"}
            prelude = "from grid_adventure.grid import GridState\nfrom grid_adventure.env import ImageObservation\n"
            exec(compile(prelude + raw.decode(), str(source), "exec"), namespace)  # noqa: S102
            self.assertIsInstance(namespace["Agent"]().step(corridor()), Action)
