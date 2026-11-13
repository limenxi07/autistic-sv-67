import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tools.package_planner_p1 import package

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_exact_bytes_hash_and_isolated_runtime(self):
        source = ROOT / "agents/planner_p1.py"
        with tempfile.TemporaryDirectory() as directory:
            out = package(source, Path(directory))
            self.assertEqual(source.read_bytes(), out.read_bytes())
            manifest = json.loads((out.parent / "manifest.json").read_text())
            self.assertEqual(
                manifest["source_sha256"], hashlib.sha256(out.read_bytes()).hexdigest()
            )
            self.assertLess(out.stat().st_size, 2_000_000)
            script = """
import importlib.util
from grid_adventure.grid import GridState
from grid_adventure.step import Action
from grid_adventure.entities import AgentEntity,ExitEntity
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
import sys
spec=importlib.util.spec_from_file_location('candidate',sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
s=GridState(width=3,height=1,movement=MOVEMENTS['cardinal'],objective=OBJECTIVES['collect_gems_and_exit'],turn_limit=150,step_cost=3)
s.add((0,0),AgentEntity());s.add((2,0),ExitEntity())
assert isinstance(m.Agent().step(s),Action)
"""
            result = subprocess.run(
                [sys.executable, "-I", "-c", script, str(out)],
                cwd=directory,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_reject_non_self_contained_imports(self):
        for module in (
            "tests",
            "public_levels.catalog",
            "tools.benchmark_planner_p1",
            "agents.helper",
        ):
            with (
                self.subTest(module=module),
                tempfile.TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "bad.py"
                path.write_text(f"import {module}\nclass Agent: pass\n")
                with self.assertRaises(ValueError):
                    package(path, Path(directory) / "out")

    def test_reject_asset_access_and_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.py"
            for text in ("class Agent: pass\nopen('model.bin')\n", "#" * 2_000_000):
                source.write_text(text)
                with self.assertRaises(ValueError):
                    package(source, Path(directory) / "out")

    def test_python_310_syntax(self):
        import ast

        ast.parse(
            (ROOT / "agents/planner_p1.py").read_text(),
            feature_version=(3, 10),
        )

    def test_grader_exec_without_module_registration(self):
        source = (ROOT / "agents/planner_p1.py").read_text()
        import __future__

        for flags in (0, __future__.annotations.compiler_flag):
            namespace: dict[str, Any] = {"__name__": "student_unregistered"}
            compiled = compile(source, "<student>", "exec", flags=flags)
            exec(compiled, namespace)  # noqa: S102
            self.assertTrue(callable(namespace["Agent"]))

    def test_grader_prelude_and_real_step(self):
        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import corridor

        source = (ROOT / "agents/planner_p1.py").read_text()
        namespace: dict[str, Any] = {"__name__": "grader_student_unregistered"}
        exec(compile("import math\n" + source, "<student>", "exec"), namespace)  # noqa: S102
        self.assertIsInstance(namespace["Agent"]().step(corridor()), Action)
