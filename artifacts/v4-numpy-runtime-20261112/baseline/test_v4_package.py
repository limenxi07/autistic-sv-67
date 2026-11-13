import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests import test_v3_package as parent_package
from tools.package_v4 import package

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts/v4/models/selected-export"
if not (EXPORT / "report.json").is_file():
    EXPORT = ROOT / "artifacts/v3/models/selected-export"


class PackageTests(parent_package.PackageTests):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.root = cls.folder / "root"
        (cls.root / "agents").mkdir(parents=True)
        (cls.root / "artifacts/p3/frozen").mkdir(parents=True)
        for p in (ROOT / "agents").glob("v4*.py"):
            shutil.copyfile(p, cls.root / "agents" / p.name)
        shutil.copyfile(
            ROOT / "artifacts/p3/frozen/task1_agent.py",
            cls.root / "artifacts/p3/frozen/task1_agent.py",
        )
        report = json.loads((EXPORT / "report.json").read_text())
        cls.policy = cls.folder / "policy.json"
        from agents.v4_model import POLICY

        cls.policy.write_text(
            json.dumps({**POLICY._asdict(), "model_sha256": report["export_sha256"]})
        )
        cls.source = package(cls.root, EXPORT, cls.policy, cls.folder / "output")

    def test_package_under_decimal_2mb(self):
        self.assertLess(self.source.stat().st_size, 1_600_000)
        ast.parse(self.source.read_text(), feature_version=(3, 10))
        other = package(self.root, EXPORT, self.policy, self.folder / "repeat")
        self.assertEqual(other.read_bytes(), self.source.read_bytes())

    def test_mismatched_policy_model_hash_rejected(self):
        policy = json.loads(self.policy.read_text())
        policy["model_sha256"] = "wrong"
        path = self.folder / "wrong-policy.json"
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "model"):
            package(self.root, EXPORT, path, self.folder / "rejected")

    def test_accidental_v3_import_rejected_before_flattening(self):
        path = self.root / "agents/v4.py"
        original = path.read_bytes()
        try:
            path.write_bytes(
                original + b"\nfrom agents.v3 import PerceptionUncertainty\n"
            )
            with self.assertRaisesRegex(ValueError, "V3"):
                package(self.root, EXPORT, self.policy, self.folder / "rejected-v3")
        finally:
            path.write_bytes(original)

    def test_frozen_backend_bindings_survive_flattening(self):
        from tools.benchmark_p1 import load_agent

        cls = load_agent(str(self.source))
        ns = cls()._step_image.__func__.__globals__
        self.assertIs(ns["reference_state_key"], ns["P3_REFERENCE_STATE_KEY"])
        self.assertTrue(issubclass(ns["PlannerAdapter"], ns["BackendPlannerAdapter"]))
        self.assertIs(type(cls()._structured), ns["StructuredAgent"])
        self.assertIsNot(ns["StructuredAgent"], ns["Task3Agent"])

    def test_notebook_cell_uses_configured_model_and_calibration(self):
        # Notebook cells execute inside a function, unlike module/exec tests.
        # Global rebinding in configure must not leave its consumers capturing
        # the unconfigured cell-local model, temperature or policy.
        worker = r"""
import io, json, sys
from pathlib import Path
import numpy as np
import torch
from grid_adventure.grid import step
from grid_adventure.step import Action
from tests.p1_fixtures import corridor
from tests.v3_fixtures import vision_observation
raw = Path(sys.argv[1]).read_text()
ns = {'__name__': 'unregistered_v4_notebook'}
code = 'def pasted_cell():\n' + ''.join('    ' + s + '\n' for s in raw.splitlines())
code += '    return Agent, classify, current_policy, POLICY\n'
exec(compile(code, '<notebook-cell>', 'exec'), ns)
cls, classify, current_policy, policy = ns['pasted_cell']()
state = corridor()
observations = [vision_observation(state), vision_observation(step(state, Action.RIGHT))]
loads = []
original_load = torch.jit.load
def load(buffer, *args, **kwargs):
    assert isinstance(buffer, io.BytesIO)
    loads.append(True)
    return original_load(buffer, *args, **kwargs)
torch.jit.load = load
a = cls()
actions = [a.step(o) for o in observations]
assert actions == [Action.RIGHT, Action.RIGHT], (actions, a.info())
assert loads == [True], loads
assert current_policy() == policy, (current_policy(), policy)
cell_scores = classify(observations[0]).probabilities.copy()
module = {'__name__': 'unregistered_v4_module'}
exec(compile(raw, '<module>', 'exec'), module)
module['Agent']()
np.testing.assert_array_equal(cell_scores, module['classify'](observations[0]).probabilities)
print(json.dumps({'actions': 2, 'calibration_equal': True, 'notebook_cell': True}))
"""
        result = subprocess.run(
            [sys.executable, "-c", worker, str(self.source)],
            cwd=ROOT,
            env={**os.environ, "MPLCONFIGDIR": "/private/tmp/v4-mpl"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertTrue(json.loads(result.stdout.splitlines()[-1])["notebook_cell"])
