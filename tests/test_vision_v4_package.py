import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tests import test_vision_v3_package as parent_package
from tools.package_vision_v4 import package

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "artifacts/v4-search-recovery-20261113/export"
if not (EXPORT / "report.json").is_file():
    EXPORT = ROOT / "artifacts/v3/models/selected-export"


class PackageTests(parent_package.PackageTests):
    def run_worker(self, mode):
        worker = r"""
import builtins, io, json, socket, sys
from pathlib import Path
import numpy as np
from grid_adventure.grid import GridState, step
from grid_adventure.env import ImageObservation
from grid_adventure.step import Action
from tests.planner_p1_fixtures import corridor
from tests.vision_v3_fixtures import vision_observation
state = corridor()
observations = [vision_observation(state), vision_observation(step(state, Action.RIGHT))] if sys.argv[2] == 'image' else []
code = compile(Path(sys.argv[1]).read_text(), '<submission>', 'exec')
original_import = builtins.__import__
def checked_import(name, *args, **kwargs):
    if name.split('.')[0] in ('agents', 'tools', 'tests', 'torch'):
        raise AssertionError('runtime dependency: ' + name)
    return original_import(name, *args, **kwargs)
def denied(*args, **kwargs):
    raise AssertionError('external file or network access')
builtins.__import__ = checked_import
builtins.open = io.open = denied
socket.socket = socket.create_connection = denied
loads = []
original_load = np.load
def memory_load(buffer, *args, **kwargs):
    assert isinstance(buffer, io.BytesIO)
    assert kwargs['allow_pickle'] is False
    loads.append(True)
    return original_load(buffer, *args, **kwargs)
np.load = memory_load
ns = {'__name__': 'unregistered_v4'}
exec(code, ns)
a = ns['Agent']()
if sys.argv[2] == 'image':
    assert [a.step(o) for o in observations] == [Action.RIGHT, Action.RIGHT], a.info()
    assert loads == [True], loads
else:
    assert a.step(state) == Action.RIGHT
    assert loads == []
assert 'torch' not in sys.modules
print(json.dumps({'embedded': True, 'unregistered': True, 'actions': 2, 'torch_loaded': False}))
"""
        result = subprocess.run(
            [sys.executable, "-c", worker, str(self.source), mode],
            cwd=ROOT,
            env={**os.environ, "MPLCONFIGDIR": "/private/tmp/v4-mpl"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout.splitlines()[-1])

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.folder = Path(cls.temporary.name)
        cls.root = cls.folder / "root"
        (cls.root / "agents").mkdir(parents=True)
        (cls.root / "artifacts/p3/frozen").mkdir(parents=True)
        for p in list((ROOT / "agents").glob("vision_v4*.py")) + [
            ROOT / "agents/boss_v4.py"
        ]:
            shutil.copyfile(p, cls.root / "agents" / p.name)
        shutil.copyfile(
            ROOT / "artifacts/p3/frozen/task1_agent.py",
            cls.root / "artifacts/p3/frozen/task1_agent.py",
        )
        report = json.loads((EXPORT / "report.json").read_text())
        cls.policy = cls.folder / "policy.json"
        from agents.vision_v4_model import POLICY

        cls.policy.write_text(
            json.dumps({**POLICY._asdict(), "model_sha256": report["export_sha256"]})
        )
        cls.source = package(cls.root, EXPORT, cls.policy, cls.folder / "output")

    def test_package_under_decimal_2mb(self):
        self.assertLess(self.source.stat().st_size, 1_600_000)
        ast.parse(self.source.read_text(), feature_version=(3, 10))
        other = package(self.root, EXPORT, self.policy, self.folder / "repeat")
        self.assertEqual(other.read_bytes(), self.source.read_bytes())

    def test_only_one_active_cached_model_definition(self):
        source = self.source.read_text()
        self.assertEqual(source.count("class _NumpyCNN:"), 1)
        ns = {}
        exec(source, ns)  # noqa: S102
        self.assertTrue(hasattr(ns["get_model"](), "_cache"))

    def test_boss_package_can_plan_after_task2_deadline_in_both_modes(self):
        from unittest.mock import patch

        from grid_adventure.grid import step
        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import corridor
        from tests.vision_v3_fixtures import vision_observation

        source = package(
            self.root, EXPORT, self.policy, self.folder / "boss", task3=True
        )
        state = corridor()
        images = [
            vision_observation(state),
            vision_observation(step(state, Action.RIGHT)),
        ]
        # Also exercise the function-scoped notebook cell used for submission.
        code = (
            "def cell():\n"
            + "\n".join("    " + line for line in source.read_text().splitlines())
            + "\n    return Agent\n"
        )
        for mode in ("gridstate", "image"):
            with self.subTest(mode=mode):
                now = [100.0]
                with patch("time.perf_counter", side_effect=lambda now=now: now[0]):
                    namespace = {}
                    exec(code, namespace)  # noqa: S102
                    agent = namespace["cell"]()()
                    original_budget = tuple(agent._budget)
                    now[0] = 106.0
                    current = state
                    for index in range(2):
                        action = agent.step(
                            current if mode == "gridstate" else images[index]
                        )
                        self.assertEqual(action, Action.RIGHT)
                        current = step(current, action)
                    self.assertTrue(current.win)
                    self.assertEqual(tuple(agent._budget), original_budget)
                    now[0] = 115.0
                    self.assertEqual(
                        agent.step(state if mode == "gridstate" else images[0]),
                        Action.WAIT,
                    )

    def test_boss_reference_fallback_keeps_longer_original_deadline(self):
        from unittest.mock import patch

        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import corridor
        from tests.vision_v3_fixtures import vision_observation

        source = package(
            self.root, EXPORT, self.policy, self.folder / "boss", task3=True
        )
        state = corridor()
        observation = vision_observation(state)
        for mode in ("gridstate", "image"):
            with self.subTest(mode=mode):
                now = [100.0]
                with patch("time.perf_counter", side_effect=lambda now=now: now[0]):
                    namespace = {}
                    exec(source.read_text(), namespace)  # noqa: S102
                    agent = namespace["Agent"]()
                    now[0] = 106.0
                    with patch.dict(
                        namespace,
                        {"encode": lambda *_, ns=namespace: self._unsupported(ns)},
                    ):
                        self.assertEqual(
                            agent.step(state if mode == "gridstate" else observation),
                            Action.RIGHT,
                        )
                    planner = (
                        agent._structured if mode == "gridstate" else agent._planner
                    )
                    self.assertIsNotNone(planner._reference)
                    now[0] = 115.0
                    self.assertEqual(
                        agent.step(state if mode == "gridstate" else observation),
                        Action.WAIT,
                    )

    @staticmethod
    def _unsupported(namespace):
        raise namespace["UnsupportedState"]("test unsupported schema")

    def test_mismatched_policy_model_hash_rejected(self):
        policy = json.loads(self.policy.read_text())
        policy["model_sha256"] = "wrong"
        path = self.folder / "wrong-policy.json"
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "model"):
            package(self.root, EXPORT, path, self.folder / "rejected")

    def test_accidental_v3_import_rejected_before_flattening(self):
        path = self.root / "agents/vision_v4.py"
        original = path.read_bytes()
        try:
            path.write_bytes(
                original + b"\nfrom agents.vision_v3 import PerceptionUncertainty\n"
            )
            with self.assertRaisesRegex(ValueError, "V3"):
                package(self.root, EXPORT, self.policy, self.folder / "rejected-v3")
        finally:
            path.write_bytes(original)

    def test_numpy_load_requires_memory_and_disables_pickle(self):
        path = self.root / "agents/vision_v4.py"
        original = path.read_bytes()
        for index, call in enumerate(
            (
                "np.load('/tmp/model.npz', allow_pickle=False)",
                "np.load(io.BytesIO(b''), allow_pickle=True)",
            )
        ):
            with self.subTest(call=call):
                try:
                    path.write_bytes(
                        original
                        + ("\ndef unsafe_loader():\n    return " + call + "\n").encode()
                    )
                    with self.assertRaisesRegex(ValueError, "NumPy loading"):
                        package(
                            self.root,
                            EXPORT,
                            self.policy,
                            self.folder / f"unsafe-numpy-{index}",
                        )
                finally:
                    path.write_bytes(original)

    def test_frozen_backend_bindings_survive_flattening(self):
        from tools.benchmark_planner_p1 import load_agent

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
from grid_adventure.grid import step
from grid_adventure.step import Action
from tests.planner_p1_fixtures import corridor
from tests.vision_v3_fixtures import vision_observation
raw = Path(sys.argv[1]).read_text()
ns = {'__name__': 'unregistered_v4_notebook'}
code = 'def pasted_cell():\n' + ''.join('    ' + s + '\n' for s in raw.splitlines())
code += '    return Agent, classify, current_policy, POLICY\n'
exec(compile(code, '<notebook-cell>', 'exec'), ns)
cls, classify, current_policy, policy = ns['pasted_cell']()
state = corridor()
observations = [vision_observation(state), vision_observation(step(state, Action.RIGHT))]
loads = []
original_load = np.load
def load(buffer, *args, **kwargs):
    assert isinstance(buffer, io.BytesIO)
    loads.append(True)
    return original_load(buffer, *args, **kwargs)
np.load = load
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

    def test_v4_observation_optimization_is_wired_without_rebinding_frozen_p3(self):
        from textwrap import indent

        from tests.planner_p1_fixtures import corridor

        suffix = """
original_v4_observe = v4_observe_key
original_frozen_observe = observe_key
calls = {"v4": 0, "frozen": 0}
def v4_observe_key(*args):
    calls["v4"] += 1
    return original_v4_observe(*args)
def observe_key(*args):
    calls["frozen"] += 1
    return original_frozen_observe(*args)
"""
        source = self.source.read_text() + suffix
        for notebook in (False, True):
            namespace = {}
            if notebook:
                source_cell = (
                    "def cell():\n"
                    + indent(source, "    ")
                    + "\n    return PlannerAdapter, StructuredAgent, EpisodeBudget, calls, perf_counter\n"
                )
                exec(source_cell, namespace)  # noqa: S102
                planner_class, structured_class, budget_class, calls, clock = namespace[
                    "cell"
                ]()
            else:
                exec(source, namespace)  # noqa: S102
                planner_class, structured_class, budget_class, calls, clock = (
                    namespace[k]
                    for k in (
                        "PlannerAdapter",
                        "StructuredAgent",
                        "EpisodeBudget",
                        "calls",
                        "perf_counter",
                    )
                )
            now = clock()
            planner_class(budget_class(now, now + 10, now + 10)).step(corridor())
            self.assertGreater(calls["v4"], 0)
            self.assertEqual(calls["frozen"], 0)
            calls.update(v4=0, frozen=0)
            structured_class().step(corridor())
            self.assertEqual(calls["v4"], 0)
            self.assertGreater(calls["frozen"], 0)
