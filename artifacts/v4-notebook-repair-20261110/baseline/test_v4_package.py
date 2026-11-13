import ast
import json
import shutil
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
