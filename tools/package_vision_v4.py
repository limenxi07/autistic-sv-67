"""Build a standalone V4 agent from the selected planner, model and policy."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from tools.diagnose_vision_v4 import P3_SHA256, sha256
from tools.package_vision_v3 import inline

MODULES = (
    "vision_v4_types",
    "vision_v4_bridge",
    "vision_v4_model",
    "vision_v4_perception",
    "vision_v4_tracking",
    "vision_v4_search",
    "vision_v4_planner",
    "vision_v4",
)


def _runtime_policy(source, temperature):
    tree = ast.parse(source)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
    if ast.literal_eval(values["TEMPERATURE"]) != temperature:
        raise ValueError("Runtime model calibration mismatch")
    call = values["POLICY"]
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Name)
        or call.func.id != "PerceptionPolicy"
        or len(call.args) != 4
    ):
        raise ValueError("Unrecognized runtime policy record")
    args = [
        temperature
        if isinstance(a, ast.Name) and a.id == "TEMPERATURE"
        else ast.literal_eval(a)
        for a in call.args
    ]
    return dict(zip(("temperature", "min_confidence", "min_margin", "view"), args))


def package(
    source_root: Path,
    export: Path,
    policy: Path,
    output_dir: Path,
    *,
    task3: bool = False,
) -> Path:
    root, export, policy, output_dir = map(
        Path, (source_root, export, policy, output_dir)
    )
    authoritative = root / "artifacts/p3/frozen/task1_agent.py"
    backend = root / "agents/vision_v4_backend.py"
    if (
        sha256(authoritative) != P3_SHA256
        or backend.read_bytes() != authoritative.read_bytes()
    ):
        raise ValueError("V4 backend differs from authoritative frozen P3")
    report = json.loads((export / "report.json").read_text())
    numpy_export = report.get("format") == "numpy-fp32"
    for filename, key in (
        ("model.npz" if numpy_export else "model.pt", "export_sha256"),
        ("model_loader.py", "loader_sha256"),
    ):
        if sha256(export / filename) != report[key]:
            raise ValueError("Export model/loader hash mismatch")
    selected = json.loads(policy.read_text())
    if selected["model_sha256"] != report["export_sha256"]:
        raise ValueError("Policy/model hash mismatch")
    keys = ("temperature", "min_confidence", "min_margin", "view")
    if selected["view"] not in ("full", "inset_005") or selected["temperature"] <= 0:
        raise ValueError("Unsupported selected policy")
    model_source = (root / "agents/vision_v4_model.py").read_text()
    if (export / "model_loader.py").read_text().strip() not in model_source:
        raise ValueError("Runtime model differs from selected export")
    runtime = _runtime_policy(model_source, report["temperature"])
    if runtime != {k: selected[k] for k in keys}:
        raise ValueError("Runtime/selected policy mismatch")
    chunks = [
        "# V4 standalone; planning is conditional on observed visual evidence.\n",
        authoritative.read_text(),
        "\nStructuredAgent = Agent\nBackendPlannerAdapter = PlannerAdapter\nP3_REFERENCE_STATE_KEY = reference_state_key\n",
    ]
    files = [authoritative, backend]
    modules = MODULES + (("boss_v4",) if task3 else ())
    for module in modules:
        path = root / ("agents/" + module + ".py")
        raw = path.read_bytes()
        for node in ast.walk(ast.parse(raw)):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("agents.")
                and node.module
                not in {
                    "agents." + m
                    for m in modules + ("vision_v4_backend",)
                }
            ):
                raise ValueError(
                    "Accidental V3/foreign agent dependency: " + (node.module or "")
                )
        if module == "vision_v4_tracking":
            chunks.append("\nperception_policy = current_policy\n")
        chunks.extend(["\n# ---- " + module + " ----\n", inline(raw)])
        files.append(path)
    if task3:
        chunks.append("\nAgent = BossAgent\n")
    raw = "".join(chunks).encode()
    if raw.count(authoritative.read_bytes()) != 1:
        raise ValueError("Frozen P3 must be inlined exactly once")
    if len(raw) >= 2_000_000:
        raise ValueError("Source must be below2,000,000 decimal bytes")
    tree = ast.parse(raw, feature_version=(3, 10))
    allowed = sys.stdlib_module_names | {
        "grid_adventure",
        "grid_universe",
        "numpy",
        "PIL",
        "torch",
    }
    for node in ast.walk(tree):
        names = (
            [a.name for a in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
        if isinstance(node, ast.ImportFrom) and node.level:
            raise ValueError("Relative runtime dependency")
        if any(name.split(".")[0] not in allowed for name in names):
            raise ValueError("Unsupported runtime import: " + str(names))
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", "")
            )
            if name in {
                "open",
                "read_text",
                "read_bytes",
                "download",
                "urlopen",
                "connect",
                "exec",
                "compile",
                "import_module",
                "__import__",
                "load_state_dict",
            }:
                raise ValueError("External runtime access: " + name)
            if isinstance(node.func, ast.Name) and name == "eval":
                raise ValueError("Dynamic Python evaluation unsupported")
            if name == "load" and ast.unparse(node.func) not in (
                {"torch.jit.load", "np.load"} if numpy_export else {"torch.jit.load"}
            ):
                raise ValueError("Only vetted in-memory TorchScript loading allowed")
            if (
                name == "load"
                and ast.unparse(node.func) == "np.load"
                and (
                    len(node.args) != 1
                    or not isinstance(node.args[0], ast.Call)
                    or ast.unparse(node.args[0].func) != "io.BytesIO"
                    or len(node.keywords) != 1
                    or node.keywords[0].arg != "allow_pickle"
                    or not isinstance(node.keywords[0].value, ast.Constant)
                    or node.keywords[0].value.value is not False
                )
            ):
                raise ValueError(
                    "NumPy loading must use BytesIO with allow_pickle=False"
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("task3_agent.py" if task3 else "task2_agent.py")
    if output.exists() and output.read_bytes() != raw:
        raise ValueError("Refusing to overwrite different frozen candidate bytes")
    output.write_bytes(raw)
    manifest = {
        "source_sha256": sha256(output),
        "bytes": len(raw),
        "backend_sha256": P3_SHA256,
        "model_sha256": report["export_sha256"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "policy_sha256": sha256(policy),
        "policy": {k: selected[k] for k in keys},
        "runtime_sources": {str(p.relative_to(root)): sha256(p) for p in files},
        "configuration": {
            "initial_seconds": 3.0,
            "cutoff_seconds": 3.5,
            "image_execution_cutoff_seconds": 4.65,
            "reference_initial_seconds": 2.0,
            "max_hypotheses": 4,
            "max_ambiguous_cells": 2,
            "default_structured": "P3.Agent",
        },
        "export_parity_passed": report["parity_passed"],
        "runtime_overrides": {},
        "inference_format": report.get("format", "torchscript"),
        "status": "Source freeze only; qualification gates are separate.",
    }
    if task3:
        manifest["configuration"].update(
            {
                "initial_seconds": 11.0,
                "cutoff_seconds": 12.0,
                "image_execution_cutoff_seconds": 14.65,
                "default_structured": "BossStructuredAgent(P3.Task3Agent)",
                "structured_execution_cutoff_seconds": 14.65,
                "task": 3,
                "episode_limit_seconds": 15.0,
            }
        )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output
