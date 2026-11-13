"""Build a standalone V3 agent from its pinned planner and CPU model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

BACKEND_SHA256 = "6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd"
MODULES = (
    "vision_v3_types",
    "vision_v3_bridge",
    "vision_v3_model",
    "vision_v3_perception",
    "vision_v3_tracking",
    "vision_v3_planner",
    "vision_v3",
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def inline(raw):
    lines = raw.decode().splitlines(keepends=True)
    for node in ast.parse(raw).body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in (
            "agents",
            "__future__",
        ):
            for i in range(node.lineno - 1, node.end_lineno):
                lines[i] = "\n"
    return "".join(lines)


def package(source_root: Path, export: Path, output_dir: Path) -> Path:
    root, export, output_dir = Path(source_root), Path(export), Path(output_dir)
    parent = root / "agents/boss_p2_next.py"
    if sha256(parent) != BACKEND_SHA256:
        raise ValueError("P2-next source differs from the selected frozen backend")
    export_report = json.loads((export / "report.json").read_text())
    for filename, key in (
        ("model.pt", "export_sha256"),
        ("model_loader.py", "loader_sha256"),
    ):
        if sha256(export / filename) != export_report[key]:
            raise ValueError("Selected export hash mismatch: " + filename)
    runtime_loader = (root / "agents/vision_v3_model.py").read_text()
    if (export / "model_loader.py").read_text().strip() not in runtime_loader:
        raise ValueError("Runtime model differs from the selected export")
    if "TEMPERATURE = " + str(export_report["temperature"]) not in runtime_loader:
        raise ValueError("Runtime calibration mismatch")
    chunks = [
        "# V3 standalone; all planning remains conditional on image perception.\n",
        parent.read_text(),
        "\nStructuredAgent = Agent\nBackendPlannerAdapter = PlannerAdapter\n",
    ]
    files = [parent]
    for module in MODULES:
        path = root / ("agents/" + module + ".py")
        chunks.extend(["\n# ---- " + module + " ----\n", inline(path.read_bytes())])
        files.append(path)
    raw = "".join(chunks).encode()
    if len(raw) >= 2_000_000:
        raise ValueError("Source must be smaller than 2,000,000 decimal bytes")
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
            raise ValueError("Unsupported runtime dependency: " + str(names))
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
                raise ValueError("Dynamic Python evaluation is unsupported")
            if name == "load" and not (
                isinstance(node.func, ast.Attribute)
                and ast.unparse(node.func) == "torch.jit.load"
            ):
                raise ValueError(
                    "Only the vetted in-memory TorchScript loader is allowed"
                )
    data_archive = root / "artifacts/v3/checkpoints/task2-datasets-v2.tar"
    required = [
        root / "tools/vision_v3_data.py",
        root / "tools/train_vision_v3.py",
        root / "tools/export_vision_v3.py",
        root / "tools/vision_v3_wandb.py",
        root / "artifacts/v3/data/manifest-v2.json",
        root / "artifacts/v3/data/manifest-v5.json",
        root / "artifacts/v3/data/final-300-v5/audit.json",
        root / "artifacts/v3/checkpoints/final-datasets-v5.tar.gz",
        root / "artifacts/v3/models/selection.json",
        root / "artifacts/v3/runs/full-compile-aug-seed20261111/best.pt",
        root / "artifacts/v3/runs/full-compile-aug-seed20261111/config.json",
        data_archive,
    ]
    if any(not p.is_file() for p in required):
        raise ValueError(
            "Missing required supplementary files: "
            + str([str(p) for p in required if not p.is_file()])
        )
    supplementary = {
        str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in required
    }
    manifest = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "backend_sha256": BACKEND_SHA256,
        "model_sha256": export_report["export_sha256"],
        "checkpoint_sha256": export_report["checkpoint_sha256"],
        "temperature": export_report["temperature"],
        "configuration": {
            "planner_mode": "p2a",
            "initial_seconds": 3.0,
            "cutoff_seconds": 3.5,
            "reference_initial_seconds": 2.0,
            "gem_shield_shape_inset": 0.05,
            "observed_score_reconciliation": True,
        },
        "runtime_sources": {str(p.relative_to(root)): sha256(p) for p in files},
        "export_parity_passed": export_report["parity_passed"],
        "status": "Frozen source only; consult qualification.json for measured acceptance.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "task2_agent.py"
    output.write_bytes(raw)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "supplementary-manifest.json").write_text(
        json.dumps(supplementary, indent=2) + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument(
        "--export", type=Path, default=Path("artifacts/v3/models/selected-export")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(package(args.source_root, args.export, args.output_dir))


if __name__ == "__main__":
    main()
