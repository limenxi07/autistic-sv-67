"""Copy a self-contained Task 1 agent unchanged and describe its exact bytes."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path


def package(source: Path, output_dir: Path) -> Path:
    raw = source.read_bytes()
    if len(raw) >= 2_000_000:
        raise ValueError("Source must be smaller than 2,000,000 bytes")
    tree = ast.parse(raw)
    allowed = sys.stdlib_module_names | {"grid_adventure", "grid_universe"}
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise ValueError("Relative import is not self-contained")
            modules = [node.module]
        for name in modules:
            if name.split(".")[0] not in allowed:
                raise ValueError(f"Unsupported dependency: {name}")
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else getattr(node.func, "attr", "")
            )
            if name in {
                "open",
                "__import__",
                "exec",
                "eval",
                "compile",
                "import_module",
                "read_text",
                "read_bytes",
                "load",
                "loads",
            }:
                raise ValueError(
                    f"Runtime external code/assets access is unsupported: {name}"
                )
    if not any(isinstance(n, ast.ClassDef) and n.name == "Agent" for n in tree.body):
        raise ValueError("Source has no top-level Agent")
    config = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "STATE_CAP",
                    "RECORD_CAP",
                    "PROFILE",
                    "PRODUCTION_MODE",
                    "SEARCH_WEIGHT",
                    "RESOURCE_PROGRESS_WEIGHT",
                    "INEFFECTIVE_PENALTY",
                    "INITIAL_SEARCH_SECONDS",
                    "SEARCH_CUTOFF_SECONDS",
                ):
                    config[target.id] = ast.literal_eval(node.value)
    if config.get("PROFILE"):
        raise ValueError("A profiled build is not a submission candidate")
    packages = {}
    for name in ("grid-adventure", "grid-universe"):
        try:
            packages[name] = version(name)
        except Exception:  # noqa: BLE001
            packages[name] = None
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "configuration": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "host": platform.node(),
            "packages": packages,
        },
        "status": "unqualified local artifact; consult qualification.json and Debugging result",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "task1_agent.py"
    output.write_bytes(raw)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("agents/planner_p1.py")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(package(args.source, args.output_dir))


if __name__ == "__main__":
    main()
