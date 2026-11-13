"""Read original source bytes for historical receipts after a module rename."""

import json
import subprocess
from pathlib import Path


def historical_bytes(path, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]).resolve()
    path = Path(path)
    relative = str(path.relative_to(root)) if path.is_absolute() else str(path)
    ledger = root / "docs/rename-map.json"
    if ledger.exists():
        mapping = json.loads(ledger.read_text())
        originals = {entry["old_path"] for entry in mapping["renames"]}
        inventory = root / "docs/reproducibility/original-inventory.json"
        originals.update(
            entry["path"]
            for entry in json.loads(inventory.read_text())
            if entry["path"].startswith(("agents/", "tools/", "tests/"))
        )
        if relative in originals:
            return subprocess.check_output(
                ["git", "show", f"{mapping['source_snapshot_commit']}:{relative}"],
                cwd=root,
            )
    return (root / relative).read_bytes()
