"""Verify deduplicated artifact content, aliases and current submissions."""

import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads(
        (root / "docs/reproducibility/deduplication.json").read_text()
    )
    aliases = 0
    for item in manifest["objects"]:
        target = root / item["canonical_path"]
        if not target.resolve().is_relative_to(root):
            raise ValueError(f"Target outside repository: {target}")
        if target.stat().st_size != item["bytes"] or digest(target) != item["sha256"]:
            raise ValueError(f"Content mismatch: {target}")
        for name in item["aliases"]:
            path = root / name
            if (
                not path.is_symlink()
                or path.readlink().is_absolute()
                or path.resolve() != target.resolve()
            ):
                raise ValueError(f"Invalid relative alias: {path}")
            aliases += 1
    current = json.loads((root / "submissions/manifest.json").read_text())["current"]
    for item in current:
        if digest(root / item["file"]) != item["sha256"]:
            raise ValueError(f"Submission changed: {item['file']}")
    return {
        "objects_verified": len(manifest["objects"]),
        "aliases_verified": aliases,
        "submissions_verified": len(current),
        "duplicate_bytes_removed": manifest["bytes_saved"],
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), indent=2))
