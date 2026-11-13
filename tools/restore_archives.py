"""Restore large archives from verified LFS chunks without replacing files."""

import hashlib
import json
import tempfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "docs/reproducibility/archive-parts.json"
    for record in json.loads(manifest.read_text()):
        target = root / record["path"]
        if target.exists():
            if digest(target) != record["sha256"]:
                raise ValueError(f"Existing archive differs: {target}")
            print(f"Verified {record['path']}")
            continue
        for part in record["parts"]:
            path = root / part["path"]
            if path.stat().st_size != part["bytes"] or digest(path) != part["sha256"]:
                raise ValueError(f"Invalid chunk: {path}")
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=target.name + ".",
            suffix=".restoring",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        try:
            with temporary.open("wb") as output:
                for part in record["parts"]:
                    with (root / part["path"]).open("rb") as source:
                        while block := source.read(8 * 1024 * 1024):
                            output.write(block)
            if digest(temporary) != record["sha256"]:
                raise ValueError(f"Reconstructed hash mismatch: {target}")
            # Exclusive creation prevents silently replacing a concurrent file.
            target.hardlink_to(temporary)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Restored {record['path']}")


if __name__ == "__main__":
    main()
