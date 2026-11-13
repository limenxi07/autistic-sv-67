"""Restore verified V4 datasets and saved training runs into a new directory."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_verified(archive, manifest, output, prefixes):
    """Extract only expected regular files; never replace an existing path."""
    output = Path(output).resolve()
    expected = {
        name: item for name, item in manifest.items() if name.startswith(prefixes)
    }
    for name in expected:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"Unsafe archive path: {name}")
    seen = set()
    # GzipFile also handles archives compressed as consecutive gzip members.
    with (
        gzip.open(archive, "rb") as compressed,
        tarfile.open(fileobj=compressed, mode="r|") as tar,
    ):
        for member in tar:
            name = member.name
            if name not in expected:
                continue
            if not member.isfile() or name in seen:
                raise ValueError(f"Not a unique regular file: {name}")
            item = expected[name]
            if member.size != item["bytes"]:
                raise ValueError(f"Size mismatch: {name}")
            target = output / name
            if not target.resolve().is_relative_to(output):
                raise ValueError(f"Path escapes output: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(f"Missing file payload: {name}")
            with source, target.open("xb") as destination:
                while block := source.read(8 * 1024 * 1024):
                    destination.write(block)
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Content mismatch: {name}")
            seen.add(name)
    missing = expected.keys() - seen
    if missing:
        raise ValueError(f"Missing {len(missing)} expected archive members")
    return len(seen)


def restore(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            "Choose a new output directory; saved results are never overwritten"
        )
    base = ROOT / "artifacts/v4/supplementary"
    parts = json.loads((base / "parts.json").read_text())["v4_parts"]
    chosen = (
        ("generated-datasets.tar.gz", ("artifacts/v4/data/corpora/",)),
        ("code-models-evidence-final.tar.gz", ("artifacts/v4/runs/batch-1/",)),
    )
    bundles = []
    for filename, prefixes in chosen:
        archive = base / filename
        if not archive.is_file():
            raise FileNotFoundError(
                f"{archive}: run python tools/restore_archives.py first"
            )
        receipt = parts[str(archive.relative_to(ROOT))]
        manifest_path = ROOT / receipt["content_manifest"]
        if (
            archive.stat().st_size != receipt["bytes"]
            or sha256(archive) != receipt["sha256"]
        ):
            raise ValueError(f"Archive hash mismatch: {archive}")
        if sha256(manifest_path) != receipt["content_manifest_sha256"]:
            raise ValueError(f"Manifest hash mismatch: {manifest_path}")
        bundles.append((archive, json.loads(manifest_path.read_text()), prefixes))
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for archive, manifest, prefixes in bundles:
        count = extract_verified(archive, manifest, output, prefixes)
        results.append(
            {"archive": str(archive.relative_to(ROOT)), "files_verified": count}
        )
        print(f"Restored and verified {count} files from {archive.name}", flush=True)
    (output / "restoration.json").write_text(
        json.dumps({"complete": True, "bundles": results}, indent=2) + "\n"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    restore(args.output)
