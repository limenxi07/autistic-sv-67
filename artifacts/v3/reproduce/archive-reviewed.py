"""Create the reviewed code/model/evidence supplementary part by explicit scope.

The actual 200k training/validation data and final-v5 data are separate immutable
archives. Only declared project paths are included; user configuration is excluded.
"""

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/v3/supplementary"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    paths = set()
    for pattern in (
        "agents/v3*.py",
        "tools/*v3*.py",
        "tests/test_v3*.py",
        "artifacts/v3/runs/**/*",
        "artifacts/v3/models/selected-export/*",
        "artifacts/v3/models/pilot-source/*",
        "artifacts/v3/models/*.txt",
        "artifacts/v3/models/*.sh",
        "artifacts/v3/models/*.json",
        "artifacts/v3/models/*.log",
        "artifacts/v3/bridge/*.json",
        "artifacts/v3/validation/*.json",
        "artifacts/v3/qualification-reviewed/*",
        "artifacts/v3/qualification/*.json",
        "artifacts/v3/qualification-v2/*.json",
        "artifacts/v3/frozen*/*",
        "artifacts/v3/reproduce/*.py",
        "artifacts/v3/reproduce/*.sh",
        "public_levels/*.py",
        "docs/superpowers/*/2026-11-09-v3-vision*.md",
    ):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    for name in (
        "agents/__init__.py",
        "tests/__init__.py",
        "agents/p2_next.py",
        "agents/p1.py",
        "agents/p2.py",
        "tools/benchmark_p1.py",
        "tools/benchmark_boss.py",
        "tests/p1_fixtures.py",
        "tests/p2_fixtures.py",
        "tests/test_p1_state.py",
        "tests/v3_fixtures.py",
        "utils.py",
        "capstone-project.py",
        "pyproject.toml",
        "uv.lock",
        "artifacts/v3/data/freeze-final-v3.py",
        "artifacts/v3/data/freeze-final-v4.py",
        "artifacts/v3/data/freeze-final-v5.py",
        "artifacts/v3/data/manifest-v2.json",
        "artifacts/v3/data/manifest-v3.json",
        "artifacts/v3/data/manifest-v4.json",
        "artifacts/v3/data/manifest-v5.json",
        "artifacts/v3/models/feature_baseline.py",
        "artifacts/v3/qualification/b10-initial.png",
        "artifacts/v3/qualification/public-jobs.json",
        "artifacts/v3/qualification/regression-jobs.json",
        "artifacts/v3/qualification/structured-baseline.json",
        "artifacts/v3/HANDOFF.md",
    ):
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(name)
        paths.add(path)
    manifest = {
        str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in sorted(paths)
    }
    OUT.mkdir(exist_ok=True)
    manifest_path = OUT / "code-models-reviewed-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive_path = OUT / "code-models-reviewed.tar.gz"
    with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths | {manifest_path}):
            archive.add(path, arcname=str(path.relative_to(ROOT)))
    # Read back every member and verify its payload, not just the outer gzip hash.
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            stream = archive.extractfile(member)
            assert stream is not None, member.name
            if member.name in manifest:
                receipt = manifest[member.name]
                assert member.size == receipt["bytes"], member.name
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest()
                    == receipt["sha256"]
                ), member.name
    parts = [
        archive_path,
        ROOT / "artifacts/v3/checkpoints/task2-datasets-v2.tar",
        ROOT / "artifacts/v3/checkpoints/final-datasets-v5.tar.gz",
    ]
    receipts = {
        str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
        for p in parts
    }
    for part in parts[1:]:
        expected = Path(str(part) + ".sha256").read_text().split()[0]
        assert receipts[str(part.relative_to(ROOT))]["sha256"] == expected, part
    (OUT / "reviewed-parts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    (OUT / "SHA256SUMS-reviewed").write_text(
        "".join(f"{r['sha256']}  {name}\n" for name, r in receipts.items())
    )
    print(json.dumps({"archived_files": len(paths), "parts": receipts}, indent=2))


if __name__ == "__main__":
    main()
