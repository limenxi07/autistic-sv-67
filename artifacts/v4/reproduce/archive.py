"""Create and stream-verify V4-only supplementary parts (run at repository root)."""

import hashlib
import json
import tarfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def make_part(name, files):
    output = Path("artifacts/v4/supplementary")
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(set(p for p in files if p.is_file() and "__pycache__" not in p.parts))
    manifest = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in files}
    manifest_path = output / (name + "-manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    target = output / (name + ".tar.gz")
    if target.exists():
        raise FileExistsError(target)
    with tarfile.open(target, "w:gz", compresslevel=1, dereference=True) as archive:
        for path in files + [manifest_path]:
            archive.add(path, arcname=str(path), recursive=False)
    return verify_part(target, manifest_path)


def verify_part(target, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    seen = set()
    with tarfile.open(target, "r:gz") as archive:
        for member in archive:
            if member.name == str(manifest_path):
                continue
            if member.name not in manifest or not member.isfile():
                raise ValueError("Unexpected archive member: " + member.name)
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Missing archive file")
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest[member.name]["sha256"]:
                raise ValueError("Archive contents changed: " + member.name)
            seen.add(member.name)
    if seen != set(manifest):
        raise ValueError("Incomplete archive")
    return {"bytes": target.stat().st_size, "sha256": digest(target),
            "files": len(manifest), "content_manifest": str(manifest_path),
            "content_manifest_sha256": digest(manifest_path), "verified": True}


def run(reuse_data=False):
    data = Path("artifacts/v4/data/corpora")
    if not (data / "audit-before-fit.json").is_file():
        raise ValueError("Actual generated corpus and independent audit required")
    batch = Path("artifacts/v4/runs/batch-1")
    if not all((batch / name / "report.json").is_file()
               for name in ("fold-0", "fold-1", "fold-2", "production")):
        raise ValueError("All four actual training run reports required")
    parts = {}
    parts["artifacts/v4/supplementary/generated-datasets.tar.gz"] = (
        verify_part(Path("artifacts/v4/supplementary/generated-datasets.tar.gz"),
                    Path("artifacts/v4/supplementary/generated-datasets-manifest.json"))
        if reuse_data else make_part("generated-datasets", list(data.rglob("*"))))
    # Parent code/fixtures are small and needed to run the full suite. Large
    # unchanged parent datasets remain checksum-bound external parts below.
    sources = [p for folder in ("agents", "tools", "tests", "public_levels") for p in Path(folder).rglob("*.py") if "__pycache__" not in p.parts]
    sources += [Path("pyproject.toml"), Path("uv.lock"), Path("utils.py"), Path("capstone-project.py")]
    for folder in ("artifacts/p1/frozen", "artifacts/p2/p2a-ready", "artifacts/p2/p2b-ready", "artifacts/p3/frozen", "artifacts/v3/frozen-reviewed"):
        sources += [p for p in Path(folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".gz", ".tar")]
    sources += [Path("docs/superpowers/plans/2026-11-09-v4-vision-recovery.md"),
                Path("docs/superpowers/specs/2026-11-09-v4-vision-recovery-design.md")]
    sources += list(Path(".superpowers/sdd/2026-11-09-v4-vision-recovery").glob("*"))
    sources += [p for p in Path("artifacts/v4").rglob("*")
                if p.is_file() and not any(s in p.parts for s in ("supplementary", "corpora", "__pycache__"))
                and (p.suffix not in (".gz", ".tar") or "checkpoints" in p.parts)]
    code_name = "code-models-evidence-final" if reuse_data else "code-models-evidence"
    parts["artifacts/v4/supplementary/" + code_name + ".tar.gz"] = make_part(
        code_name, sources)
    # Parent parts are required inputs, referenced once without duplicating GBs.
    parent_receipt = json.loads(Path("artifacts/v4/qualification/parent-parts-verified.json").read_text())
    if parent_receipt.get("verified") is not True:
        raise ValueError("Local physical parent archive verification required")
    parents = parent_receipt["parts"]
    for name, record in parents.items():
        path = Path(name)
        if path.is_file() and (path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]):
            raise ValueError("Required checksum-bound parent missing/changed: " + name)
    result = {"v4_parts": parts, "required_parent_parts": parents,
              "parent_parts_manifest_sha256": parent_receipt["parent_parts_manifest_sha256"],
              "external_parent_verification": parent_receipt,
              "actual_generated_datasets_included": True,
              "all_four_training_runs_included": True}
    out = Path("artifacts/v4/supplementary/parts.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    import sys

    run(reuse_data="--reuse-data" in sys.argv)
