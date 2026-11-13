"""Hash-bound V4 baseline extraction and evaluator-only diagnostic replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P3_SHA256 = "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4"
V3_SOURCE = Path("artifacts/v3/frozen-reviewed/task2_agent.py")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_regressions(report: dict) -> dict:
    rows = report["results"]
    lost = [r["job"] for r in rows if r["status"] != "win"]
    wrong = [r["job"] for r in rows if not r["initial_board_exact"]]
    unique = {json.dumps(j, sort_keys=True): j for j in lost + wrong}
    return {
        "lost_jobs": lost,
        "incorrect_initial_jobs": wrong,
        "diagnostic_jobs": list(unique.values()),
        "baseline_source_sha256": report["source_sha256"],
    }


def verify_baseline():
    from tools.historical_sources import historical_bytes

    manifest = json.loads(
        (
            ROOT / "artifacts/v3/qualification-reviewed/acceptance-manifest.json"
        ).read_text()
    )
    for name, expected in manifest["runtime_sources"].items():
        if hashlib.sha256(historical_bytes(name, ROOT)).hexdigest() != expected:
            raise ValueError("Changed protected runtime: " + name)
    if sha256(ROOT / V3_SOURCE) != manifest["source_sha256"]:
        raise ValueError("Changed V3 frozen executable")
    hierarchical_p3 = ROOT / "artifacts/p3/frozen/task1_agent.py"
    if (
        sha256(hierarchical_p3) != P3_SHA256
        or hierarchical_p3.stat().st_size != 96112
    ):
        raise ValueError("Changed P3 frozen executable")
    # sha256sum's paths are repository-relative in this frozen receipt.
    for line in (
        (hierarchical_p3.parent / "CHECKSUMS.sha256").read_text().splitlines()
    ):
        expected, name = line.split(maxsplit=1)
        path = hierarchical_p3.parent / name.lstrip("*")
        if not path.exists():
            path = ROOT / name.lstrip("*")
        if sha256(path) != expected:
            raise ValueError("Changed P3 receipt: " + str(path))
    return {
        "v3_runtime_sources": manifest["runtime_sources"],
        "v3_source_sha256": manifest["source_sha256"],
        "p3_sha256": P3_SHA256,
        "p3_bytes": hierarchical_p3.stat().st_size,
        "p3_default_rows": 266,
        "p3_task3_rows": 60,
        "p3_platform_report": json.loads(
            (hierarchical_p3.parent / "platform-validation.json").read_text()
        ),
        "evidence": "Saved P3 default Agent public66+generated200; Task3Agent stress36+holdout18+boss6 are longer-budget only.",
    }


def prepare_baseline():
    output = ROOT / "artifacts/v4/baseline"
    output.mkdir(parents=True, exist_ok=True)
    baseline = verify_baseline()
    (output / "references.json").write_text(json.dumps(baseline, indent=2) + "\n")
    report = json.loads(
        (ROOT / "artifacts/v3/qualification-reviewed/final-300.json").read_text()
    )
    jobs = extract_regressions(report)
    (output / "regressions.json").write_text(json.dumps(jobs, indent=2) + "\n")
    protected = [
        p
        for prefix in (
            "agents/planner_p1",
            "agents/compact_p2",
            "agents/boss_p2_next",
            "agents/hierarchical_p3",
            "agents/vision_v3",
            "tools/",
            "tests/",
            "artifacts/p1/",
            "artifacts/p2/",
            "artifacts/p3/",
            "artifacts/v3/",
        )
        for p in ROOT.glob(prefix + "*")
        if p.is_file()
    ]
    # Recurse artifact directories; omit interpreter caches, which are not evidence.
    protected += [
        p
        for folder in ("artifacts/p1", "artifacts/p2", "artifacts/p3", "artifacts/v3")
        for p in (ROOT / folder).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    ]
    protected += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    inventory = {
        str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(set(protected))
        if not p.name.startswith(
            (
                "test_v4",
                "benchmark_vision_v4",
                "diagnose_vision_v4",
            )
        )
    }
    (output / "protected-files.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return jobs


def package_p3_control(output: Path) -> Path:
    """Unrepaired V3 logic/model, exact P3 backend; independent immutable control."""
    from tools.package_vision_v3 import MODULES, inline

    verify_baseline()
    backend = ROOT / "artifacts/p3/frozen/task1_agent.py"
    chunks = [
        "# Diagnostic V3-on-P3 control; unchanged image logic.\n",
        backend.read_text(),
        "\nStructuredAgent = Agent\nBackendPlannerAdapter = PlannerAdapter\nP3_REFERENCE_STATE_KEY = reference_state_key\n",
    ]
    components = {str(backend.relative_to(ROOT)): sha256(backend)}
    for module in MODULES:
        path = ROOT / ("agents/" + module + ".py")
        raw = path.read_bytes()
        if module == "vision_v3_planner":
            raw = raw.replace(
                b"6c668eaa24acf234d7188cb0606203e9af1ece9a3323b980adab75d50f5a30fd",
                P3_SHA256.encode(),
            )
            raw = raw.replace(
                b"agents/boss_p2_next.py",
                b"artifacts/p3/frozen/task1_agent.py",
            )
        chunks.extend(["\n# ---- " + module + " ----\n", inline(raw)])
        components[str(path.relative_to(ROOT))] = sha256(path)
    raw = "".join(chunks).encode()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "task2_agent.py"
    target.write_bytes(raw)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(target),
                "bytes": len(raw),
                "backend_sha256": P3_SHA256,
                "components": components,
                "changes_from_v3": [
                    "exact P3 backend and adapter metadata wiring only"
                ],
                "diagnostic_instrumentation": False,
                "tracker": "unchanged V3",
                "perception": "unchanged V3",
            },
            indent=2,
        )
        + "\n"
    )
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--source", type=Path, default=V3_SOURCE)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/v4/diagnostics/v3")
    )
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        jobs = prepare_baseline()
        print({k: len(v) for k, v in jobs.items() if isinstance(v, list)})
        return
    from tools.benchmark_vision_v4 import replay_job, run_case

    jobs = json.loads(Path("artifacts/v4/baseline/regressions.json").read_text())
    for i, job in enumerate(jobs["diagnostic_jobs"]):
        output = args.output / f"{i:02d}.json"
        report = replay_job(job, args.source, output, trace=True)
        print(i, job["label"], report["results"][0]["status"], flush=True)
        if args.oracle and job in jobs["lost_jobs"]:
            spec = dict(job)
            label = spec.pop("label")
            row = run_case(
                label,
                source=str(args.source),
                observation="oracle",
                cold_limit=30,
                **spec,
            )
            row.update(job=job, diagnostic_only=True, evaluator_oracle=True)
            output.with_name(f"{i:02d}-oracle.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()


def package_recovery_control(output: Path) -> Path:
    """Development-only recovery build retaining the exact V3 model/policy."""
    from tools.package_vision_v3 import inline

    verify_baseline()
    backend = ROOT / "artifacts/p3/frozen/task1_agent.py"
    if sha256(ROOT / "agents/vision_v4_backend.py") != P3_SHA256:
        raise ValueError("V4 backend differs from frozen P3")
    chunks = [
        "# V4 recovery-only development control; V3 perception.\n",
        backend.read_text(),
        "\nStructuredAgent = Agent\nBackendPlannerAdapter = PlannerAdapter\nP3_REFERENCE_STATE_KEY = reference_state_key\n",
    ]
    for module in (
        "vision_v4_types",
        "vision_v4_bridge",
        "vision_v4_model",
        "vision_v4_perception",
        "vision_v4_tracking",
        "vision_v4_planner",
        "vision_v4",
    ):
        chunks.extend(
            [
                "\n# ---- " + module + " ----\n",
                inline((ROOT / ("agents/" + module + ".py")).read_bytes()),
            ]
        )
    raw = "".join(chunks).encode()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "task2_agent.py"
    target.write_bytes(raw)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_sha256": sha256(target),
                "bytes": len(raw),
                "backend_sha256": P3_SHA256,
                "development_only": True,
                "perception": "unchanged V3 model/policy",
            },
            indent=2,
        )
        + "\n"
    )
    return target
