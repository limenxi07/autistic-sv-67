"""Compare V4 components while holding the P3 planner fixed."""

from __future__ import annotations

import json
from pathlib import Path

from tools.benchmark_vision_v4 import (
    corpus_jobs,
    run_jobs,
    run_matrix,
    validate_timing_report,
)
from tools.vision_v4_data import sha
from tools.vision_v4_selection import stage_runtime
from tools.vision_v3_data import FAMILIES
from tools.package_vision_v4 import package
from tools.qualify_vision_v4 import assess_fresh, qualify

ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(Path(path).read_text())


def matrix_sources(
    candidate, output, *, production_policy=None, new_export=None, old_export=None
):
    production = (
        production_policy or _json("artifacts/v4/models/selection.json")["policy"]
    )
    new_export = Path(new_export or "artifacts/v4/models/selected-export")
    old_export = Path(old_export or "artifacts/v4/baseline/parent-model-export")
    old_policy = {
        "temperature": 0.25,
        "min_confidence": 0.85,
        "min_margin": 0.2,
        "view": "full",
    }
    result = {
        "old_perception_old_tracking": Path(
            "artifacts/v4/baseline/p3-control/task2_agent.py"
        ),
        "new_perception_new_tracking": candidate,
        "historical_v3": Path("artifacts/v3/frozen-reviewed/task2_agent.py"),
    }
    for name, export, policy in (
        ("old_perception_new_tracking", old_export, old_policy),
        ("new_perception_old_tracking", new_export, production),
    ):
        directory = output / name
        runtime = stage_runtime(directory / "runtime", export, policy)
        perception = runtime / "agents/vision_v4_perception.py"
        if name == "old_perception_new_tracking":
            perception.write_text(
                perception.read_text()
                + """
# Immutable ablation: retain exact V3 selective gem/shield preprocessing.
def classify_evidence(observation, cells=None):
    return PerceptionEvidence(classify_initial(observation, cells), (), "v3_gem_shield")
"""
            )
        else:
            agent = (
                (ROOT / "agents/vision_v3.py")
                .read_text()
                .replace(
                    "agents.vision_v3", "agents.vision_v4"
                )
            )
            agent = agent.replace(
                "from agents.vision_v4_model import TEMPERATURE, get_model",
                "from agents.vision_v4_model import POLICY, TEMPERATURE, get_model",
            )
            agent = agent.replace(
                "configure(get_model, TEMPERATURE)",
                "configure(get_model, TEMPERATURE, policy=POLICY)",
            )
            (runtime / "agents/vision_v4.py").write_text(agent)
            tracking = (
                (ROOT / "agents/vision_v3_tracking.py")
                .read_text()
                .replace(
                    "agents.vision_v3", "agents.vision_v4"
                )
            )
            tracking = tracking.replace(
                "from agents.boss_p2_next import reference_state_key",
                "from agents.vision_v4_planner import reference_state_key",
            )
            tracking = tracking.replace(
                "MIN_CONFIDENCE = 0.85",
                "MIN_CONFIDENCE = " + repr(policy["min_confidence"]),
            )
            tracking = tracking.replace(
                "MIN_MARGIN = 0.20", "MIN_MARGIN = " + repr(policy["min_margin"])
            )
            (runtime / "agents/vision_v4_tracking.py").write_text(
                tracking
            )
            perception.write_text(
                perception.read_text()
                + """
# Immutable ablation: legacy tracker consumes selected fused scores only.
def classify_initial(observation, cells=None):
    return classify_evidence(observation, cells).scores
"""
            )
        exported = _json(export / "report.json")
        path = directory / "policy.json"
        path.write_text(
            json.dumps(
                {
                    **policy,
                    "model_sha256": exported["export_sha256"],
                    "ablation": name,
                    "legacy_preprocessing": "v3_gem_shield"
                    if name.startswith("old_")
                    else None,
                },
                indent=2,
            )
            + "\n"
        )
        source = package(runtime, export, path, directory / "candidate")
        metadata = _json(source.parent / "manifest.json")
        metadata["ablation"] = name
        metadata["perception_implementation"] = (
            "unchanged V3 gem/shield selective"
            if name.startswith("old_")
            else "selected V4"
        )
        (source.parent / "manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        result[name] = source
    return result


def acceptance_manifest(candidate):
    manifest = _json(candidate.parent / "manifest.json")
    corpora = Path("artifacts/v4/data/corpora")
    records = [
        json.loads(line)
        for line in (corpora / "final-300/boards.jsonl").read_text().splitlines()
    ]
    final = _json(corpora / "final-manifest.json")
    audit = _json(corpora / "audit-before-fit.json")
    exposure = _json("artifacts/v4/data/exposure.json")
    manifest.update(
        artifact_path=str(candidate),
        public_jobs=_json("artifacts/v3/qualification/public-jobs.json"),
        structured_jobs=_json("artifacts/v3/qualification/regression-jobs.json"),
        retired_jobs=[
            r["job"]
            for r in _json("artifacts/v3/qualification-reviewed/final-300.json")[
                "results"
            ]
        ],
        fresh_jobs=corpus_jobs(
            corpora / "final-300", corpora / "final-appearance-manifest.json"
        ),
        fresh_records=records,
        fresh_appearance_manifest=_json(corpora / "final-appearance-manifest.json"),
        fresh_appearance_manifest_path=str(corpora / "final-appearance-manifest.json"),
        fresh_asset_root=str(corpora / "final-300/asset-root"),
        corpus_contract=_json(corpora / "contract.json"),
        corpus_final_manifest=final,
        data_audit=audit,
        fresh_class_counts=final["final"]["class_counts"],
        fresh_family_counts={k: 30 for k in FAMILIES},
        known_failure_attribution=_json("artifacts/v4/diagnostics/attribution.json")[
            "losses"
        ],
        training_groups=[g["group"] for g in exposure["groups"]],
        exposure={g["group"]: g["exposed"] for g in exposure["groups"]},
        untouched_groups=[],
        oracle_baseline_source_sha256=_json(
            "artifacts/v4/baseline/p3-paired/gates.json"
        )["true_adapter_sha256"],
    )
    return manifest


def regression_bundle(output):
    return {
        "results": _json(output / "structured-266.json")["results"],
        "baseline": _json("artifacts/v4/baseline/p3-paired/structured-parent.json")[
            "results"
        ],
        "oracle": _json(output / "oracle-22.json")["results"],
        "oracle_baseline": _json(
            "artifacts/v4/baseline/p3-paired/oracle-true-state.json"
        )["results"],
        "retired_v5": _json(output / "retired-v5-300.json"),
        "v3_structured": _json(
            "artifacts/v3/qualification-reviewed/structured-266.json"
        )["results"],
        "v3_public": _json("artifacts/v3/qualification-reviewed/public-66.json")[
            "results"
        ],
        "v3_retired": _json("artifacts/v3/qualification-reviewed/final-300.json")[
            "results"
        ],
    }


def run():
    candidate = Path(_json("artifacts/v4/models/candidate.json")["source"])
    output = Path("artifacts/v4/validation/development")
    output.mkdir(parents=True, exist_ok=False)
    sources = matrix_sources(candidate, output / "matrix-packages")
    jobs = _json("artifacts/v4/baseline/regressions.json")["diagnostic_jobs"]
    run_matrix(jobs, sources, output / "matrix")
    manifest = acceptance_manifest(candidate)
    (output / "acceptance-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    public = run_jobs(
        manifest["public_jobs"],
        output / "public-66.json",
        source=str(candidate),
        observation="image",
    )
    structured = run_jobs(
        manifest["structured_jobs"],
        output / "structured-266.json",
        source=str(candidate),
        observation="gridstate",
    )
    oracle_jobs = [j for j in manifest["public_jobs"] if j["seed"] == 2]
    oracle = run_jobs(
        oracle_jobs,
        output / "oracle-22.json",
        source=str(candidate),
        observation="oracle",
    )
    retired = run_jobs(
        manifest["retired_jobs"],
        output / "retired-v5-300.json",
        source=str(candidate),
        observation="image",
    )
    for report in (public, structured, oracle, retired):
        validate_timing_report(report)
    empty = {"source_sha256": manifest["source_sha256"], "results": []}
    fresh_names = set(assess_fresh(empty, manifest)["checks"])
    measured = qualify(public, regression_bundle(output), empty, manifest)
    checks = {k: v for k, v in measured["checks"].items() if k not in fresh_names}
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [k for k, v in checks.items() if not v],
        "source_sha256": sha(candidate),
        "final_outcomes_unlocked": False,
        "public_wins": sum(r["status"] == "win" for r in public["results"]),
        "retired_v5_wins": sum(r["status"] == "win" for r in retired["results"]),
        "retired_v5_exact_boards": sum(
            r["initial_board_exact"] is True for r in retired["results"]
        ),
    }
    (output / "gates.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    run()
