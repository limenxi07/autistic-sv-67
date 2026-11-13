"""Freeze a failed development gate without rerunning or unlocking any corpus."""
import json
import shutil
from pathlib import Path

from tools.qualify_v4 import assess_fresh, qualify
from tools.v4_data import sha
from tools.v4_development import regression_bundle


def read(path):
    return json.loads(Path(path).read_text())


def run():
    development = Path("artifacts/v4/validation/development")
    if Path("artifacts/v4/qualification/final-single-attempt").exists():
        raise ValueError("Stopped pre-final assembly requires an untouched final corpus")
    selected = read("artifacts/v4/models/candidate.json")
    candidate = Path(selected["source"])
    if sha(candidate) != selected["source_sha256"]:
        raise ValueError("Changed candidate")
    manifest = read(development / "acceptance-manifest.json")
    empty = {"source_sha256": selected["source_sha256"], "results": []}
    public = read(development / "public-66.json")
    result = qualify(public, regression_bundle(development), empty, manifest)
    fresh_names = set(assess_fresh(empty, manifest)["checks"])
    checks = {k: v for k, v in result["checks"].items() if k not in fresh_names}
    retired = read(development / "retired-v5-300.json")["results"]
    gates = {"passed": all(checks.values()), "checks": checks,
             "failures": [k for k, v in checks.items() if not v],
             "source_sha256": sha(candidate), "final_outcomes_unlocked": False,
             "public_wins": result["public_wins"],
             "retired_v5_wins": result["retired_v5_wins"],
             "retired_v5_exact_boards": result["retired_v5_exact_boards"],
             "retired_v5_attempted": len(retired),
             "retired_v5_unattempted": 300 - len(retired),
             "stop_reason": "First retired-corpus worker hit strict5s cold watchdog; fail closed without replacing the failure or unlocking final data.",
             "failed_rows": [r for r in retired if r["status"] != "win"]}
    if gates["passed"]:
        raise ValueError("Stopped assembly is only for failed development gates")
    (development / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
    result.update(stage="stopped_before_final", final_outcomes_unlocked=False,
                  fresh_layout_wins=None, fresh_layout_exact_boards=None,
                  fresh_layout_evaluated=False,
                  retired_v5_attempted=len(retired), retired_v5_unattempted=300-len(retired),
                  stop_reason=gates["stop_reason"], failed_rows=gates["failed_rows"],
                  development_checks=checks)
    out = Path("artifacts/v4/qualification/stopped-before-final.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    frozen = Path("artifacts/v4/frozen-not-promoted")
    frozen.mkdir(parents=True, exist_ok=False)
    for source, name in ((candidate, "task2_agent.py"),
                         (candidate.parent / "manifest.json", "manifest.json"),
                         (Path("artifacts/v4/models/policy.json"), "policy.json"),
                         (out, "qualification.json")):
        shutil.copyfile(source, frozen / name)
    print(json.dumps(gates, indent=2))


if __name__ == "__main__":
    run()
