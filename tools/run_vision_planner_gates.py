"""Exact-source paired P3 gate; launch under flock on designated CPU host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.benchmark_vision_v4 import run_jobs, validate_timing_report
from tools.diagnose_vision_v4 import P3_SHA256, package_p3_control, sha256


def run():
    out = Path("artifacts/v4/baseline/p3-paired")
    out.mkdir(parents=True, exist_ok=True)
    control = package_p3_control(Path("artifacts/v4/baseline/p3-control"))
    parent = Path("artifacts/p3/frozen/task1_agent.py")
    adapter = out / "true-state-adapter.py"
    adapter.write_text(
        parent.read_text()
        + """
P3StructuredAgent = Agent
P3PlannerAdapter = PlannerAdapter
class Agent(P3PlannerAdapter):
    def __init__(self):
        start = perf_counter()
        super().__init__(EpisodeBudget(start, start+3, start+3.5),
                         reference_initial_deadline=start+2)
"""
    )
    jobs = json.loads(
        Path("artifacts/v3/qualification/regression-jobs.json").read_text()
    )
    public = json.loads(Path("artifacts/v3/qualification/public-jobs.json").read_text())
    oracle_jobs = [j for j in public if j["seed"] == 2]
    reports = {}
    for name, source, mode, cases in (
        ("structured-parent", parent, "gridstate", jobs),
        ("structured-control", control, "gridstate", jobs),
        ("oracle-control", control, "oracle", oracle_jobs),
        ("oracle-true-state", adapter, "gridstate", oracle_jobs),
    ):
        (out / "stage").write_text(name)
        reports[name] = run_jobs(
            cases, out / (name + ".json"), source=str(source), observation=mode
        )
        validate_timing_report(reports[name])
    historical = json.loads(
        Path("artifacts/v3/qualification-reviewed/structured-266.json").read_text()
    )
    checks = []
    for label, a, b in (
        ("structured", reports["structured-control"], reports["structured-parent"]),
        ("oracle", reports["oracle-control"], reports["oracle-true-state"]),
        ("preserved-v3", reports["structured-control"], historical),
    ):
        pairs = []
        for row, baseline in zip(a["results"], b["results"], strict=True):
            same = (row["label"], row["seed"]) == (baseline["label"], baseline["seed"])
            passed = (
                same
                and row["status"] == baseline["status"] == "win"
                and row["reward"] == baseline["reward"]
            )
            pairs.append(
                {
                    "label": row["label"],
                    "seed": row["seed"],
                    "passed": passed,
                    "reward": row.get("reward"),
                    "parent_reward": baseline.get("reward"),
                    "status": row["status"],
                    "parent_status": baseline["status"],
                }
            )
        checks.append(
            {"gate": label, "passed": all(p["passed"] for p in pairs), "pairs": pairs}
        )
    result = {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "backend_sha256": P3_SHA256,
        "control_sha256": sha256(control),
        "true_adapter_sha256": sha256(adapter),
    }
    (out / "gates.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "stage").write_text("complete" if result["passed"] else "failed")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
