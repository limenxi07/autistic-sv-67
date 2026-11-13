"""Public-only controlled startup diagnostic; never consumes final data."""

import json
from pathlib import Path

from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tools.benchmark_v4 import run_jobs


if __name__ == "__main__":
    root = Path("artifacts/v4-budget-recovery-20261112")
    jobs = [{"label": label, "seed": 42} for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS]
    summary = {}
    for revision in ("previous", "repaired"):
        folder = root / ("slow-start-" + revision)
        report = run_jobs(jobs, folder / "public-22.json", source=str(folder / "task2_agent.py"))
        report.update(
            diagnostic_only=True,
            injected_first_perception_seconds=3.05,
            evidence_boundary="Controlled public startup experiment; source has latency injection, not qualification.",
        )
        (folder / "public-22.json").write_text(json.dumps(report, indent=2) + "\n")
        rows = report["results"]
        summary[revision] = {
            "wins": sum(row.get("win", False) for row in rows),
            "cases": len(rows),
            "max_strict_cold_seconds": max(row["strict_cold_seconds"] for row in rows),
            "failures": [
                {key: row.get(key) for key in ("label", "status", "reward", "agent_info")}
                for row in rows if not row.get("win")
            ],
        }
        (root / "slow-start-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
