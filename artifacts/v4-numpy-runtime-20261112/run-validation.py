"""Validate a NumPy runtime on public and retired development data only."""

import hashlib
import json
from pathlib import Path

from tools.benchmark_v4 import run_jobs, validate_timing_report


def main():
    root = Path("artifacts/v4-numpy-runtime-20261112")
    binding = json.loads((root / "candidate-binding.json").read_text())
    baseline = json.loads(Path("artifacts/v4/validation/development/public-66.json").read_text())
    retired = json.loads(Path("artifacts/v4/validation/development/retired-v5-300.json").read_text())
    jobs = [r["job"] for r in baseline["results"]]
    debug_jobs = [{"label": label, "seed": 42} for label in dict.fromkeys(j["label"] for j in jobs)]
    batches = [
        ("public-module-66", jobs, binding["source"], baseline),
        ("public-notebook-66", jobs, binding["notebook_cell_wrapper"], baseline),
        ("debug-seed42-22", debug_jobs, binding["source"], None),
        ("retired-300", [r["job"] for r in retired["results"]], binding["source"], retired),
    ]
    summary = {}
    for name, batch, source, old in batches:
        report = run_jobs(batch, root / (name + ".json"), source=source)
        validate_timing_report(report)
        rows = report["results"]
        checks = {
            "all_jobs": len(rows) == len(batch),
            "all_win": all(r["win"] and r["status"] == "win" for r in rows),
            "all_exact": all(r["initial_board_exact"] for r in rows),
            "no_torch_import": all("torch_threads" not in r for r in rows),
            "original_budgets": all(not r["budget_reset"] and r["budget_seconds"] == [3.0, 3.5] for r in rows),
            "resources": all(r["evaluation_seconds"] < 4.25 and r["strict_cold_seconds"] < 5 and r["peak_rss_bytes"] < 1_000_000_000 for r in rows),
            "performance_profile": all(r["power_state_before"] == r["power_state_after"] and r["power_state_before"]["epp"] == "performance" for r in rows),
        }
        if old is not None:
            checks["same_rewards"] = all(r["reward"] == p["reward"] for r, p in zip(rows, old["results"], strict=True))
        summary[name] = {
            "checks": checks, "passed": all(checks.values()),
            "wins": sum(r["win"] for r in rows), "exact_boards": sum(r["initial_board_exact"] for r in rows),
            "cases": len(rows), "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            "max_episode_seconds": max(r["evaluation_seconds"] for r in rows),
            "max_strict_cold_seconds": max(r["strict_cold_seconds"] for r in rows),
            "peak_rss_bytes": max(r["peak_rss_bytes"] for r in rows),
            "failures": [{k: r.get(k) for k in ("label", "job", "status", "reward", "agent_info")} for r in rows if not r["win"]],
        }
        (root / "validation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return all(r["passed"] for r in summary.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
