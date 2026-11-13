"""Check the repaired source in module and pasted-cell scopes; no final data."""

import hashlib
import json
from pathlib import Path

from tools.benchmark_v4 import run_jobs, validate_timing_report


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path("artifacts/v4-notebook-repair-20261110")
    binding = json.loads((root / "candidate-binding.json").read_text())
    source = Path(binding["source"])
    wrapper = Path(binding["notebook_cell_wrapper"])
    assert digest(source) == binding["source_sha256"]
    assert digest(wrapper) == binding["notebook_cell_wrapper_sha256"]
    expected_wrapper = (
        "def _pasted_agent_cell():\n"
        + "".join("    " + line + "\n" for line in source.read_text().splitlines())
        + "    return Agent\n\nAgent = _pasted_agent_cell()\n"
    )
    assert wrapper.read_text() == expected_wrapper
    baseline = json.loads(
        Path("artifacts/v4/validation/development/public-66.json").read_text()
    )
    jobs = [row["job"] for row in baseline["results"]]
    assert len(jobs) == 66
    outputs = {}
    for scope, path in (("module", source), ("notebook-cell", wrapper)):
        report = run_jobs(jobs, root / (scope + "-public-66.json"), source=str(path))
        validate_timing_report(report)
        rows = report["results"]
        checks = {
            "all_66": len(rows) == 66,
            "all_win": all(row["status"] == "win" and row["win"] for row in rows),
            "all_exact": all(row["initial_board_exact"] for row in rows),
            "all_rewards_equal_frozen_v4": all(
                row["reward"] == old["reward"]
                for row, old in zip(rows, baseline["results"], strict=True)
            ),
            "all_episode_under_4_25s": all(row["evaluation_seconds"] < 4.25 for row in rows),
            "all_strict_cold_under_5s": all(row["strict_cold_seconds"] < 5 for row in rows),
            "all_rss_under_1gb": all(row["peak_rss_bytes"] < 1_000_000_000 for row in rows),
            "all_cpu_threads_one": all(row["torch_threads"] == [1, 1] for row in rows),
            "original_budget_unchanged": all(
                not row["budget_reset"] and row["budget_seconds"] == [3.0, 3.5]
                for row in rows
            ),
            "same_power_before_and_after": all(
                row["power_state_before"] == row["power_state_after"] == old["power_state_before"]
                for row, old in zip(rows, baseline["results"], strict=True)
            ),
        }
        outputs[scope] = {
            "checks": checks,
            "passed": all(checks.values()),
            "wins": sum(row["win"] for row in rows),
            "exact_boards": sum(row["initial_board_exact"] for row in rows),
            "max_episode_seconds": max(row["evaluation_seconds"] for row in rows),
            "max_strict_cold_seconds": max(row["strict_cold_seconds"] for row in rows),
            "peak_rss_bytes": max(row["peak_rss_bytes"] for row in rows),
        }
        (root / "public-summary.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2), flush=True)
    return all(result["passed"] for result in outputs.values())


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
