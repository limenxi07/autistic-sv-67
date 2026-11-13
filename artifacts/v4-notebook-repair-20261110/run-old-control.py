"""Public-only timing diagnostic for unchanged V4 in the current host mode."""

from tools.benchmark_v4 import run_jobs


if __name__ == "__main__":
    run_jobs(
        [
            {"label": "B3 So Close Yet So Far", "seed": 2},
            {"label": "B4 Measure Twice and Cut Once", "seed": 2},
        ],
        "artifacts/v4-notebook-repair-20261110/old-v4-power-control.json",
        source="artifacts/v4/frozen/task2_agent.py",
    )
