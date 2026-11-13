"""One diagnostic replay; never replaces the recorded cold-timeout outcome."""
import json
from pathlib import Path

from tools.benchmark_v4 import replay_job


if __name__ == "__main__":
    root = Path("artifacts/v4/validation/development")
    row = json.loads((root / "retired-v5-300.json").read_text())["results"][0]
    selected = json.loads(Path("artifacts/v4/models/candidate.json").read_text())
    output = Path("artifacts/v4/diagnostics/cold-stop/first-case.json")
    if output.exists():
        raise FileExistsError(output)
    report = replay_job(row["job"], Path(selected["source"]), output, trace=True)
    result = report["results"][0]
    print(json.dumps({"diagnostic_only": True, "original_status": row["status"],
                      "original_cold_seconds": row["strict_cold_seconds"],
                      "diagnostic_status": result["status"],
                      "diagnostic_initial_exact": result.get("initial_board_exact"),
                      "diagnostic_cold_seconds": result["strict_cold_seconds"],
                      "first_frame_stages": result.get("trace", [{}])[0].get("stages"),
                      "original_failure_retained": True}, indent=2))
