"""Diagnostic-only replay of failed development rows; preserves timing reports."""
import json
import sys
from pathlib import Path

from tools.benchmark_v4 import replay_job


if __name__ == "__main__":
    report = json.loads(Path("artifacts/v4/validation/matrix-new-new.json").read_text())
    candidate = json.loads(Path("artifacts/v4/models/candidate.json").read_text())
    for i, row in enumerate(report["results"]):
        if row["status"] != "win":
            root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts/v4/diagnostics/new-new-remaining-local")
            target = root / f"{i:02d}.json"
            if target.exists():
                raise FileExistsError(target)
            result = replay_job(row["job"], Path(candidate["source"]), target, trace=True)
            print(row["label"], result["results"][0]["status"], flush=True)
