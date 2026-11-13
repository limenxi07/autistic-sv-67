"""One complete development rerun after documented host resource pressure.

The earlier failed attempt remains immutable and is never merged row-wise.
No fitting, policy change, deadline extension, or fresh-final access occurs.
"""
import json
import shutil
from pathlib import Path
from time import time

from tools.benchmark_v4 import power_state, run_jobs, validate_timing_report
from tools.qualify_v4 import assess_fresh, qualify
from tools.v4_data import sha
from tools.v4_development import regression_bundle


def read(path):
    return json.loads(Path(path).read_text())


def host():
    root = Path("/sys/devices/system/cpu/cpu0/thermal_throttle")
    return {"time": time(), "power": power_state(),
            "throttle": {p.name: p.read_text().strip() for p in root.glob("*_throttle_count")},
            "pressure": {k: Path("/proc/pressure", k).read_text() for k in ("cpu", "memory", "io")}}


def run():
    development = Path("artifacts/v4/validation/development")
    original = read(development / "retired-v5-300.json")
    if len(original["results"]) != 1 or original["results"][0]["status"] != "cold_timeout":
        raise ValueError("Expected the preserved original first-row timeout")
    output = development / "attempts/clean-retired-once"
    output.mkdir(parents=True, exist_ok=False)
    archived = development / "attempts/thermal-memory-pressure"
    archived.mkdir(parents=True, exist_ok=False)
    for name in ("retired-v5-300.json", "gates.json"):
        shutil.copy2(development / name, archived / name)
    selected = read("artifacts/v4/models/candidate.json")
    candidate = Path(selected["source"])
    manifest = read(development / "acceptance-manifest.json")
    if sha(candidate) != selected["source_sha256"] or sha(candidate) != manifest["source_sha256"]:
        raise ValueError("Changed exact candidate")
    before = host()
    if before["power"]["epp"] != "performance":
        raise ValueError("Comparable Performance/EPP host required")
    (output / "host-before.json").write_text(json.dumps(before, indent=2) + "\n")
    try:
        report = run_jobs(manifest["retired_jobs"], output / "retired-v5-300.json",
                          source=str(candidate), observation="image")
        validate_timing_report(report)
        bundle = regression_bundle(development)
        bundle["retired_v5"] = report
        empty = {"source_sha256": sha(candidate), "results": []}
        measured = qualify(read(development / "public-66.json"), bundle, empty, manifest)
        fresh_names = set(assess_fresh(empty, manifest)["checks"])
        checks = {k: v for k, v in measured["checks"].items() if k not in fresh_names}
        gates = {"passed": all(checks.values()), "checks": checks,
                 "failures": [k for k, v in checks.items() if not v],
                 "source_sha256": sha(candidate), "final_outcomes_unlocked": False,
                 "public_wins": measured["public_wins"],
                 "retired_v5_wins": measured["retired_v5_wins"],
                 "retired_v5_exact_boards": measured["retired_v5_exact_boards"],
                 "attempt": "One complete unchanged-source development rerun after documented thermal/memory pressure; original failed attempt retained separately."}
        (output / "gates.json").write_text(json.dumps(gates, indent=2) + "\n")
        if gates["passed"]:
            shutil.copyfile(output / "retired-v5-300.json", development / "retired-v5-300.json")
            shutil.copyfile(output / "gates.json", development / "gates.json")
        print(json.dumps(gates, indent=2), flush=True)
        if not gates["passed"]:
            raise SystemExit("Clean complete development attempt failed; no more attempts or final unlock")
    finally:
        (output / "host-after.json").write_text(json.dumps(host(), indent=2) + "\n")


if __name__ == "__main__":
    run()
