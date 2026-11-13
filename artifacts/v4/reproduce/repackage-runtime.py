"""Package the diagnosed tracking repair with identical selected model/policy."""
import json
import re
import shutil
import sys
from pathlib import Path

from tools.package_v4 import package
from tools.v4_data import sha


def run(name="unchanged-ambiguity-repair"):
    if re.fullmatch(r"[a-z0-9-]+", name) is None:
        raise ValueError("Expected a plain V4 repair name")
    models = Path("artifacts/v4/models")
    previous = models / "candidate.json"
    saved = models / ("candidate-before-" + name + ".json")
    if saved.exists():
        raise FileExistsError(saved)
    old = json.loads(previous.read_text())
    source = package(Path.cwd(), models / "selected-export", models / "policy.json",
                     Path("artifacts/v4/candidates") / name)
    revision = sha(source)[:12]
    frozen = Path("artifacts/v4/candidates") / revision
    shutil.copytree(source.parent, frozen)
    shutil.copyfile(previous, saved)
    result = {**old, "revision": revision, "source": str(frozen / "task2_agent.py"),
              "source_sha256": sha(source), "previous_candidate_sha256": old["source_sha256"],
              "runtime_change": name,
              "status": "Repaired development candidate; new exact-source gates required."}
    previous.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "unchanged-ambiguity-repair")
