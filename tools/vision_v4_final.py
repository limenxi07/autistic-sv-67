"""Unlock the frozen fresh-layout corpus once, after exact-source development gates."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from tools.benchmark_vision_v4 import run_jobs, validate_timing_report
from tools.vision_v4_data import canonical, digest, sha
from tools.vision_v4_development import regression_bundle
from tools.qualify_vision_v4 import qualify


def _json(path):
    return json.loads(Path(path).read_text())


def group_recalls(rows, manifest):
    groups = {kind: {} for kind in manifest["fresh_class_counts"]}
    for asset in manifest["fresh_appearance_manifest"]["assets"]:
        groups[asset["class"]].setdefault(
            asset["group"], {"correct": 0, "support": 0, "recall": None}
        )
    records = {
        f"{r['family']}/{r['layout_sha256'][:12]}/{r['appearance_seed']}": r
        for r in manifest["fresh_records"]
    }
    families = defaultdict(lambda: Counter(wins=0, exact_initial_boards=0, cases=0))
    for row in rows:
        record = records[row["label"]]
        byclass = {a["class"]: a["group"] for a in record["constituents"]}
        families[record["family"]].update(
            cases=1,
            wins=int(row["status"] == "win"),
            exact_initial_boards=int(row.get("initial_board_exact") is True),
        )
        for actual, predicted, n in row.get("initial_confusion", ()):
            if actual in byclass:
                value = groups[actual][byclass[actual]]
                value["support"] += n
                if actual == predicted:
                    value["correct"] += n
    for values in groups.values():
        for value in values.values():
            if value["support"]:
                value["recall"] = value["correct"] / value["support"]
    return {"per_class_artwork_group": groups, "per_family": dict(families)}


def run():
    development = Path("artifacts/v4/validation/development")
    gates = _json(development / "gates.json")
    candidate = Path(_json("artifacts/v4/models/candidate.json")["source"])
    if not gates["passed"] or sha(candidate) != gates["source_sha256"]:
        raise ValueError(
            "Exact candidate must pass all development gates before final unlock"
        )
    manifest = _json(development / "acceptance-manifest.json")
    output = Path("artifacts/v4/qualification/final-single-attempt")
    output.mkdir(parents=True, exist_ok=False)
    (output / "unlock.json").write_text(
        json.dumps(
            {
                "source_sha256": sha(candidate),
                "model_sha256": manifest["model_sha256"],
                "policy_sha256": manifest["policy_sha256"],
                "jobs_sha256": digest(canonical(manifest["fresh_jobs"])),
                "contract_sha256": manifest["data_audit"]["contract_sha256"],
                "protocol": "Single exact-source final attempt. Any later tuning retires this corpus.",
            },
            indent=2,
        )
        + "\n"
    )
    fresh = run_jobs(
        manifest["fresh_jobs"],
        output / "fresh-300.json",
        source=str(candidate),
        observation="image",
    )
    validate_timing_report(fresh)
    public = _json(development / "public-66.json")
    result = qualify(public, regression_bundle(development), fresh, manifest)
    result.update(group_recalls(fresh["results"], manifest))
    result["remaining_losses"] = [
        {
            "job": r["job"],
            "reward": r.get("reward"),
            "reason": r.get("agent_info", {}).get("reason"),
            "status": r["status"],
        }
        for r in fresh["results"]
        if r["status"] != "win"
    ]
    result["incorrect_initial_jobs"] = [
        r["job"] for r in fresh["results"] if r.get("initial_board_exact") is not True
    ]
    (output / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    frozen = Path(
        "artifacts/v4/frozen"
        if result["regression_qualified"]
        else "artifacts/v4/frozen-not-promoted"
    )
    frozen.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(candidate, frozen / "task2_agent.py")
    for name in ("manifest.json",):
        shutil.copyfile(candidate.parent / name, frozen / name)
    shutil.copyfile("artifacts/v4/models/policy.json", frozen / "policy.json")
    shutil.copyfile(output / "qualification.json", frozen / "qualification.json")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("per_class_artwork_group", "incorrect_initial_jobs")
            },
            indent=2,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    run()
