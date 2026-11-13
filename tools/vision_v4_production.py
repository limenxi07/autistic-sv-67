"""The fourth and final V4 fit, followed by a separate pinned-CPU export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fit():
    from tools.vision_v4_data import sha
    from tools.train_vision_v4 import fit_production
    from tools.train_vision_v3 import CLASSES

    batch = Path("artifacts/v4/runs/batch-1")
    path = Path("artifacts/v4/validation/policy-screen/selection.json")
    selection = json.loads(path.read_text())
    if not selection["downstream_selection_complete"]:
        raise ValueError("Downstream policy comparison incomplete")
    if not all((batch / f"fold-{i}" / "report.json").is_file() for i in range(3)):
        raise ValueError("Exactly three completed fold fits required")
    config = {
        "fold": "production",
        "epochs": selection["epoch"],
        "seed": 20261112,
        "checkpoint": None,
        "class_order": list(CLASSES),
        "data": "artifacts/v4/data/corpora/production",
        "output": str(batch / "production"),
        "device": "cuda",
        "batch": 448,
        "compile": True,
    }
    result = fit_production(config, selection)
    output = Path("artifacts/v4/models")
    output.mkdir(parents=True, exist_ok=True)
    selected = {
        "policy": selection["policy"],
        "epoch": selection["epoch"],
        "checkpoint": result["checkpoint"],
        "checkpoint_sha256": result["checkpoint_sha256"],
        "transfer_passed": result["transfer_passed"],
        "transfer": result["transfer"],
        "downstream_selection_sha256": sha(path),
        "downstream_selection": selection,
        "fit_count": 4,
        "max_fits": 4,
        "no_final_outcomes_used": True,
    }
    (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2), flush=True)
    if not result["transfer_passed"]:
        raise SystemExit(
            "Production transfer failed; four-run batch stopped without further fitting"
        )
    return selected


def export_and_package():
    from tools.export_vision_v3 import export_model
    from tools.vision_v4_data import sha
    from tools.vision_v4_selection import model_source
    from tools.package_vision_v4 import MODULES, package

    models = Path("artifacts/v4/models")
    selected = json.loads((models / "selection.json").read_text())
    if not selected["transfer_passed"] or selected["fit_count"] != 4:
        raise ValueError("Production transfer failed or fit budget violated")
    if sha(Path(selected["checkpoint"])) != selected["checkpoint_sha256"]:
        raise ValueError("Selected final checkpoint changed")
    export = models / "selected-export"
    report = export_model(Path(selected["checkpoint"]), export)
    policy = {
        **selected["policy"],
        "model_sha256": report["export_sha256"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "selection_sha256": sha(models / "selection.json"),
    }
    policy_file = models / "policy.json"
    policy_file.write_text(json.dumps(policy, indent=2) + "\n")
    (ROOT / "agents/vision_v4_model.py").write_text(
        model_source(export, selected["policy"])
    )
    from tools.vision_v4_data import canonical, digest

    signature = digest(
        canonical(
            {
                "policy": sha(policy_file),
                "runtime": {m: sha(ROOT / ("agents/" + m + ".py")) for m in MODULES},
            }
        )
    )[:12]
    stage = Path("artifacts/v4/candidates") / ("package-" + signature)
    source = package(ROOT, export, policy_file, stage)
    revision = sha(source)[:12]
    frozen = Path("artifacts/v4/candidates") / revision
    shutil.copytree(stage, frozen)
    result = {
        "revision": revision,
        "source": str(frozen / "task2_agent.py"),
        "source_sha256": sha(source),
        "model_sha256": report["export_sha256"],
        "policy_sha256": sha(policy_file),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "production_epoch": selected["epoch"],
        "status": "Exact development candidate; qualification not yet run.",
    }
    (models / "candidate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "export"))
    args = parser.parse_args()
    fit() if args.stage == "fit" else export_and_package()
