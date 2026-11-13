"""Compare policies on held-out layouts before a separate production fit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.benchmark_vision_v4 import (
    corpus_jobs,
    run_jobs,
    validate_timing_report,
)
from tools.export_vision_v3 import export_model
from tools.vision_v4_data import sha
from tools.package_vision_v4 import MODULES, package

ROOT = Path(__file__).resolve().parents[1]


def model_source(export, policy):
    return (
        "# V4 selected model and policy; generated loader bytes retained.\n"
        f"TEMPERATURE = {policy['temperature']!r}\n"
        + (Path(export) / "model_loader.py").read_text()
        + "\nfrom agents.vision_v4_types import PerceptionPolicy\nPOLICY = PerceptionPolicy(TEMPERATURE, "
        + repr(policy["min_confidence"])
        + ", "
        + repr(policy["min_margin"])
        + ", "
        + repr(policy["view"])
        + ")\n"
    )


def stage_runtime(output, export, policy):
    output = Path(output)
    (output / "agents").mkdir(parents=True, exist_ok=True)
    (output / "artifacts/p3/frozen").mkdir(parents=True, exist_ok=True)
    for name in MODULES + ("vision_v4_backend",):
        shutil.copyfile(
            ROOT / ("agents/" + name + ".py"), output / ("agents/" + name + ".py")
        )
    shutil.copyfile(
        ROOT / "artifacts/p3/frozen/task1_agent.py",
        output / "artifacts/p3/frozen/task1_agent.py",
    )
    (output / "agents/vision_v4_model.py").write_text(
        model_source(export, policy)
    )
    return output


def collapse_margin_candidates(candidates):
    """For normalized p, max(p)>=c implies the top-two margin>=2*c-1."""
    groups = {}
    for key, candidate in candidates.items():
        p = candidate["policy"]
        redundant = p["min_margin"] <= 2 * p["min_confidence"] - 1 - 1e-6
        group = (
            p["view"],
            p["temperature"],
            p["min_confidence"],
            None if redundant else p["min_margin"],
        )
        groups.setdefault(group, []).append(key)
    representatives = {}
    mapping = {}
    for keys in groups.values():
        first = min(keys, key=lambda k: candidates[k]["policy"]["min_margin"])
        representatives[first] = candidates[first]
        mapping.update({k: first for k in keys})
    return representatives, mapping


def screen(
    corpora=Path("artifacts/v4/data/corpora"),
    batch=Path("artifacts/v4/runs/batch-1"),
    output=Path("artifacts/v4/validation/policy-screen"),
    *,
    preliminary_path=None,
    collapse_margins=False,
    board_records=None,
):
    import numpy as np
    import torch

    from tools.train_vision_v4 import load_board_contracts, select_policy

    if board_records is None:
        board_records = load_board_contracts(corpora)
    output.mkdir(parents=True, exist_ok=False)
    preliminary = json.loads(
        Path(preliminary_path or batch / "oof/preselection.json").read_text()
    )
    contenders = {
        c["key"]: c
        for c in preliminary["candidates"]
        if c["key"] in preliminary["contender_keys"]
    }
    equivalence = {k: k for k in contenders}
    if collapse_margins:
        contenders, equivalence = collapse_margin_candidates(contenders)
    epoch = preliminary["epoch"]
    exports = {}
    results = {}
    binding = []
    for fold in range(3):
        directory = corpora / f"fold-{fold}"
        jobs = corpus_jobs(
            directory / "validation-boards", directory / "validation-manifest.json"
        )[:30]
        (output / f"jobs-fold-{fold}.json").write_text(
            json.dumps(jobs, indent=2) + "\n"
        )
        for key, candidate in contenders.items():
            policy = candidate["policy"]
            view = policy["view"]
            export_key = (fold, view)
            if export_key not in exports:
                base = output / f"fold-{fold}" / view
                base.mkdir(parents=True)
                checkpoint = batch / f"fold-{fold}" / f"epoch-{epoch:02d}.pt"
                weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
                weights["temperature"] = policy["temperature"]
                calibrated = base / "calibrated.pt"
                torch.save(weights, calibrated)
                export = base / "export"
                report = export_model(calibrated, export)
                exports[export_key] = (export, report)
            export, report = exports[export_key]
            name = key.replace("/", "-")
            directory = output / f"fold-{fold}" / name
            runtime = stage_runtime(directory / "runtime", export, policy)
            policy_file = directory / "policy.json"
            policy_file.write_text(
                json.dumps(
                    {**policy, "model_sha256": report["export_sha256"]}, indent=2
                )
                + "\n"
            )
            source = package(runtime, export, policy_file, directory / "candidate")
            measured = run_jobs(
                jobs,
                directory / "episodes.json",
                source=str(source),
                observation="image",
            )
            validate_timing_report(measured)
            rows = measured["results"]
            metrics = results.setdefault(
                key, {"wins": 0, "cases": 0, "seconds": 0.0, "exact_initial_boards": 0}
            )
            metrics["wins"] += sum(r["status"] == "win" and r["win"] for r in rows)
            metrics["cases"] += len(rows)
            metrics["seconds"] += sum(r.get("evaluation_seconds", 5) for r in rows)
            metrics["exact_initial_boards"] += sum(
                r["initial_board_exact"] is True for r in rows
            )
            binding.append(
                {
                    "key": key,
                    "fold": fold,
                    "source_sha256": sha(source),
                    "report_sha256": sha(directory / "episodes.json"),
                }
            )
    for key, representative in equivalence.items():
        if key != representative:
            results[key] = {
                **results[representative],
                "equivalent_to": representative,
                "evidence": "Decision-rule equivalence on normalized probabilities; timing measured only for representative.",
            }
    with np.load(batch / "oof/pooled.npz", allow_pickle=False) as arrays:
        oof = {
            "epoch": epoch,
            "arrays": {k: arrays[k].copy() for k in arrays.files},
            "receipts": preliminary["oof_receipts"],
            "historical_exposure": preliminary["selection_scope"],
            "board_records": board_records,
        }
    selected = select_policy(oof, results)
    selected["downstream_report_bindings"] = binding
    selected["margin_equivalence"] = equivalence
    selected["selection_rules_sha256"] = sha(
        Path("artifacts/v4/data/selection-rules.json")
    )
    (output / "selection.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected, indent=2), flush=True)
    return selected


if __name__ == "__main__":
    screen()
