"""Compare raw cell labels and valid-board choices using saved model scores."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def _json(path):
    return json.loads(Path(path).read_text())


def run():
    import numpy as np
    import torch

    from agents.vision_v4_types import PerceptionPolicy
    from tools.vision_v4_data import sha
    from tools.vision_v4_selection import screen
    from tools.train_vision_v4 import (
        choose_shared_epoch,
        load_board_contracts,
        score_policy,
        select_policy,
    )

    torch.set_num_threads(1)
    batch = Path("artifacts/v4/runs/batch-1")
    corpora = Path("artifacts/v4/data/corpora")
    output = Path("artifacts/v4/validation/admissible-policy-correction")
    output.mkdir(parents=True, exist_ok=False)
    old = _json("artifacts/v4/models/selection.json")
    reports = [_json(batch / f"fold-{i}/report.json") for i in range(3)]
    epoch = choose_shared_epoch(reports)
    if epoch != old["epoch"]:
        raise SystemExit(
            "Correct shared epoch differs; stop four-run batch without another fit"
        )
    previous = _json(batch / "oof/preselection.json")
    contracts = load_board_contracts(corpora)
    with np.load(batch / "oof/pooled.npz", allow_pickle=False) as values:
        oof = {
            "epoch": epoch,
            "arrays": {k: values[k].copy() for k in values.files},
            "receipts": previous["oof_receipts"],
            "historical_exposure": previous["selection_scope"],
            "board_records": contracts,
        }
    preliminary = select_policy(oof, {})
    path = output / "preselection.json"
    path.write_text(json.dumps(preliminary, indent=2) + "\n")
    selected = screen(
        corpora,
        batch,
        output / "screen",
        preliminary_path=path,
        collapse_margins=True,
        board_records=contracts,
    )
    production_contracts = load_board_contracts(corpora, production=True)
    with np.load(
        batch / "production" / f"logits-{epoch:02d}.npz", allow_pickle=False
    ) as values:
        before = score_policy(
            values["full"],
            values["inset"],
            values["labels"],
            values["board_ids"],
            PerceptionPolicy(**old["policy"]),
            board_records=production_contracts,
        )
        after = score_policy(
            values["full"],
            values["inset"],
            values["labels"],
            values["board_ids"],
            PerceptionPolicy(**selected["policy"]),
            board_records=production_contracts,
        )
    support = np.array(after["class_support"])
    tp = np.array(after["class_true_positive"])
    passed = (
        after["exact_boards"] == after["boards"] == 120
        and all(support > 0)
        and float(np.min(tp / support)) >= 0.99
    )
    result = {
        "passed": bool(passed),
        "epoch_unchanged": epoch,
        "fit_count": 4,
        "before": before,
        "after": after,
        "old_policy": old["policy"],
        "selected_policy": selected["policy"],
        "old_selection_sha256": sha(Path("artifacts/v4/models/selection.json")),
        "new_selection_sha256": sha(output / "screen/selection.json"),
        "final_outcomes_accessed": False,
    }
    (output / "transfer.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise SystemExit(
            "Corrected fixed-policy production transfer failed; stop four-run batch"
        )
    models = Path("artifacts/v4/models")
    shutil.copyfile(models / "selection.json", models / "selection-v1-raw-metric.json")
    if (models / "candidate.json").exists():
        shutil.copyfile(
            models / "candidate.json", models / "candidate-v1-raw-metric.json"
        )
    if (models / "policy.json").exists():
        shutil.copyfile(models / "policy.json", models / "policy-v1-raw-metric.json")
    if (models / "selected-export").exists():
        (models / "selected-export").rename(models / "selected-export-v1-raw-metric")
    checkpoint = Path(old["checkpoint"])
    if selected["policy"]["temperature"] != old["policy"]["temperature"]:
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights["temperature"] = selected["policy"]["temperature"]
        checkpoint = batch / "production/calibrated-final-admissible.pt"
        torch.save(weights, checkpoint)
    replacement = {
        **old,
        "policy": selected["policy"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "weights_source_checkpoint": old["checkpoint"],
        "weights_source_checkpoint_sha256": old["checkpoint_sha256"],
        "transfer": after,
        "transfer_passed": True,
        "downstream_selection": selected,
        "downstream_selection_sha256": sha(output / "screen/selection.json"),
        "previous_selection_sha256": result["old_selection_sha256"],
        "correction": "Exact boards use actual bounded parser; model tensor weights/common epoch unchanged.",
    }
    (models / "selection.json").write_text(json.dumps(replacement, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    run()
