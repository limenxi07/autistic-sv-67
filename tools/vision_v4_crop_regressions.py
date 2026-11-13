"""Check cell predictions on retired V3 images and saved trajectories."""

from __future__ import annotations

import json
from pathlib import Path


def compare_crops(targets, old, new):
    import numpy as np

    from agents.vision_v4_bridge import CLASSES

    shape = (len(targets), len(targets[0]), len(CLASSES))
    if old.shape != shape or new.shape != shape:
        raise ValueError("Crop/target geometry mismatch")
    result = []
    for y, row in enumerate(targets):
        for x, target in enumerate(row):
            if target is None:
                continue
            a, b = old[y, x], new[y, x]
            first = CLASSES[int(a.argmax())] if np.isfinite(a).all() else None
            second = CLASSES[int(b.argmax())] if np.isfinite(b).all() else None
            if first != target:
                result.append(
                    {
                        "x": x,
                        "y": y,
                        "target": target,
                        "old": first,
                        "new": second,
                        "old_probability": float(a.max()) if first else None,
                        "new_probability": float(b.max()) if second else None,
                        "corrected": second == target,
                    }
                )
    return result


def run(candidate):
    import hashlib

    import numpy as np
    from grid_adventure.grid import from_state
    from grid_adventure.step import Action
    from PIL import Image

    from agents.vision_v4_bridge import CLASSES
    from tests.vision_v3_fixtures import image_env
    from tools.benchmark_vision_v4 import (
        oracle_labels,
        recorded_appearance,
    )
    from tools.benchmark_planner_p1 import load_agent
    from tools.vision_v4_data import sha, visible_target
    from tools.vision_v3_data import new_layout

    output = Path("artifacts/v4/diagnostics/selected-crops")
    output.mkdir(parents=True, exist_ok=False)
    old_source = Path("artifacts/v3/frozen-reviewed/task2_agent.py")
    old_cls, new_cls = load_agent(str(old_source)), load_agent(str(candidate))
    # Constructor configures the isolated standalone module, without stepping.
    old_agent, new_agent = old_cls(), new_cls()
    old_ns = old_agent._step_image.__func__.__globals__
    new_ns = new_agent._step_image.__func__.__globals__
    baseline = json.loads(Path("artifacts/v4/baseline/regressions.json").read_text())
    retired = Path("artifacts/v3/data/final-300-v5")
    if not (retired / "boards.jsonl").is_file():
        retired = Path("artifacts/v4/baseline/retired-images")
    records = {
        r["image_sha256"]: r
        for r in map(
            json.loads,
            (retired / "boards.jsonl").read_text().splitlines(),
        )
    }
    initial, temporal = [], []
    for job in baseline["incorrect_initial_jobs"]:
        record = records[job["expected_image_sha256"]]
        image = retired / record["image"]
        if sha(image) != job["expected_image_sha256"]:
            raise ValueError("Retired image bytes changed")
        pixels = np.asarray(Image.open(image))
        if (
            hashlib.sha256(pixels.tobytes()).hexdigest()
            != job["expected_pixels_sha256"]
        ):
            raise ValueError("Retired decoded pixels changed")
        obs = {
            "image": pixels,
            "info": {"config": {"width": record["width"], "height": record["height"]}},
        }
        old = old_ns["classify_initial"](obs).probabilities
        new = new_ns["classify_evidence"](obs).scores.probabilities
        targets = [[visible_target(c) for c in row] for row in record["labels"]]
        errors = compare_crops(targets, old, new)
        for error in errors:
            initial.append({"job": job, "image": str(image), **error})
    if len(initial) != 23:
        raise ValueError(
            f"Expected exactly23 frozen initial errors, found{len(initial)}"
        )
    attribution = json.loads(
        Path("artifacts/v4/diagnostics/attribution.json").read_text()
    )
    for index, loss in enumerate(attribution["losses"]):
        job = loss["job"]
        event = loss["first_nonplan_event"]
        turn = event["turn"]
        trace = json.loads(Path(loss["trace"]).read_text())["results"][0]["trace"]
        state = new_layout(
            job["generated"],
            split=job["layout_split"],
            multiple_gems=job["multiple_gems"],
        )[0]
        env = recorded_appearance(image_env(state, job["seed"]), job)
        obs, _ = env.reset()
        for entry in trace:
            if entry["turn"] >= turn:
                break
            obs, _, win, lose, _ = env.step(Action[entry["action"]])
            if win or lose:
                raise ValueError("Saved trajectory terminates before diagnostic frame")
        if obs["info"]["status"]["turn"] != turn:
            raise ValueError("Saved trajectory turn mismatch")
        image = output / f"loss-{index}-turn-{turn}.png"
        Image.fromarray(obs["image"]).save(image)
        (image.with_suffix(".json")).write_text(
            json.dumps(obs["info"], indent=2) + "\n"
        )
        old = old_ns["classify_initial"](obs).probabilities
        new = new_ns["classify_evidence"](obs).scores.probabilities
        truth = from_state(env.state)
        targets = [[visible_target(c) for c in row] for row in oracle_labels(truth)]
        errors = compare_crops(targets, old, new)
        actor = next(
            (x, y)
            for y, row in enumerate(targets)
            for x, kind in enumerate(row)
            if kind == "human"
        )
        x, y = actor
        temporal.append(
            {
                "job": job,
                "turn": turn,
                "image": str(image),
                "image_sha256": sha(image),
                "old_error_cells": errors,
                "actor": {
                    "x": x,
                    "y": y,
                    "old": CLASSES[int(old[y, x].argmax())],
                    "new": CLASSES[int(new[y, x].argmax())],
                },
            }
        )
    report = {
        "diagnostic_only": True,
        "source_sha256": sha(candidate),
        "old_source_sha256": sha(old_source),
        "initial_errors": initial,
        "initial_corrected": sum(r["corrected"] for r in initial),
        "initial_total": len(initial),
        "temporal_losses": temporal,
        "truth_used_for_inference": False,
        "timing_qualification": False,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "initial_corrected": report["initial_corrected"],
                "initial_total": 23,
                "temporal_actors": [r["actor"] for r in temporal],
            },
            indent=2,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1]))
