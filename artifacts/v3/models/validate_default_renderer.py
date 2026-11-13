"""Validate normal public renderer observations using only validation artwork."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from agents.v3_bridge import CLASSES
from agents.v3_perception import classify, configure
from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import to_state
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tools.benchmark_v3 import oracle_labels

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
loader = args.export / "model_loader.py"
namespace = {}
exec(loader.read_text(), namespace)
temperature = json.loads(loader.with_name("report.json").read_text())["temperature"]
configure(namespace["get_model"], temperature)
root = ROOT / "artifacts/v3/data/new-layout-unseen-art-v2/asset-root"
original = inspect.signature(GridAdventureEnv).parameters["render_image_map"].default
image_map = type(original)({k: v for k, v in original.items() if v in CLASSES})
rows = []
for seed in (2, 43, 20261109):
    for name, builder in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
        state = builder()
        state.seed, state.turn_limit = seed, 150
        env = GridAdventureEnv(initial_state_fn=lambda *a, **k: to_state(state), width=state.width, height=state.height,
                               render_asset_root=str(root), render_image_map=image_map, observation_type="image")
        observation, _ = env.reset()
        scores = classify(observation)
        guessed = scores.probabilities.argmax(2)
        labels = oracle_labels(state)
        mismatches = []
        for y, row in enumerate(labels):
            for x, cell in enumerate(row):
                if len(cell) != 1 or CLASSES[int(guessed[y, x])] != cell[0]:
                    mismatches.append({"cell": [x, y], "truth": cell,
                                       "predicted": CLASSES[int(guessed[y, x])], "confidence": float(scores.probabilities[y, x].max())})
        rows.append({"level": name, "seed": seed, "cells": state.width * state.height, "mismatches": mismatches})
report = {"boards": len(rows), "exact_boards": sum(not row["mismatches"] for row in rows), "results": rows,
          "evidence_boundary": "Validation artwork root only, public layouts, normal renderer; no final-model outcome."}
report["semantic_exact_boards"] = sum(not any(not (set(m["truth"]) == {"wall"} and m["predicted"] == "wall") for m in row["mismatches"]) for row in rows)
report["unobservable_duplicate_wall_cells"] = sum(set(m["truth"]) == {"wall"} and m["predicted"] == "wall" for row in rows for m in row["mismatches"])
report["semantic_convention"] = "Only duplicate inert static-wall entities collapse to one wall; other multiplicities retained. All-action/phase equivalence tested."
report["loader_sha256"] = hashlib.sha256(loader.read_bytes()).hexdigest()
report["temperature"] = temperature
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"boards": report["boards"], "semantic_exact_boards": report["semantic_exact_boards"], "raw_exact_boards": report["exact_boards"]}, indent=2))
