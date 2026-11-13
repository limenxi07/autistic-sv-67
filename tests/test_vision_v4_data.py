import json
import tempfile
import unittest
from pathlib import Path

from agents.vision_v4_bridge import CLASSES
from tools.vision_v4_data import (
    audit_vision_v4_data,
    build_exposure,
    make_folds,
    validate_fold_initialization,
)


def fixture_exposure():
    groups = []
    for kind in CLASSES:
        for i in range(6):
            group = f"{kind}-{i}"
            groups.append(
                {
                    "group": group,
                    "classes": [kind],
                    "pixel_sha256": [group],
                    "exposed": True,
                    "used_for_fit": True,
                    "used_for_selection": False,
                    "evaluated_before_v4": False,
                    "source_manifests": ["fixture"],
                }
            )
    return {"groups": groups, "unseen_art_status": "unproven"}


class ExposureTests(unittest.TestCase):
    def test_three_deterministic_class_stratified_folds_include_floor(self):
        exposure = fixture_exposure()
        folds = make_folds(exposure)
        self.assertEqual(folds, make_folds(exposure))
        self.assertEqual(len(folds["folds"]), 3)
        for fold in folds["folds"]:
            self.assertEqual(set(fold["class_support"]), set(CLASSES))
            self.assertEqual(fold["class_support"]["floor"], 2)
        manifest = {"exposure": exposure, "folds": folds}
        self.assertEqual(audit_vision_v4_data(manifest)["violations"], [])

    def test_pixel_and_near_duplicate_groups_cannot_cross_folds(self):
        exposure = fixture_exposure()
        exposure["groups"][1]["pixel_sha256"] = exposure["groups"][0]["pixel_sha256"]
        with self.assertRaisesRegex(ValueError, "pixel"):
            make_folds(exposure)
        exposure = fixture_exposure()
        folds = make_folds(exposure)
        folds["folds"][1]["groups"].append(folds["folds"][0]["groups"][0])
        self.assertTrue(
            audit_vision_v4_data({"exposure": exposure, "folds": folds})[
                "violations"
            ]
        )

    def test_missing_class_support_fails(self):
        exposure = fixture_exposure()
        exposure["groups"] = [g for g in exposure["groups"] if g["classes"] != ["gem"]]
        with self.assertRaisesRegex(ValueError, "support"):
            make_folds(exposure)

    def test_v3_checkpoint_cannot_initialize_held_fold(self):
        with self.assertRaisesRegex(ValueError, "scratch"):
            validate_fold_initialization({"checkpoint": "artifacts/v3/runs/best.pt"})
        validate_fold_initialization({"checkpoint": None})

    def test_old_final_v5_is_exposed_even_with_new_seeds(self):
        exposure = build_exposure(
            [Path("artifacts/v3/data/manifest-v5.json")], [Path("artifacts/v3")]
        )
        self.assertTrue(all(g["exposed"] for g in exposure["groups"]))
        self.assertEqual(exposure["untouched_groups"], [])
        self.assertEqual(exposure["unseen_art_status"], "unproven")
        folds = make_folds(exposure, seed=111)
        self.assertEqual(folds["unseen_art_status"], "unproven")
        tampered = {
            "exposure": exposure,
            "folds": folds,
            "untouched_groups": [exposure["groups"][0]["group"]],
        }
        self.assertIn(
            "exposed group declared untouched",
            audit_vision_v4_data(tampered)["violations"],
        )


class CorpusTests(unittest.TestCase):
    def test_critical_coverage_legal_boards_and_replayed_witnesses(self):
        from collections import Counter

        from grid_adventure.grid import step

        from tools.benchmark_vision_v3 import oracle_labels
        from tools.vision_v4_data import new_layout

        counts = Counter()
        for index in range(90000, 90100):
            state, witness, family, _seed = new_layout(
                index, split="final", multiple_gems=True
            )
            for row in oracle_labels(state):
                for cell in row:
                    self.assertEqual(len(cell), 1)
                    counts.update(cell)
            for action in witness:
                state = step(state, action)
            self.assertTrue(state.win, (index, family))
        for kind in ("gem", "coin", "ghost", "box", "wall"):
            self.assertGreaterEqual(counts[kind] * 3, 100, kind)

    def test_composite_targets_preserve_semantic_layers(self):
        from tools.vision_v4_data import visible_target

        self.assertEqual(visible_target(("gem", "human")), "human")
        self.assertEqual(visible_target(("coin", "box")), "box")
        self.assertEqual(visible_target(("wall", "human")), "human")
        with self.assertRaisesRegex(ValueError, "unestablished"):
            visible_target(("sleeping",))

    def test_included_composites_are_reachable_through_public_steps(self):
        from tests.vision_v4_fixtures import composite_sequence
        from tools.benchmark_vision_v3 import oracle_labels
        from tools.vision_v4_data import visible_target

        for box, kinds in [
            (
                False,
                (
                    "gem",
                    "coin",
                    "key",
                    "boots",
                    "ghost",
                    "shield",
                    "wall",
                    "locked",
                    "opened",
                    "lava",
                    "exit",
                ),
            ),
            (True, ("gem", "coin", "key", "boots", "ghost", "shield", "exit")),
        ]:
            for kind in kinds:
                _, frames, _ = composite_sequence(kind, box=box)
                overlaps = [
                    c
                    for frame in frames
                    for row in oracle_labels(frame)
                    for c in row
                    if kind in c and ("box" if box else "human") in c
                ]
                self.assertTrue(overlaps, (box, kind))
                self.assertTrue(
                    all(
                        visible_target(c) == ("box" if box else "human")
                        for c in overlaps
                    )
                )

    def test_public_rendered_composites_keep_dominant_foreground(self):
        from copy import deepcopy

        import numpy as np
        from PIL import Image

        from tests.vision_v4_fixtures import composite_sequence
        from tools.vision_v4_data import render_frame

        manifest = json.loads(Path("artifacts/v3/data/manifest-v5.json").read_text())
        manifest["assets"] = [{**a, "split": "train"} for a in manifest["assets"]]
        manifest["layouts"] = []
        for box, kinds in [
            (
                False,
                (
                    "gem",
                    "coin",
                    "key",
                    "boots",
                    "ghost",
                    "shield",
                    "wall",
                    "locked",
                    "opened",
                    "lava",
                    "exit",
                ),
            ),
            (True, ("gem", "coin", "key", "boots", "ghost", "shield", "exit")),
        ]:
            foreground = "box" if box else "human"
            for kind in kinds:
                _, frames, _ = composite_sequence(kind, box=box)
                frame = next(
                    f
                    for f in frames
                    if any(
                        kind
                        in [
                            e.appearance.name
                            if e.appearance.name != "door"
                            else (
                                "locked"
                                if getattr(e, "locked", None) is not None
                                else "opened"
                            )
                            for e in cell
                        ]
                        and any(e.appearance.name == foreground for e in cell)
                        for col in f.grid
                        for cell in col
                    )
                )
                xy = next(
                    (x, y)
                    for x, col in enumerate(frame.grid)
                    for y, cell in enumerate(col)
                    if len(cell) > 1
                    and any(e.appearance.name == foreground for e in cell)
                    and any(
                        e.appearance.name == kind
                        or e.appearance.name == "door"
                        and kind in ("locked", "opened")
                        for e in cell
                    )
                )
                clean = deepcopy(frame)
                x, y = xy
                clean.grid[x][y][:] = [
                    e for e in clean.grid[x][y] if e.appearance.name == foreground
                ]
                with tempfile.TemporaryDirectory() as d:
                    out = Path(d)
                    record = render_frame(
                        manifest,
                        "train",
                        frame,
                        20261109,
                        out / "composite",
                        resolution=384,
                    )
                    reference = render_frame(
                        manifest,
                        "train",
                        clean,
                        20261109,
                        out / "reference",
                        resolution=384,
                    )
                    a = np.asarray(
                        Image.open(out / "composite" / record["image"]).convert("RGB")
                    )
                    b = np.asarray(
                        Image.open(out / "reference" / reference["image"]).convert(
                            "RGB"
                        )
                    )
                    ih, iw = a.shape[:2]
                    a = a[
                        y * ih // 3 : (y + 1) * ih // 3, x * iw // 8 : (x + 1) * iw // 8
                    ]
                    b = b[
                        y * ih // 3 : (y + 1) * ih // 3, x * iw // 8 : (x + 1) * iw // 8
                    ]
                    asset = next(
                        c for c in record["constituents"] if c["class"] == foreground
                    )
                    with Image.open(
                        out / "composite" / "asset-root" / asset["path"]
                    ) as image:
                        alpha = np.asarray(
                            image.convert("RGBA").resize(
                                (a.shape[1], a.shape[0]), Image.Resampling.BILINEAR
                            )
                        )[:, :, 3]
                    mask = alpha == 255
                    self.assertGreater(int(mask.sum()), 0, (box, kind))
                    # Items are visibly rendered as a small corner badge on boxes.
                    # Compare dominant opaque foreground, not pixel identity.
                    agreement = float(np.mean(np.all(a[mask] == b[mask], axis=1)))
                    self.assertGreaterEqual(agreement, 0.80, (box, kind, agreement))


class ContractTests(unittest.TestCase):
    def test_final_layout_contract_rejects_any_historical_regression_layout(self):
        from tools.audit_vision_v4_data import audit_layout_contract

        contract = {
            "layouts": {
                "train": [{"sha256": "train"}],
                "validation": [{"sha256": "val"}],
                "transfer": [{"sha256": "transfer"}],
                "final": [{"sha256": "old-regression"}],
            }
        }
        with self.assertRaisesRegex(ValueError, "historical"):
            audit_layout_contract(contract, {"old-regression"})
        contract["layouts"]["final"][0]["sha256"] = "new"
        self.assertEqual(
            audit_layout_contract(contract, {"old-regression"})["violations"], []
        )
