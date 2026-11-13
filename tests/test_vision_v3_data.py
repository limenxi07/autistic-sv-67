import json
import tempfile
import unittest
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from grid_adventure.grid import step
from PIL import Image

from agents.vision_v3_bridge import CLASSES
from tools.vision_v3_data import (
    audit_dataset,
    fit_linear_baseline,
    linear_predict,
    make_manifest,
    new_layout,
    render_dataset,
    render_frame,
    template_catalog,
    template_predict,
    validate_manifest,
)


class DataTests(unittest.TestCase):
    def test_offline_baselines_predict_separable_pixels(self):
        images = np.stack(
            [np.full((48, 48, 3), color, dtype=np.uint8) for color in (0, 5, 250, 255)]
        )
        model = fit_linear_baseline(images, np.array([0, 0, 1, 1]))
        np.testing.assert_array_equal(linear_predict(model, images), [0, 0, 1, 1])
        catalog = template_catalog(self.root)
        self.assertEqual(set(catalog["labels"]), set(range(14)))
        # Every public artwork variant is available to this explicitly optimistic
        # diagnostic; it is never a valid held-out-art generalization result.
        self.assertEqual(
            set(catalog["paths"]),
            {str(p.relative_to(self.root)) for p in self.root.rglob("*.png")},
        )
        index = catalog["labels"].index(4)  # floor has no composited foreground
        np.testing.assert_array_equal(
            template_predict(catalog, catalog["images"][index : index + 1]), [4]
        )

    def test_audit_rejects_sample_label_drift(self):
        manifest = make_manifest(self.root, 20261109)
        output = Path(self.temp.name) / "data"
        render_dataset(manifest, "train", output, crops=14)
        self.assertEqual(audit_dataset(manifest, output)["verified_crops"], 14)
        path = output / "samples.jsonl"
        lines = path.read_text().splitlines()
        sample = json.loads(lines[0])
        sample["class"] = "floor"
        lines[0] = json.dumps(sample)
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            audit_dataset(manifest, output)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "assets"
        for ci, kind in enumerate(CLASSES):
            directory = self.root / kind
            directory.mkdir(parents=True)
            for index in range(5):
                rng = np.random.default_rng(ci * 100 + index)
                pixels = rng.integers(0, 256, (48, 48, 4), dtype=np.uint8)
                pixels[:, :, 3] = 255
                Image.fromarray(pixels).save(directory / f"{kind}_{index}.png")

    def test_duplicate_assets_cannot_cross_splits(self):
        duplicate = self.root / "human" / "copy.png"
        duplicate.write_bytes((self.root / "human" / "human_0.png").read_bytes())
        # Same decoded pixels with changed file encoding must group together too.
        with Image.open(duplicate) as image:
            image.save(self.root / "human" / "copy_encoding.png", compress_level=0)
        manifest = make_manifest(self.root, 20261109)
        groups = defaultdict(set)
        for asset in manifest["assets"]:
            groups[asset["group"]].add(asset["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        copies = [
            a
            for a in manifest["assets"]
            if a["class"] == "human"
            and a["path"].split("/")[-1]
            in ("human_0.png", "copy.png", "copy_encoding.png")
        ]
        self.assertEqual(len(copies), 3)
        self.assertEqual(len({a["group"] for a in copies}), 1)
        tampered = deepcopy(manifest)
        entries = [a for a in tampered["assets"] if a["group"] == copies[0]["group"]]
        entries[0]["split"] = "final" if entries[1]["split"] != "final" else "train"
        with self.assertRaises(ValueError):
            validate_manifest(tampered)

    def test_invisible_rgb_cannot_create_independent_appearance_groups(self):
        pixels = np.zeros((48, 48, 4), dtype=np.uint8)
        pixels[12:36, 12:36] = [220, 120, 60, 255]
        Image.fromarray(pixels).save(self.root / "human" / "alpha_a.png")
        pixels[pixels[:, :, 3] == 0, :3] = 255
        Image.fromarray(pixels).save(self.root / "human" / "alpha_b.png")
        manifest = make_manifest(self.root, 20261109)
        entries = [a for a in manifest["assets"] if "alpha_" in a["path"]]
        self.assertEqual(len({a["group"] for a in entries}), 1)

    def test_floor_is_split_independently_of_foreground(self):
        manifest = make_manifest(self.root, 20261109)
        for kind in CLASSES:
            self.assertEqual(
                {a["split"] for a in manifest["assets"] if a["class"] == kind},
                {"train", "validation", "final"},
            )
        output = Path(self.temp.name) / "data"
        report = render_dataset(manifest, "validation", output, crops=28)
        self.assertEqual(report["crops"], 28)
        samples = [
            json.loads(line)
            for line in (output / "samples.jsonl").read_text().splitlines()
        ]
        for sample in samples:
            constituents = sample["constituents"]
            self.assertIn("floor", {a["class"] for a in constituents})
            self.assertTrue(all(a["split"] == "validation" for a in constituents))

    def test_every_rendered_constituent_has_provenance(self):
        manifest = make_manifest(self.root, 20261109)
        output = Path(self.temp.name) / "data"
        report = render_dataset(manifest, "train", output, crops=14)
        self.assertEqual(report["class_counts"], dict.fromkeys(CLASSES, 1))
        by_path = {a["path"]: a for a in manifest["assets"]}
        samples = [
            json.loads(line)
            for line in (output / "samples.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(samples), 14)
        for sample in samples:
            self.assertEqual(len(sample["output_sha256"]), 64)
            self.assertTrue(sample["augmentation"])
            for constituent in sample["constituents"]:
                asset = by_path[constituent["path"]]
                self.assertEqual(constituent["sha256"], asset["sha256"])
                self.assertEqual(constituent["group"], asset["group"])
        # A stale asset must fail generation, not silently invalidate the manifest.
        path = self.root / manifest["assets"][0]["path"]
        path.write_bytes(b"replaced")
        with self.assertRaises(ValueError):
            render_dataset(manifest, "train", output / "bad", crops=14)

    def test_initial_layouts_have_no_overlaps(self):
        families = set()
        for index in range(100):
            state, witness, family, _ = new_layout(index, split="final")
            families.add(family)
            for col in state.grid:
                self.assertTrue(all(len(cell) <= 1 for cell in col))
            for action in witness:
                state = step(state, action)
            self.assertTrue(state.win, (index, family))
            self.assertFalse(state.lose)
        self.assertEqual(len(families), 10)

    def test_strict_final_multi_gem_family_has_multiple_required_gems(self):
        for index in range(10):
            state, witness, family, _ = new_layout(
                index, split="final", multiple_gems=True
            )
            if family in ("multi-gem", "combined"):
                self.assertGreaterEqual(
                    sum(
                        e.appearance.name == "gem"
                        for col in state.grid
                        for cell in col
                        for e in cell
                    ),
                    2,
                )
            for action in witness:
                state = step(state, action)
            self.assertTrue(state.win)

    def test_final_layouts_are_disjoint_from_training(self):
        manifest = make_manifest(self.root, 20261109)
        final_classes = {
            kind
            for layout in manifest["layouts"]
            if layout["split"] == "final"
            for row in layout["layout"]["labels"]
            for cell in row
            for kind in cell
        }
        self.assertEqual(final_classes, set(CLASSES))
        hashes = defaultdict(set)
        for layout in manifest["layouts"]:
            hashes[layout["split"]].add(layout["sha256"])
        self.assertEqual(len(hashes["final"]), 100)
        self.assertFalse(hashes["final"] & (hashes["train"] | hashes["validation"]))
        tampered = deepcopy(manifest)
        tampered["layouts"][0]["sha256"] = next(iter(hashes["final"]))
        with self.assertRaises(ValueError):
            validate_manifest(tampered)

    def test_full_board_and_reachable_overlap_use_only_assigned_art(self):
        from grid_adventure.step import Action

        from tests.planner_p1_fixtures import mechanic

        manifest = make_manifest(self.root, 20261109)
        state, _ = mechanic("box_coin")
        state = step(step(state, Action.RIGHT), Action.RIGHT)
        record = render_frame(
            manifest, "validation", state, 12345, Path(self.temp.name) / "frame"
        )
        # The second move pushed the box onto the coin at (3, 1).
        self.assertEqual(set(record["labels"][1][3]), {"box", "coin"})
        self.assertTrue(all(a["split"] == "validation" for a in record["constituents"]))
        self.assertIn("floor", {a["class"] for a in record["constituents"]})
        with Image.open(Path(self.temp.name) / "frame" / record["image"]) as image:
            self.assertEqual(image.size, (640, 192))
