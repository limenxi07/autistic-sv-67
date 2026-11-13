"""Track prior artwork exposure and build reproducible, disjoint data splits."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from agents.vision_v4_bridge import CLASSES


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_exposure(manifests: list[Path], evidence_roots: list[Path]) -> dict:
    groups = {}
    assets = {}
    receipts = {}
    lineage = {}
    for source in manifests:
        manifest = json.loads(source.read_text())
        receipts[str(source)] = sha(source)
        for asset in manifest["assets"]:
            key = asset["group"]
            group = groups.setdefault(
                key,
                {
                    "group": key,
                    "classes": set(),
                    "pixel_sha256": set(),
                    "exposed": True,
                    "used_for_fit": False,
                    "used_for_selection": False,
                    "evaluated_before_v4": False,
                    "source_manifests": set(),
                    "unknown_provenance_is_exposed": True,
                },
            )
            group["classes"].add(asset["class"])
            group["pixel_sha256"].add(asset["pixel_sha256"])
            group["source_manifests"].add(str(source))
            group["used_for_fit"] |= asset["split"] == "train"
            group["used_for_selection"] |= asset["split"] == "validation"
            # V3's final-v5 has been evaluated and informs this task. Unknown
            # usage of earlier public partitions is conservatively exposed too.
            group["evaluated_before_v4"] |= asset["split"] == "final"
            identity = (asset["path"], asset["sha256"])
            if identity not in assets:
                assets[identity] = dict(asset)
                assets[identity]["source_asset_root"] = manifest["asset_root"]
    for root in evidence_roots:
        # Record selected and nonselected run/checkpoint lineage. Large corpus
        # archives already have immutable parent receipts; no archive copying.
        for pattern in (
            "**/config.json",
            "**/selection.json",
            "**/report.json",
            "**/*.pt",
            "**/manifest*.json",
        ):
            for source in sorted(root.glob(pattern)):
                if source.is_file():
                    lineage[str(source)] = {
                        "sha256": sha(source),
                        "bytes": source.stat().st_size,
                    }
    records = []
    for group in groups.values():
        records.append(
            {k: sorted(v) if isinstance(v, set) else v for k, v in group.items()}
        )
    return {
        "version": 4,
        "groups": sorted(records, key=lambda g: g["group"]),
        "assets": list(assets.values()),
        "manifest_hashes": receipts,
        "checkpoint_and_selection_lineage": lineage,
        "untouched_groups": [],
        "unseen_art_status": "unproven",
        "retired_final": "all V3 final-v5 is development/regression; no artwork reset by new seeds",
    }


def make_folds(exposure: dict, seed: int = 20261109) -> dict:
    groups = exposure["groups"]
    pixels = {}
    for group in groups:
        for pixel in group["pixel_sha256"]:
            if pixel in pixels and pixels[pixel] != group["group"]:
                raise ValueError(
                    "Duplicate pixel assigned to different near-duplicate groups"
                )
            pixels[pixel] = group["group"]
    support = Counter(kind for group in groups for kind in group["classes"])
    if any(support[k] < 3 for k in CLASSES):
        raise ValueError("Insufficient class group support for three folds")
    assignments = [[], [], []]
    counts = [Counter(), Counter(), Counter()]
    # Shared cross-class groups are placed once globally. Deterministic greedy
    # stratification handles them before the independent per-class groups.
    ordered = sorted(
        groups,
        key=lambda g: (
            -len(g["classes"]),
            g["classes"],
            hashlib.sha256(canonical([seed, g["group"]])).hexdigest(),
        ),
    )
    for group in ordered:
        fold = min(
            range(3),
            key=lambda i: (
                sum(counts[i][k] for k in group["classes"]),
                len(assignments[i]),
                i,
            ),
        )
        assignments[fold].append(group["group"])
        counts[fold].update(group["classes"])
    if any(any(c[k] == 0 for k in CLASSES) for c in counts):
        raise ValueError("Three folds could not establish every class support")
    return {
        "seed": seed,
        "folds": [
            {
                "fold": i,
                "groups": sorted(assignments[i]),
                "class_support": {k: counts[i][k] for k in CLASSES},
            }
            for i in range(3)
        ],
        "exposure_sha256": hashlib.sha256(canonical(exposure)).hexdigest(),
        "unseen_art_status": "unproven",
        "evaluation_scope": "historically exposed public artwork development folds",
    }


def validate_fold_initialization(config):
    if (
        config.get("checkpoint") is not None
        or config.get("initial_checkpoint") is not None
    ):
        raise ValueError("Fold models must initialize from scratch")


def audit_vision_v4_data(manifest: dict) -> dict:
    exposure, folds = manifest["exposure"], manifest["folds"]
    violations = []
    declared = {g["group"]: g for g in exposure["groups"]}
    seen = set()
    pixel_folds = {}
    for fold in folds["folds"]:
        counts = Counter()
        for key in fold["groups"]:
            if key in seen:
                violations.append("group crosses folds")
            seen.add(key)
            if key not in declared:
                violations.append("unknown fold group")
                continue
            group = declared[key]
            counts.update(group["classes"])
            for pixel in group["pixel_sha256"]:
                if pixel in pixel_folds and pixel_folds[pixel] != fold["fold"]:
                    violations.append("pixel crosses folds")
                pixel_folds[pixel] = fold["fold"]
        if any(counts[k] == 0 for k in CLASSES):
            violations.append("missing class support")
    if seen != set(declared):
        violations.append("unassigned group")
    for key in manifest.get("untouched_groups", exposure.get("untouched_groups", ())):
        if key not in declared or declared[key]["exposed"]:
            violations.append("exposed group declared untouched")
    return {
        "violations": sorted(set(violations)),
        "groups": len(declared),
        "class_support": dict(
            Counter(k for g in declared.values() for k in g["classes"])
        ),
        "unseen_art_status": "unproven",
    }


def balanced_asset_draws(assets, count, seed):
    """Class then source-group then variant, independent of duplicate count."""
    inventory = {k: defaultdict(list) for k in CLASSES}
    for asset in assets:
        inventory[asset["class"]][asset["group"]].append(asset)
    if any(not inventory[k] for k in CLASSES):
        raise ValueError("Missing class support")
    rng = random.Random(seed)
    result = []
    for i in range(count):
        kind = CLASSES[i % len(CLASSES)]
        group = rng.choice(sorted(inventory[kind]))
        result.append(rng.choice(inventory[kind][group]))
    return result


VISIBLE_PRIORITY = (
    "human",
    "box",
    "wall",
    "locked",
    "opened",
    "gem",
    "coin",
    "key",
    "boots",
    "ghost",
    "shield",
    "lava",
    "exit",
    "floor",
)


def visible_target(layers):
    if not layers or any(k not in CLASSES for k in layers):
        raise ValueError("unestablished visible target")
    return next(k for k in VISIBLE_PRIORITY if k in layers)


def new_layout(index, *, split, multiple_gems=False):
    """V4-only optional side objects establish predeclared critical coverage."""
    from grid_adventure.grid import step

    from agents.vision_v4_bridge import ENTITY_TYPES
    from agents.vision_v4_tracking import agent_position
    from tools.vision_v3_data import new_layout as parent_layout

    state, witness, family, seed = parent_layout(
        index, split=split, multiple_gems=multiple_gems
    )
    visited = {agent_position(state)}
    current = state
    for action in witness:
        current = step(current, action)
        visited.add(agent_position(current))
    if not current.win:
        raise ValueError("Parent witness failed")
    for kind in ("coin", "ghost", "box"):
        options = [
            (x, y)
            for x in range(1, state.width - 1)
            for y in range(1, state.height - 1)
            if (x, y) not in visited and not state.grid[x][y]
        ]
        if not options:
            options = [
                (x, y)
                for x in range(1, state.width - 1)
                for y in range(1, state.height - 1)
                if (x, y) not in visited
                and len(state.grid[x][y]) == 1
                and state.grid[x][y][0].appearance.name == "wall"
            ]
        if not options:
            raise ValueError("No legal optional-object cell")
        x, y = random.Random(seed + CLASSES.index(kind)).choice(sorted(options))
        state.grid[x][y].clear()
        state.add((x, y), ENTITY_TYPES[kind]())
    current = state
    for action in witness:
        current = step(current, action)
    if not current.win or current.lose:
        raise ValueError("V4 witness failed")
    return state, witness, family, seed


def choose_asset(entries, rng):
    bygroup = defaultdict(list)
    for entry in entries:
        bygroup[entry["group"]].append(entry)
    return rng.choice(bygroup[rng.choice(sorted(bygroup))])


# Unchanged offline renderer plumbing from V3; choice policy and semantic
# targets are versioned here. Never imported by the submission runtime.
import inspect

import numpy as np
from PIL import Image, ImageEnhance

from agents.vision_v4_bridge import ENTITY_TYPES
from tools.vision_v3_data import (
    SPLITS,
    _isolated_root,
    digest,
    validate_manifest,
)


def render_frame(manifest, split, state, appearance_seed, output, *, resolution=640):
    """Render a real full board, including reachable layered later frames."""
    from grid_adventure.env import GridAdventureEnv
    from grid_adventure.grid import to_state
    from grid_universe.renderer.image import ImageRenderer

    from tools.benchmark_vision_v3 import oracle_labels

    validate_manifest(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = _isolated_root(manifest, split, output)
    rng = random.Random(appearance_seed)
    selected = {}
    for kind in CLASSES:
        entries = [
            a for a in manifest["assets"] if a["class"] == kind and a["split"] == split
        ]
        if not entries:
            raise ValueError("Split lacks a required appearance class")
        selected[kind] = choose_asset(entries, rng)
    constituents = {}

    def lookup(obj, size):
        name = obj.appearance.name
        if name == "door":
            name = "locked" if "locked" in obj.properties else "opened"
        entry = selected[name]
        constituents[entry["path"]] = entry
        with Image.open(root / entry["path"]) as image:
            return image.convert("RGBA").resize((size, size), Image.Resampling.BILINEAR)

    original_map = (
        inspect.signature(GridAdventureEnv).parameters["render_image_map"].default
    )
    image_map = type(original_map)(
        {k: v for k, v in original_map.items() if v in CLASSES}
    )
    renderer = ImageRenderer(
        resolution=resolution,
        asset_root=str(root),
        image_map=image_map,
        tex_lookup_fn=lookup,
    )
    image = renderer.render(to_state(state))
    labels = oracle_labels(state)
    identity = digest(canonical([labels, appearance_seed, state.turn, resolution]))[:16]
    name = f"frame-{identity}.png"
    path = output / name
    image.save(path)
    return {
        "image": name,
        "image_sha256": digest(path.read_bytes()),
        "labels": labels,
        "visible_labels": [[visible_target(cell) for cell in row] for row in labels],
        "pixels_sha256": digest(np.asarray(image).tobytes()),
        "width": state.width,
        "height": state.height,
        "turn": state.turn,
        "appearance_seed": appearance_seed,
        "appearance_split": split,
        "constituents": list(constituents.values()),
        "resolution": resolution,
        "renderer_version": manifest["renderer_version"],
        "manifest_sha256": digest(canonical(manifest)),
    }


def render_dataset(manifest: dict, split: str, output: Path, crops: int) -> dict:
    from grid_adventure.env import GridAdventureEnv
    from grid_adventure.grid import GridState, to_state
    from grid_adventure.movements import MOVEMENTS
    from grid_adventure.objectives import OBJECTIVES
    from grid_universe.renderer.image import ImageRenderer

    validate_manifest(manifest)
    if split not in SPLITS or crops < 1:
        raise ValueError("Invalid generation request")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = _isolated_root(manifest, split, output)
    inventory = {
        kind: [
            a for a in manifest["assets"] if a["split"] == split and a["class"] == kind
        ]
        for kind in CLASSES
    }
    if any(not entries for entries in inventory.values()):
        raise ValueError("Split lacks a required class")
    rng = random.Random(manifest["seed"] + SPLITS.index(split) * 1000000007)
    original_map = (
        inspect.signature(GridAdventureEnv).parameters["render_image_map"].default
    )
    # The 14-class initial-state dataset has no dead human. Do not require or
    # silently read the extra sleeping artwork through the renderer's map.
    image_map = type(original_map)(
        {key: value for key, value in original_map.items() if value in CLASSES}
    )
    cache = {}
    shard_images, shard_labels, shards = [], [], []
    counts = Counter()
    with (output / "samples.jsonl").open("x") as samples:
        for index in range(crops):
            kind = CLASSES[index % len(CLASSES)]
            selected = {"floor": choose_asset(inventory["floor"], rng)}
            if kind != "floor":
                selected[kind] = choose_asset(inventory[kind], rng)
            key = tuple((k, a["path"]) for k, a in sorted(selected.items()))
            if key not in cache:
                constituents = []

                def lookup(obj, size, selected=selected, constituents=constituents):
                    name = obj.appearance.name
                    if name == "door":
                        name = "locked" if "locked" in obj.properties else "opened"
                    entry = selected[name]
                    constituents.append(entry)
                    with Image.open(root / entry["path"]) as image:
                        return image.convert("RGBA").resize(
                            (size, size), Image.Resampling.BILINEAR
                        )

                state = GridState(
                    width=1,
                    height=1,
                    movement=MOVEMENTS["cardinal"],
                    objective=OBJECTIVES["collect_gems_and_exit"],
                    seed=index,
                )
                if kind != "floor":
                    state.add((0, 0), ENTITY_TYPES[kind]())
                renderer = ImageRenderer(
                    resolution=48,
                    asset_root=str(root),
                    image_map=image_map,
                    tex_lookup_fn=lookup,
                )
                rendered = renderer.render(to_state(state)).convert("RGB")
                if not constituents or "floor" not in {
                    a["class"] for a in constituents
                }:
                    raise ValueError("Renderer failed to attribute floor")
                cache[key] = rendered, constituents
            rendered, constituents = cache[key]
            brightness, contrast, saturation = (rng.uniform(0.9, 1.1) for _ in range(3))
            dx, dy = rng.randint(-1, 1), rng.randint(-1, 1)
            augmented = ImageEnhance.Brightness(rendered).enhance(brightness)
            augmented = ImageEnhance.Contrast(augmented).enhance(contrast)
            augmented = ImageEnhance.Color(augmented).enhance(saturation)
            # Reflect-pad one pixel rather than erasing borders or adding black.
            array = np.asarray(augmented)
            padded = np.pad(array, ((1, 1), (1, 1), (0, 0)), mode="reflect")
            array = padded[1 + dy : 49 + dy, 1 + dx : 49 + dx].copy()
            sample = {
                "index": index,
                "class": kind,
                "split": split,
                "layout": "single-cell-public-renderer",
                "constituents": constituents,
                "augmentation": {
                    "brightness": brightness,
                    "contrast": contrast,
                    "saturation": saturation,
                    "dx": dx,
                    "dy": dy,
                },
                "output_sha256": digest(array.tobytes()),
            }
            samples.write(json.dumps(sample, separators=(",", ":")) + "\n")
            shard_images.append(array)
            shard_labels.append(CLASSES.index(kind))
            counts[kind] += 1
            if len(shard_images) == 2048 or index == crops - 1:
                path = output / f"crops-{len(shards):04d}.npz"
                np.savez_compressed(
                    path,
                    images=np.stack(shard_images),
                    labels=np.asarray(shard_labels, dtype=np.uint8),
                )
                shards.append(
                    {
                        "path": path.name,
                        "sha256": digest(path.read_bytes()),
                        "samples": len(shard_images),
                        "bytes": path.stat().st_size,
                    }
                )
                shard_images.clear()
                shard_labels.clear()
    sample_path = output / "samples.jsonl"
    report = {
        "split": split,
        "crops": sum(s["samples"] for s in shards),
        "class_counts": dict(counts),
        "shards": shards,
        "samples_sha256": digest(sample_path.read_bytes()),
        "manifest_sha256": digest(canonical(manifest)),
        "renderer_version": manifest["renderer_version"],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def layout_record(index, split):
    from tools.benchmark_vision_v3 import oracle_labels

    state, witness, family, seed = new_layout(index, split=split, multiple_gems=True)
    payload = {
        "width": state.width,
        "height": state.height,
        "labels": oracle_labels(state),
    }
    return {
        "index": index,
        "family": family,
        "seed": seed,
        "split": split,
        "layout": payload,
        "sha256": digest(canonical(payload)),
        "witness": [a.name for a in witness],
        "appearance_seeds": [
            2026110900001 + index * 104729 + a * 7919 for a in range(3)
        ],
        "generator": "v4_optional_sides",
        "multiple_gems": True,
    }


def _board_corpus(
    manifest,
    records,
    geometry_split,
    appearance_split,
    output,
    *,
    appearances,
    temporal=False,
):
    from grid_adventure.grid import step

    from agents.vision_v4_tracking import agent_position

    output.mkdir(parents=True, exist_ok=False)
    saved = []
    for layout in records:
        state, witness, _, _ = new_layout(
            layout["index"], split=geometry_split, multiple_gems=True
        )
        frames = [(state, None)]
        if temporal:
            frames = []
            current = state
            # At most six reachable states per layout; prefer actual overlaps,
            # then spaced visible transitions. Preserve all semantic layers.
            available = []
            for n, action in enumerate(witness):
                current = step(current, action)
                if current.win or current.lose:
                    break
                pos = agent_position(current)
                overlap = len(current.grid[pos[0]][pos[1]]) > 1
                available.append((not overlap, n, current))
            available.sort(key=lambda v: (v[0], v[1]))
            for _, n, current in available[:6]:
                frames.append((current, n))
        for state, frame in frames:
            for a in range(appearances):
                record = render_frame(
                    manifest,
                    appearance_split,
                    state,
                    layout["appearance_seeds"][a],
                    output,
                    resolution=(640, 480, 512)[a % 3],
                )
                record.update(
                    layout_sha256=layout["sha256"],
                    geometry_split=geometry_split,
                    family=layout["family"],
                    layout_seed=layout["seed"],
                    layout_index=layout["index"],
                    witness=layout["witness"],
                    reachable_frame=frame,
                )
                if not temporal and any(
                    len(cell) != 1 for row in record["labels"] for cell in row
                ):
                    raise ValueError("Illegal initial overlap")
                saved.append(record)
        print("board corpus", output, layout["index"], len(saved), flush=True)
    records_path = output / "boards.jsonl"
    records_path.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in saved)
    )
    report = {
        "boards": len(saved),
        "geometry_split": geometry_split,
        "appearance_split": appearance_split,
        "records_sha256": sha(records_path),
        "manifest_sha256": digest(canonical(manifest)),
        "temporal": temporal,
        "layout_count": len(records),
        "class_counts": dict(
            Counter(k for r in saved for row in r["visible_labels"] for k in row)
        ),
        "semantic_overlap_counts": dict(
            Counter(
                "+".join(sorted(c))
                for r in saved
                for row in r["labels"]
                for c in row
                if len(c) > 1
            )
        ),
    }
    _write_json(output / "report.json", report)
    return report


def prepare_corpora(exposure: dict, folds: dict, output: Path) -> dict:
    """Freeze layout/job rules before outcomes, then render bounded run inputs."""
    audit = audit_vision_v4_data({"exposure": exposure, "folds": folds})
    if audit["violations"]:
        raise ValueError("Exposure/fold audit failed: " + str(audit["violations"]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads(Path(next(iter(exposure["manifest_hashes"]))).read_text())
    from tools.vision_v3_data import asset_root

    source["asset_root"] = str(asset_root())
    # The historical hash set includes every version, not only the selected V3.
    prior = set()
    for path in exposure["manifest_hashes"]:
        prior.update(r["sha256"] for r in json.loads(Path(path).read_text())["layouts"])
    layouts = {}
    seen = set(prior)
    for split, start, count in [
        ("train", 10000, 200),
        ("validation", 20000, 40),
        ("transfer", 30000, 40),
        ("final", 90000, 100),
    ]:
        records = []
        geometry = "validation" if split == "transfer" else split
        index = start
        while len(records) < count:
            record = layout_record(index, geometry)
            # Preserve ten-family balance when skipping an actual duplicate.
            if record["sha256"] not in seen:
                records.append(record)
                seen.add(record["sha256"])
            index += 1
        layouts[split] = records
    if Counter(r["family"] for r in layouts["final"]) != Counter(
        {r["family"]: 10 for r in layouts["final"]}
    ):
        raise ValueError("Final family balance failed")
    # Freeze every final semantic/job choice without loading any model.
    contract = {
        "version": 4,
        "exposure_sha256": digest(canonical(exposure)),
        "folds_sha256": digest(canonical(folds)),
        "unseen_art_status": "unproven",
        "layouts": layouts,
        "generator_sha256": sha(__file__),
        "selection_rules": {
            "runs": 4,
            "max_epochs": 20,
            "crop_count": 200000,
            "initial_boards": 600,
            "max_reachable_frames": 600,
            "view": ["full", "inset_005"],
            "temperature_grid": [0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4, 5.6, 8],
            "confidence": [0.85, 0.9, 0.95],
            "margin": [0.1, 0.2, 0.3],
            "epoch_rank": [
                "exact_boards",
                "worst_class_recall",
                "lower_nll",
                "earlier_epoch",
            ],
        },
    }
    _write_json(output / "contract.json", contract)
    runs = []
    for fold in range(4):
        name = f"fold-{fold}" if fold < 3 else "production"
        directory = output / name
        directory.mkdir()
        held = set(folds["folds"][fold]["groups"]) if fold < 3 else set()
        manifest = dict(source)
        manifest["seed"] = 20261109 + fold
        manifest["assets"] = [
            {**a, "split": "validation" if a["group"] in held else "train"}
            for a in source["assets"]
        ]
        manifest["layouts"] = (
            layouts["train"] + layouts["validation"] + layouts["final"]
        )
        if fold == 3:
            # Production transfer uses separate layouts with all exposed art.
            validation_manifest = {
                **manifest,
                "assets": [{**a, "split": "validation"} for a in manifest["assets"]],
            }
        else:
            validation_manifest = manifest
        _write_json(directory / "manifest.json", manifest)
        _write_json(directory / "validation-manifest.json", validation_manifest)
        render_dataset(manifest, "train", directory / "train-crops", 200000)
        from tools.vision_v3_data import audit_dataset

        audit_dataset(manifest, directory / "train-crops")
        initial = _board_corpus(
            manifest,
            layouts["train"],
            "train",
            "train",
            directory / "train-initial",
            appearances=3,
        )
        reachable = _temporal_corpus(
            manifest, layouts["train"][:84], directory / "train-temporal"
        )
        validation_records = layouts["transfer" if fold == 3 else "validation"]
        validation = _board_corpus(
            validation_manifest,
            validation_records,
            "validation",
            "validation",
            directory / "validation-boards",
            appearances=3,
        )
        render_dataset(
            validation_manifest, "validation", directory / "validation-crops", 14000
        )
        audit_dataset(validation_manifest, directory / "validation-crops")
        report = {
            "name": name,
            "seed": 20261109 + fold,
            "train_manifest_sha256": sha(directory / "manifest.json"),
            "initial": initial,
            "reachable": reachable,
            "validation": validation,
            "checkpoint": None,
            "held_groups": sorted(held),
            "historical_art_exposure": True,
        }
        _write_json(directory / "receipt.json", report)
        runs.append(report)
    # Final images are rendered once from already frozen legal states and art.
    # They must never enter any trainer/selection loader.
    final_manifest = {
        **source,
        "layouts": layouts["final"],
        "assets": [{**a, "split": "final"} for a in source["assets"]],
    }
    _write_json(output / "final-appearance-manifest.json", final_manifest)
    final = _board_corpus(
        final_manifest,
        layouts["final"],
        "final",
        "final",
        output / "final-300",
        appearances=3,
    )
    critical = ("gem", "coin", "ghost", "box", "wall")
    if any(final["class_counts"].get(k, 0) < 100 for k in critical):
        raise ValueError("Final critical-class coverage failed")
    report = {
        "runs": runs,
        "final": final,
        "contract_sha256": sha(output / "contract.json"),
        "unseen_art_status": "unproven",
        "audit": audit,
    }
    _write_json(output / "final-manifest.json", report)
    return report


def _temporal_corpus(manifest, layouts, output):
    from tests.vision_v4_fixtures import composite_sequence
    from tools.benchmark_vision_v3 import oracle_labels

    report = _board_corpus(
        manifest, layouts, "train", "train", output, appearances=1, temporal=True
    )
    path = output / "boards.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    families = [
        (False, k)
        for k in (
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
        )
    ]
    families += [
        (True, k) for k in ("gem", "coin", "key", "boots", "ghost", "shield", "exit")
    ]
    for i, (box, kind) in enumerate(families):
        state, frames, actions = composite_sequence(kind, box=box)
        layout_hash = digest(
            canonical(
                {
                    "width": state.width,
                    "height": state.height,
                    "labels": oracle_labels(state),
                }
            )
        )
        for n, frame in enumerate(frames[:6]):
            record = render_frame(
                manifest, "train", frame, 202611090001 + i, output, resolution=384
            )
            record.update(
                layout_sha256=layout_hash,
                geometry_split="train",
                family=f"composite-{box}-{kind}",
                layout_seed=state.seed,
                layout_index=f"composite-{i}",
                witness=[a.name for a in actions],
                reachable_frame=n,
            )
            records.append(record)
    path.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    )
    report.update(
        boards=len(records),
        layout_count=len(layouts) + len(families),
        records_sha256=sha(path),
        class_counts=dict(
            Counter(k for r in records for row in r["visible_labels"] for k in row)
        ),
        semantic_overlap_counts=dict(
            Counter(
                "+".join(sorted(c))
                for r in records
                for row in r["labels"]
                for c in row
                if len(c) > 1
            )
        ),
    )
    if report["boards"] > 600 or report["layout_count"] < 100:
        raise ValueError("Temporal corpus bounds failed")
    _write_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exposure", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_corpora(
        json.loads(args.exposure.read_text()),
        json.loads(args.folds.read_text()),
        args.output,
    )
    print(json.dumps(result, indent=2))
