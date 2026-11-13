"""Reproducible V3 appearance/layout splits and public-renderer training crops."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import shutil
import sys
from collections import Counter, deque
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.vision_v3_bridge import CLASSES, ENTITY_TYPES

SPLITS = ("train", "validation", "final")
FAMILIES = (
    "maze",
    "multi-gem",
    "coins",
    "keys-doors",
    "boxes",
    "lava-shields",
    "boots",
    "phasing",
    "mixed-effects",
    "combined",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def asset_root():
    from grid_adventure.env import GridAdventureEnv

    return Path(
        inspect.signature(GridAdventureEnv).parameters["render_asset_root"].default
    )


def new_layout(index, *, split, multiple_gems=False):
    """Independent DFS layout family, with a geometry-derived public witness."""
    from grid_adventure.entities import AgentEntity, ExitEntity, WallEntity
    from grid_adventure.grid import GridState
    from grid_adventure.movements import MOVEMENTS
    from grid_adventure.objectives import OBJECTIVES
    from grid_adventure.step import Action

    if split not in SPLITS:
        raise ValueError("Unknown layout split")
    seed = 202611090000 + SPLITS.index(split) * 1000000007 + index * 104729
    rng = random.Random(seed)
    width, height = 7 + 2 * (index % 5), 7 + 2 * (index // 5 % 5)
    origin = (1 if index % 2 else width - 2, 1 if index % 3 else height - 2)
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    free = {origin}
    stack = [origin]
    while stack:
        x, y = stack[-1]
        options = [
            (dx, dy)
            for dx, dy in directions
            if 0 < x + 2 * dx < width - 1
            and 0 < y + 2 * dy < height - 1
            and (x + 2 * dx, y + 2 * dy) not in free
        ]
        if not options:
            stack.pop()
            continue
        dx, dy = rng.choice(options)
        free.update(((x + dx, y + dy), (x + 2 * dx, y + 2 * dy)))
        stack.append((x + 2 * dx, y + 2 * dy))
    parents = {origin: None}
    queue = deque([origin])
    last = origin
    while queue:
        last = queue.popleft()
        for dx, dy in directions:
            point = (last[0] + dx, last[1] + dy)
            if point in free and point not in parents:
                parents[point] = last
                queue.append(point)
    route = []
    point = last
    while point is not None:
        route.append(point)
        point = parents[point]
    route.reverse()
    family = FAMILIES[index % 10]
    state = GridState(
        width=width,
        height=height,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=150,
        step_cost=3,
    )
    state.add(origin, AgentEntity())
    state.add(route[-1], ExitEntity())
    for x in range(width):
        for y in range(height):
            if (x, y) not in free:
                state.add((x, y), WallEntity())
    occupied = {origin, route[-1]}
    pickups = {}
    locked = set()

    def place(n, kind):
        point = route[n]
        if point in occupied:
            raise ValueError("Overlapping initial generation")
        occupied.add(point)
        state.add(point, ENTITY_TYPES[kind]())
        if kind in ("coin", "gem", "key", "shield", "boots", "ghost"):
            pickups[n] = kind
        if kind == "locked":
            locked.add(n)

    if family in ("multi-gem", "combined"):
        place(len(route) - 3, "gem")
        if multiple_gems:
            place(len(route) - 2, "gem")
    if family in ("coins", "combined"):
        place(len(route) - 5, "coin")
    if family in ("keys-doors", "combined"):
        place(1, "key")
        place(3, "locked")
    if family in ("lava-shields", "mixed-effects", "combined"):
        place(2, "shield")
        for n in (4, 5):
            place(n, "lava")
    if family in ("boots", "mixed-effects"):
        place(1, "boots")
    if family in ("phasing", "mixed-effects"):
        place(6, "ghost")
    if family in ("keys-doors", "combined"):
        # Include the fourteenth class in independently held-out initial boards,
        # in addition to later frames produced by unlocking a door.
        point = max(free - occupied)
        state.add(point, ENTITY_TYPES["opened"]())
        occupied.add(point)
    if family in ("boxes", "combined"):
        options = sorted(free - set(route) - occupied)
        if options:
            state.add(rng.choice(options), ENTITY_TYPES["box"]())
        else:
            # An optional box off the witness preserves solvability/no overlaps.
            walls = sorted(
                (x, y)
                for x in range(1, width - 1)
                for y in range(1, height - 1)
                if (x, y) not in free
            )
            point = rng.choice(walls)
            state.grid[point[0]][point[1]].clear()
            state.add(point, ENTITY_TYPES["box"]())
    actions = []
    mapping = {
        (-1, 0): Action.LEFT,
        (1, 0): Action.RIGHT,
        (0, -1): Action.UP,
        (0, 1): Action.DOWN,
    }
    for n, (p, q) in enumerate(pairwise(route), 1):
        if n in locked:
            actions.append(Action.USE_KEY)
        actions.append(mapping[(q[0] - p[0], q[1] - p[1])])
        if n in pickups:
            actions.append(Action.PICK_UP)
            if pickups[n] in ("boots", "ghost"):
                actions.extend([Action.WAIT] * 5)
    if len(actions) >= 150:
        raise ValueError("Witness exceeds episode horizon")
    return state, tuple(actions), family, seed


def layout_record(index, split, *, multiple_gems=False):
    from tools.benchmark_vision_v3 import oracle_labels

    state, witness, family, seed = new_layout(
        index, split=split, multiple_gems=multiple_gems
    )
    payload = {
        "width": state.width,
        "height": state.height,
        "labels": oracle_labels(state),
    }
    record = {
        "index": index,
        "family": family,
        "seed": seed,
        "split": split,
        "sha256": digest(canonical(payload)),
        "layout": payload,
        "witness": [a.name for a in witness],
        "appearance_seeds": [2026110901 + i * 7919 + index * 104729 for i in range(3)],
    }
    if multiple_gems:
        record["multiple_gems"] = True
    return record


def make_manifest(asset_root: Path, seed: int) -> dict:
    asset_root = Path(asset_root).resolve()
    assets, features = [], []
    for path in sorted(asset_root.rglob("*.png")):
        if path.parent.name not in CLASSES:
            continue
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            pixels = np.asarray(rgba.resize((32, 32)), dtype=np.float32) / 255
            pixel_hash = digest(rgba.tobytes() + canonical(rgba.size))
        assets.append(
            {
                "path": path.relative_to(asset_root).as_posix(),
                "class": path.parent.name,
                "sha256": digest(path.read_bytes()),
                "pixel_sha256": pixel_hash,
            }
        )
        features.append(pixels)
    parent = list(range(len(assets)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, asset in enumerate(assets):
        for j in range(i):
            # Global exact decoded duplicates; conservative same-class near
            # duplicates: <=2.5% RMS on normalized 32px RGBA. Transitive groups.
            identical = asset["pixel_sha256"] == assets[j]["pixel_sha256"]
            near = (
                asset["class"] == assets[j]["class"]
                and float(np.sqrt(np.mean((features[i] - features[j]) ** 2))) <= 0.025
            )
            if identical or near:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(assets)):
        groups.setdefault(find(i), []).append(i)
    assigned = {}
    for kind in CLASSES:
        candidates = [
            g
            for g, indices in groups.items()
            if any(assets[i]["class"] == kind for i in indices)
        ]
        candidates.sort(
            key=lambda g: digest(
                canonical([seed, sorted(assets[i]["sha256"] for i in groups[g])])
            )
        )
        n = len(candidates)
        ntrain = max(1, round(n * 0.6)) if n else 0
        nval = max(1, round(n * 0.2)) if n >= 3 else 0
        ntrain = min(ntrain, n - nval - (1 if n >= 3 else 0))
        for position, group in enumerate(candidates):
            chosen = (
                "train"
                if position < ntrain
                else "validation"
                if position < ntrain + nval
                else "final"
            )
            # A group shared across classes has one global assignment. Counts
            # may depart from the requested ratio; the manifest exposes them.
            assigned.setdefault(group, chosen)
    for group, indices in groups.items():
        identity = digest(canonical(sorted(assets[i]["pixel_sha256"] for i in indices)))
        for i in indices:
            assets[i].update(group=identity, split=assigned[group])
    layouts, seen_layouts = [], set()
    for split, count in (("train", 200), ("validation", 40), ("final", 100)):
        for index in range(count):
            record = layout_record(index, split)
            # Distinct seeds can generate the same small maze. Reject by actual
            # semantic hash, keeping the family and geometry on each retry.
            while record["sha256"] in seen_layouts:
                index += 1000
                record = layout_record(index, split)
            seen_layouts.add(record["sha256"])
            layouts.append(record)
    manifest = {
        "schema": 1,
        "seed": seed,
        "asset_root": str(asset_root),
        "assets": assets,
        "layouts": layouts,
        "renderer_version": version("grid-universe"),
        "grouping": {
            "exact": "decoded RGBA pixels and geometry",
            "near": "same-class 32px RGBA RMS <= 0.025, transitive union",
        },
        "class_order": list(CLASSES),
        "counts": {
            split: dict(Counter(a["class"] for a in assets if a["split"] == split))
            for split in SPLITS
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    groups, pixels, layouts = {}, {}, {}
    for asset in manifest["assets"]:
        split = asset["split"]
        if split not in SPLITS:
            raise ValueError("Unknown split")
        for mapping, key in ((groups, asset["group"]), (pixels, asset["pixel_sha256"])):
            if key in mapping and mapping[key] != split:
                raise ValueError("Asset group crosses splits")
            mapping[key] = split
    for layout in manifest["layouts"]:
        key, split = layout["sha256"], layout["split"]
        if key in layouts:
            raise ValueError("Duplicate layout hash")
        layouts[key] = split


def _isolated_root(manifest, split, output):
    source = Path(manifest["asset_root"])
    if not source.is_dir():
        # Recorded host paths may differ; the frozen hashes still bind the inputs.
        source = asset_root()
    # Check all constituents against the frozen inventory before rendering.
    for entry in manifest["assets"]:
        path = source / entry["path"]
        if not path.is_file() or digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError(
                "Asset bytes differ from frozen manifest: " + entry["path"]
            )
    root = output / "asset-root"
    for entry in manifest["assets"]:
        if entry["split"] == split:
            target = root / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry["path"], target)
    expected = {a["path"] for a in manifest["assets"] if a["split"] == split}
    if {p.relative_to(root).as_posix() for p in root.rglob("*.png")} != expected:
        raise ValueError("Rendering root contains unassigned assets")
    return root


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
        selected[kind] = rng.choice(entries)
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


def render_boards(
    manifest, geometry_split, appearance_split, output, *, count=None, appearances=1
):
    """Save geometry/appearance axes independently. No classifier is evaluated."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    layouts = [r for r in manifest["layouts"] if r["split"] == geometry_split]
    if count is not None:
        layouts = layouts[:count]
    for layout in layouts:
        state, _, _, _ = new_layout(
            layout["index"],
            split=geometry_split,
            multiple_gems=layout.get("multiple_gems", False),
        )
        for appearance in range(appearances):
            record = render_frame(
                manifest,
                appearance_split,
                state,
                layout["appearance_seeds"][appearance],
                output,
                resolution=(640, 480, 512)[appearance % 3],
            )
            record.update(
                layout_sha256=layout["sha256"],
                geometry_split=geometry_split,
                family=layout["family"],
                layout_seed=layout["seed"],
            )
            records.append(record)
    path = output / "boards.jsonl"
    path.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in records)
    )
    report = {
        "boards": len(records),
        "geometry_split": geometry_split,
        "appearance_split": appearance_split,
        "records_sha256": digest(path.read_bytes()),
        "manifest_sha256": digest(canonical(manifest)),
        "family_counts": dict(Counter(r["family"] for r in records)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


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
            selected = {"floor": rng.choice(inventory["floor"])}
            if kind != "floor":
                selected[kind] = rng.choice(inventory[kind])
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


def appearance_features(images):
    features = [
        np.asarray(
            Image.fromarray(image).resize((8, 8), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).reshape(-1)
        / 255
        for image in images
    ]
    return np.stack(features)


def fit_linear_baseline(images, labels, *, regularization=0.1):
    """Small ridge classifier fitted only on caller-supplied training examples."""
    features = appearance_features(images).astype(np.float64)
    features = np.column_stack((features, np.ones(len(features))))
    classes = np.unique(labels)
    target = np.stack([np.asarray(labels) == kind for kind in classes], axis=1)
    weights = np.linalg.solve(
        features.T @ features + regularization * np.eye(features.shape[1]),
        features.T @ target,
    )
    return {"weights": weights.astype(np.float32), "classes": classes}


def linear_predict(model, images):
    features = appearance_features(images)
    features = np.column_stack((features, np.ones(len(features))))
    return model["classes"][(features @ model["weights"]).argmax(axis=1)]


def template_catalog(root):
    """Optimistic all-public-art template diagnostic, never a promoted model."""
    root = Path(root)
    floor_paths = sorted((root / "floor").glob("*.png"))
    images, labels, paths = [], [], set()
    for ci, kind in enumerate(CLASSES):
        for path in sorted((root / kind).glob("*.png")):
            paths.add(path.relative_to(root).as_posix())
            with Image.open(path) as source:
                sprite = source.convert("RGBA").resize((48, 48))
            backgrounds = [path] if kind == "floor" else floor_paths
            for floor_path in backgrounds:
                with Image.open(floor_path) as source:
                    floor = source.convert("RGBA").resize((48, 48))
                image = (
                    floor if kind == "floor" else Image.alpha_composite(floor, sprite)
                )
                images.append(np.asarray(image.convert("RGB")))
                labels.append(ci)
    images = np.stack(images)
    return {
        "images": images,
        "features": appearance_features(images),
        "labels": labels,
        "paths": sorted(paths),
        "evidence_boundary": "All public artwork; no unseen-art generalization claim.",
    }


def template_predict(catalog, images):
    features, templates = appearance_features(images), catalog["features"]
    distance = (
        (features * features).sum(axis=1)[:, None]
        + (templates * templates).sum(axis=1)[None, :]
        - 2 * features @ templates.T
    )
    return np.asarray(catalog["labels"])[distance.argmin(axis=1)]


def audit_boards(manifest, output):
    """Validate saved images/semantic layouts and every constituent, no model."""
    validate_manifest(manifest)
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    if report["manifest_sha256"] != digest(canonical(manifest)):
        raise ValueError("Board manifest mismatch")
    records = output / "boards.jsonl"
    if digest(records.read_bytes()) != report["records_sha256"]:
        raise ValueError("Board records changed")
    assets = {a["path"]: a for a in manifest["assets"]}
    layouts = {r["sha256"]: r for r in manifest["layouts"]}
    classes, families, pixels = Counter(), Counter(), {}
    count = 0
    for record in map(json.loads, records.read_text().splitlines()):
        path = output / record["image"]
        if digest(path.read_bytes()) != record["image_sha256"]:
            raise ValueError("Board image changed")
        layout = layouts[record["layout_sha256"]]
        if canonical(layout["layout"]["labels"]) != canonical(record["labels"]):
            raise ValueError("Board semantic labels changed")
        for entry in record["constituents"]:
            if (
                assets[entry["path"]] != entry
                or entry["split"] != record["appearance_split"]
            ):
                raise ValueError("Board constituent crosses split or has drifted")
        with Image.open(path) as image:
            pixels[record["image"]] = digest(np.asarray(image).tobytes())
        classes.update(k for row in record["labels"] for cell in row for k in cell)
        families[record["family"]] += 1
        count += 1
    if count != report["boards"]:
        raise ValueError("Board count mismatch")
    result = {
        "verified_boards": count,
        "violations": 0,
        "manifest_sha256": report["manifest_sha256"],
        "records_sha256": report["records_sha256"],
        "class_counts": dict(classes),
        "family_counts": dict(families),
        "pixels_sha256": pixels,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def audit_dataset(manifest, output):
    """Verify actual shard bytes, labels, counts and every constituent record."""
    validate_manifest(manifest)
    output = Path(output)
    report = json.loads((output / "report.json").read_text())
    if report["manifest_sha256"] != digest(canonical(manifest)):
        raise ValueError("Dataset manifest mismatch")
    sample_path = output / "samples.jsonl"
    if digest(sample_path.read_bytes()) != report["samples_sha256"]:
        raise ValueError("Sample manifest drift")
    assets = {a["path"]: a for a in manifest["assets"]}
    counts = Counter()
    verified = 0
    with sample_path.open() as samples:
        for shard in report["shards"]:
            path = output / shard["path"]
            if digest(path.read_bytes()) != shard["sha256"]:
                raise ValueError("Shard bytes changed")
            with np.load(path, allow_pickle=False) as arrays:
                images, labels = arrays["images"], arrays["labels"]
                if (
                    images.shape != (shard["samples"], 48, 48, 3)
                    or len(labels) != shard["samples"]
                ):
                    raise ValueError("Shard shape/count mismatch")
                for image, label in zip(images, labels, strict=True):
                    sample = json.loads(next(samples))
                    if (
                        sample["index"] != verified
                        or sample["class"] != CLASSES[int(label)]
                    ):
                        raise ValueError("Sample label/index drift")
                    if sample["output_sha256"] != digest(image.tobytes()):
                        raise ValueError("Output image hash mismatch")
                    classes = set()
                    for constituent in sample["constituents"]:
                        if (
                            assets.get(constituent["path"]) != constituent
                            or constituent["split"] != report["split"]
                        ):
                            raise ValueError("Unattributed or cross-split constituent")
                        classes.add(constituent["class"])
                    if "floor" not in classes or sample["class"] not in classes:
                        raise ValueError("Foreground/floor attribution missing")
                    counts[sample["class"]] += 1
                    verified += 1
        if samples.readline():
            raise ValueError("Extra sample records")
    if verified != report["crops"] or dict(counts) != report["class_counts"]:
        raise ValueError("Dataset count mismatch")
    result = {
        "verified_crops": verified,
        "shard_count": len(report["shards"]),
        "class_counts": dict(counts),
        "provenance_violations": 0,
        "manifest_sha256": report["manifest_sha256"],
        "data_report_sha256": digest((output / "report.json").read_bytes()),
        "audit_source_sha256": digest(Path(__file__).read_bytes()),
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--split", choices=SPLITS)
    parser.add_argument("--crops", type=int, default=200000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--boards", action="store_true")
    parser.add_argument("--geometry-split", choices=SPLITS, default="validation")
    parser.add_argument("--appearances", type=int, default=1)
    args = parser.parse_args()
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        validate_manifest(manifest)
    else:
        manifest = make_manifest(args.asset_root or asset_root(), args.seed)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print("assets", len(manifest["assets"]), "counts", manifest["counts"], flush=True)
    if args.split:
        if args.output is None:
            parser.error("--output required when rendering")
        result = (
            render_boards(
                manifest,
                args.geometry_split,
                args.split,
                args.output,
                appearances=args.appearances,
            )
            if args.boards
            else render_dataset(manifest, args.split, args.output, args.crops)
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
