"""V4 cell classification with a lazily loaded CPU model."""

from types import SimpleNamespace

from agents.vision_v4_bridge import CLASSES, split_cells
from agents.vision_v4_types import (
    CellScores,
    PerceptionEvidence,
    PerceptionPolicy,
)

# Mutate one shared object so pasted function-scoped notebook cells and module
# execution use the same configuration; rebinding globals splits those scopes.
_perception_runtime = SimpleNamespace(
    loader=None,
    model=None,
    temperature=1.0,
    policy=PerceptionPolicy(1.0, 0.85, 0.20, "full"),
    pixel_cache={},
    pixel_cache_bytes=0,
)


def configure(loader, temperature=1.0, *, policy=None):
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    selected = policy or PerceptionPolicy(float(temperature), 0.85, 0.20, "full")
    if selected.temperature != temperature or selected.view not in (
        "full",
        "inset_005",
    ):
        raise ValueError("Model calibration/policy mismatch")
    _perception_runtime.loader = loader
    _perception_runtime.model = None
    _perception_runtime.temperature = float(temperature)
    _perception_runtime.policy = selected
    _perception_runtime.pixel_cache.clear()
    _perception_runtime.pixel_cache_bytes = 0


def _get_model():
    if _perception_runtime.model is None:
        if _perception_runtime.loader is None:
            raise RuntimeError("V3 vision model has not been configured")
        _perception_runtime.model = _perception_runtime.loader().eval()
    if not getattr(_perception_runtime.model, "numpy_inference", False):
        import torch

        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("Inference thread configuration failed")
    return _perception_runtime.model


def _cell_access(observation):
    """Retain split_cells geometry/index semantics without building unused views."""
    image = observation["image"]
    config = observation["info"]["config"]
    width, height = config["width"], config["height"]
    ih, iw = image.shape[:2]
    if (
        type(width) is int
        and type(height) is int
        and image.ndim == 3
        and width > 0
        and height > 0
        and iw >= width
        and ih >= height
    ):
        xs, ys = range(width), range(height)

        def tile(x, y):
            # range indexing keeps list-style negative indices and rejects
            # invalid coordinates before NumPy's permissive slice handling.
            y, x = ys[y], xs[x]
            return image[
                y * ih // height : (y + 1) * ih // height,
                x * iw // width : (x + 1) * iw // width,
                :,
            ]

        return width, height, tile
    rows = split_cells(observation)
    return len(rows[0]), len(rows), lambda x, y: rows[y][x]


def classify(observation, cells=None):
    import numpy as np
    from PIL import Image

    model = _get_model()
    width, height, tile_at = _cell_access(observation)
    selected = (
        tuple((x, y) for y in range(height) for x in range(width))
        if cells is None
        else tuple(cells)
    )
    if len(set(selected)) != len(selected) or any(
        not (0 <= x < width and 0 <= y < height) for x, y in selected
    ):
        raise ValueError("Invalid selected cells")
    probabilities = np.full((height, width, len(CLASSES)), np.nan, dtype=np.float32)
    if not selected:
        return CellScores(probabilities, width, height)
    cache = _perception_runtime.pixel_cache
    pending = {}
    for x, y in selected:
        tile = tile_at(x, y)
        key = (tile.shape, tile.dtype.str, tile.tobytes())
        if key in cache:
            probabilities[y, x] = cache[key]
        else:
            pending.setdefault(key, []).append((x, y))
    if not pending:
        return CellScores(probabilities, width, height)
    crops = np.stack(
        [
            np.asarray(
                Image.fromarray(tile_at(x, y))
                .convert("RGB")
                .resize((48, 48), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).transpose(2, 0, 1)
            / 255
            for points in pending.values()
            for x, y in points[:1]
        ]
    )
    if getattr(model, "numpy_inference", False):
        logits = model(crops) / _perception_runtime.temperature
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        values /= values.sum(axis=1, keepdims=True)
    else:
        import torch

        with torch.inference_mode():
            values = torch.softmax(
                model(torch.from_numpy(crops)) / _perception_runtime.temperature, dim=1
            ).numpy()
    for (key, points), value in zip(pending.items(), values):
        size = len(key[2])
        if size <= 8 * 1024 * 1024:
            while cache and (
                len(cache) >= 1024
                or _perception_runtime.pixel_cache_bytes + size > 8 * 1024 * 1024
            ):
                oldest = next(iter(cache))
                _perception_runtime.pixel_cache_bytes -= len(oldest[2])
                del cache[oldest]
            cache[key] = value.copy()
            _perception_runtime.pixel_cache_bytes += size
        for x, y in points:
            probabilities[y, x] = value
    return CellScores(probabilities, width, height)


def classify_subcrops(observation, cells, inset=0.10):
    """Batched interior views for temporal actor/composite localization only."""
    import numpy as np
    from PIL import Image

    if not 0 <= inset < 0.25:
        raise ValueError("Unsupported subcrop inset")
    width, height, tile_at = _cell_access(observation)
    values = np.full((height, width, len(CLASSES)), np.nan, np.float32)
    cells = tuple(cells)
    if not cells:
        return CellScores(values, width, height)
    regions = []
    for x, y in cells:
        crop = tile_at(x, y)
        ih, iw = crop.shape[:2]
        crop = crop[
            int(ih * inset) : ih - int(ih * inset),
            int(iw * inset) : iw - int(iw * inset),
        ]
        regions.append(
            np.asarray(
                Image.fromarray(crop)
                .convert("RGB")
                .resize((48, 48), Image.Resampling.BILINEAR)
            )
        )
    tiled = {
        "image": np.concatenate(regions, axis=1),
        "info": {"config": {"width": len(regions), "height": 1}},
    }
    scores = classify(tiled)
    for (x, y), probabilities in zip(cells, scores.probabilities[0]):
        values[y, x] = probabilities
    return CellScores(values, width, height)


def classify_foregrounds(observation, previous_image, cells):
    """Classify changed pixels on a neutral background.

    Use measured pixels only; remove static texture that can hide the actor.
    """
    import numpy as np
    from PIL import Image

    width, height, tile_at = _cell_access(observation)
    _, _, old_at = _cell_access({**observation, "image": previous_image})
    values = np.full((height, width, len(CLASSES)), np.nan, np.float32)
    cells = tuple(cells)
    crops = []
    for x, y in cells:
        current, before = tile_at(x, y)[:, :, :3], old_at(x, y)[:, :, :3]
        mask = (
            np.max(np.abs(current.astype(np.int16) - before.astype(np.int16)), axis=2)
            > 4
        )
        crop = current.copy()
        crop[~mask] = 127
        crops.append(
            np.asarray(
                Image.fromarray(crop).resize((48, 48), Image.Resampling.BILINEAR)
            )
        )
    if crops:
        scores = classify(
            {
                "image": np.concatenate(crops, axis=1),
                "info": {"config": {"width": len(crops), "height": 1}},
            }
        )
        for (x, y), probabilities in zip(cells, scores.probabilities[0]):
            values[y, x] = probabilities
    return CellScores(values, width, height)


def make_actor_template(observation, position, floor_cells):
    """Observe the episode's sprite, using actual empty cells as background."""
    import numpy as np
    from PIL import Image

    if position is None or not floor_cells:
        return None
    rows = split_cells(observation)

    def crop(point):
        x, y = point
        return np.asarray(
            Image.fromarray(rows[y][x])
            .convert("RGB")
            .resize((48, 48), Image.Resampling.BILINEAR)
        ).astype(np.int16)

    actor = crop(position)
    border = np.ones((48, 48), bool)
    border[4:-4, 4:-4] = False

    def distinct_floors():
        seen = set()
        for point in floor_cells:
            x, y = point
            tile = rows[y][x]
            key = (tile.shape, tile.dtype.str, tile.tobytes())
            if key not in seen:
                seen.add(key)
                yield crop(point)

    # Exact pixel duplicates produce identical crops and scores. Keep the first
    # occurrence so the original minimum's tie order is unchanged.
    floor = min(distinct_floors(), key=lambda v: np.abs(v - actor)[border].mean())
    mask = np.abs(actor - floor).max(2) > 40
    mask[:4] = mask[-4:] = False
    mask[:, :4] = mask[:, -4:] = False
    mask[1:-1, 1:-1] &= (
        mask[:-2, 1:-1] & mask[2:, 1:-1] & mask[1:-1, :-2] & mask[1:-1, 2:]
    )
    return (actor, mask) if mask.sum() >= 50 else None


def locate_actor_template(observation, cells, template):
    """Return a unique measured template match, without consulting predictions."""
    import numpy as np
    from PIL import Image

    if template is None:
        return None
    actor, mask = template
    _, _, tile_at = _cell_access(observation)
    ranked = []
    for x, y in cells:
        crop = np.asarray(
            Image.fromarray(tile_at(x, y))
            .convert("RGB")
            .resize((48, 48), Image.Resampling.BILINEAR)
        ).astype(np.int16)
        score = float((np.abs(crop - actor).max(2)[mask] < 40).mean())
        ranked.append((score, (x, y)))
    ranked.sort(reverse=True)
    if (
        ranked
        and ranked[0][0] >= 0.75
        and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= 0.20)
    ):
        return ranked[0][1]
    return None


def classify_initial(observation, cells=None):
    """Use a cropped view to distinguish gems from shields.

    Keep full cells for structures, whose borders matter. Planning never ranks
    the visual labels.
    """
    scores = classify(observation, cells)
    pair = {CLASSES.index("gem"), CLASSES.index("shield")}
    cells = [
        (x, y)
        for y, row in enumerate(scores.probabilities)
        for x, p in enumerate(row)
        if int(p.argmax()) in pair
    ]
    if cells:
        shapes = classify_subcrops(observation, cells, inset=0.05)
        for x, y in cells:
            probabilities = shapes.probabilities[y, x]
            if int(probabilities.argmax()) in pair and probabilities.max() >= 0.98:
                scores.probabilities[y, x] = probabilities
    return scores


def current_policy():
    return _perception_runtime.policy


def classify_evidence(observation, cells=None):
    """Bounded views; disagreement survives even when fusion is confident."""
    import numpy as np

    scores = classify(observation, cells)
    probabilities = scores.probabilities
    disagreements = []
    if _perception_runtime.policy.view == "inset_005":
        rival = {CLASSES.index(k) for k in ("gem", "coin", "ghost", "shield")}
        protected = {
            CLASSES.index(k)
            for k in ("floor", "wall", "box", "exit", "locked", "opened", "lava")
        }
        selected = []
        for y, x in np.argwhere(np.isfinite(probabilities).all(axis=2)):
            winners = np.argsort(probabilities[y, x])[-2:][::-1]
            if int(winners[0]) not in protected and any(
                int(k) in rival for k in winners
            ):
                selected.append((int(x), int(y)))
        if selected:
            inset = classify_subcrops(observation, tuple(selected), inset=0.05)
            for x, y in selected:
                full_winner = int(probabilities[y, x].argmax())
                inset_winner = int(inset.probabilities[y, x].argmax())
                if full_winner != inset_winner:
                    disagreements.append(
                        (x, y, CLASSES[full_winner], CLASSES[inset_winner])
                    )
                probabilities[y, x] = (
                    probabilities[y, x] + inset.probabilities[y, x]
                ) / 2
    return PerceptionEvidence(
        scores, tuple(disagreements), _perception_runtime.policy.view
    )
