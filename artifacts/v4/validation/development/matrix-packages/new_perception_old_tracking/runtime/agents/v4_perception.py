"""Lazy CPU perception: runtime model bytes are supplied by the V3 package."""

from agents.v4_bridge import CLASSES, split_cells
from agents.v4_types import CellScores, PerceptionEvidence, PerceptionPolicy

_loader = None
_model = None
_temperature = 1.0
_policy = PerceptionPolicy(1.0, 0.85, 0.20, "full")


def configure(loader, temperature=1.0, *, policy=None):
    global _loader, _model, _temperature, _policy
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    selected = policy or PerceptionPolicy(float(temperature), 0.85, 0.20, "full")
    if selected.temperature != temperature or selected.view not in (
        "full",
        "inset_005",
    ):
        raise ValueError("Model calibration/policy mismatch")
    _loader, _model, _temperature = loader, None, float(temperature)
    _policy = selected


def _get_model():
    global _model
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Inference thread configuration failed")
    if _model is None:
        if _loader is None:
            raise RuntimeError("V3 vision model has not been configured")
        _model = _loader().eval()
    return _model


def classify(observation, cells=None):
    import numpy as np
    import torch
    from PIL import Image

    model = _get_model()
    rows = split_cells(observation)
    height, width = len(rows), len(rows[0])
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
    crops = np.stack(
        [
            np.asarray(
                Image.fromarray(rows[y][x])
                .convert("RGB")
                .resize((48, 48), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ).transpose(2, 0, 1)
            / 255
            for x, y in selected
        ]
    )
    with torch.inference_mode():
        values = torch.softmax(
            model(torch.from_numpy(crops)) / _temperature, dim=1
        ).numpy()
    for (x, y), value in zip(selected, values):
        probabilities[y, x] = value
    return CellScores(probabilities, width, height)


def classify_subcrops(observation, cells, inset=0.10):
    """Batched interior views for temporal actor/composite localization only."""
    import numpy as np
    from PIL import Image

    if not 0 <= inset < 0.25:
        raise ValueError("Unsupported subcrop inset")
    rows = split_cells(observation)
    height, width = len(rows), len(rows[0])
    values = np.full((height, width, len(CLASSES)), np.nan, np.float32)
    cells = tuple(cells)
    if not cells:
        return CellScores(values, width, height)
    regions = []
    for x, y in cells:
        crop = rows[y][x]
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
    """Classify measured changed pixels over a neutral background.

    Removes static lava/wall texture when it hides the actor from the full-cell
    classifier. Neither predictions nor symbolic labels supply image pixels.
    """
    import numpy as np
    from PIL import Image

    rows = split_cells(observation)
    old = split_cells({**observation, "image": previous_image})
    height, width = len(rows), len(rows[0])
    values = np.full((height, width, len(CLASSES)), np.nan, np.float32)
    cells = tuple(cells)
    crops = []
    for x, y in cells:
        current, before = rows[y][x][:, :, :3], old[y][x][:, :, :3]
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
    floor = min(
        (crop(p) for p in floor_cells), key=lambda v: np.abs(v - actor)[border].mean()
    )
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
    rows = split_cells(observation)
    ranked = []
    for x, y in cells:
        crop = np.asarray(
            Image.fromarray(rows[y][x])
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
    """A small shape view resolves the diagnosed gem/shield confusion pair.

    Structural cells retain the full-cell scale; cropped box/door borders hurt
    validation. No objective or planner ranks the visual labels.
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
    return _policy


def classify_evidence(observation, cells=None):
    """Bounded views; disagreement survives even when fusion is confident."""
    import numpy as np

    scores = classify(observation, cells)
    probabilities = scores.probabilities
    disagreements = []
    if _policy.view == "inset_005":
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
    return PerceptionEvidence(scores, tuple(disagreements), _policy.view)

# Immutable ablation: legacy tracker consumes selected fused scores only.
def classify_initial(observation, cells=None):
    return classify_evidence(observation, cells).scores
