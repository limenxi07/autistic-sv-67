"""Conditional board tracking with independent image and public metadata checks."""

from collections import Counter
from copy import deepcopy
from itertools import product
from time import perf_counter
from typing import NamedTuple

import numpy as np
from grid_adventure.grid import step as grid_step
from grid_adventure.step import Action

from agents.boss_p2_next import reference_state_key
from agents.vision_v3_bridge import CLASSES, decode_metadata, reconstruct
from agents.vision_v3_perception import (
    classify_foregrounds,
    classify_initial,
    classify_subcrops,
    locate_actor_template,
)
from agents.vision_v3_types import BoardHypothesis, ObservationCheck

MIN_CONFIDENCE = 0.85
MIN_MARGIN = 0.20


class PredictionHistory(NamedTuple):
    """Keep the predicted board tied to the observation turn it describes."""
    hypotheses: tuple
    action: object


def agent_position(state):
    positions = [
        (x, y)
        for x, col in enumerate(state.grid)
        for y, cell in enumerate(col)
        if any(getattr(e, "agent", None) is not None for e in cell)
    ]
    return positions[0] if len(positions) == 1 else None


def entity_kind(entity):
    name = entity.appearance.name
    if name == "door":
        return "locked" if getattr(entity, "locked", None) is not None else "opened"
    return name


def state_labels(state):
    return [
        [
            tuple(entity_kind(e) for e in state.grid[x][y]) or ("floor",)
            for x in range(state.width)
        ]
        for y in range(state.height)
    ]


def visible_kind(cell):
    kinds = tuple(entity_kind(e) for e in cell)
    for kind in (
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
    ):
        if kind in kinds:
            return kind
    return "floor"


def fatal_uncertainty(hypothesis):
    return any(
        not (len(entry) >= 3 and entry[2] == "ambiguous")
        for entry in hypothesis.unresolved
    )


def _enumerate(
    labels, info, alternatives, *, previous=None, confidence=1.0, inherited=()
):
    base = reconstruct(labels, info, previous)
    active = set(alternatives) | {
        (entry[0], entry[1])
        for entry in inherited
        if len(entry) >= 3 and entry[2] == "ambiguous"
    }
    if len(active) > 2:
        return (
            BoardHypothesis(
                base.state,
                0.0,
                base.unresolved
                + inherited
                + (("ambiguity_limit", tuple(alternatives)),),
            ),
        )
    results = []
    for choices in product((0, 1), repeat=len(alternatives)):
        current = [list(row) for row in labels]
        ranking = confidence
        unresolved: list[tuple] = list(inherited)
        for ((x, y), options), choice in zip(alternatives.items(), choices):
            kind, probability = options[choice]
            current[y][x] = kind
            ranking *= probability
            unresolved.append((x, y, "ambiguous", tuple(k for k, _ in options)))
        hypothesis = reconstruct(current, info, previous)
        # Reject invalid configurations; do not alter confident labels to make
        # them solvable, and do not consult a planner to rank visual parses.
        agents = sum(
            "human" in (c if isinstance(c, (list, tuple)) else (c,))
            for row in current
            for c in row
        )
        exits = sum(
            "exit" in (c if isinstance(c, (list, tuple)) else (c,))
            for row in current
            for c in row
        )
        if agents == 1 and exits >= 1 and not hypothesis.unresolved:
            results.append(
                BoardHypothesis(
                    hypothesis.state, ranking, tuple(dict.fromkeys(unresolved))
                )
            )
    if not results:
        return (
            BoardHypothesis(
                base.state,
                0.0,
                base.unresolved + inherited + (("no_complete_candidate",),),
            ),
        )
    return tuple(sorted(results, key=lambda h: -h.confidence))


def initial_hypotheses(scores, info, *, previous=None):
    values = np.asarray(scores.probabilities)
    if (
        values.shape != (scores.height, scores.width, 14)
        or not np.isfinite(values).all()
    ):
        return ()
    labels, alternatives = [], {}
    for y in range(scores.height):
        row = []
        for x in range(scores.width):
            indices = np.argsort(values[y, x])[-2:][::-1]
            a, b = map(int, indices)
            row.append(CLASSES[a])
            if (
                values[y, x, a] < MIN_CONFIDENCE
                or values[y, x, a] - values[y, x, b] < MIN_MARGIN
            ):
                alternatives[(x, y)] = (
                    (CLASSES[a], float(values[y, x, a])),
                    (CLASSES[b], float(values[y, x, b])),
                )
        labels.append(row)
    return _enumerate(labels, info, alternatives, previous=previous)


def _agent(state):
    position = agent_position(state)
    if position is None:
        return None
    return next(
        e
        for e in state.grid[position[0]][position[1]]
        if getattr(e, "agent", None) is not None
    )


def _resource_signatures(agent, entries, *, effects=False):
    objects = agent.status_list if effects else agent.inventory_list
    include_ids = all(e.entity_id >= 0 for e in objects)
    expected, observed = [], []
    for entity in objects:
        kind = entity_kind(entity)
        if effects:
            timer = getattr(entity, "time_limit", None)
            usage = getattr(entity, "usage_limit", None)
            if timer is None and usage is None:
                raise ValueError("Unsupported effect without a limit")
            expected.append(
                (
                    entity.entity_id if include_ids else None,
                    kind,
                    "TIME" if timer is not None else "USAGE",
                    timer.amount if timer is not None else getattr(usage, "amount", -1),
                    entity.speed.multiplier if kind == "boots" else -1,
                )
            )
        else:
            expected.append(
                (
                    entity.entity_id if include_ids else None,
                    kind,
                    entity.key.key_id if kind == "key" else "",
                )
            )
    for entry in entries:
        if effects:
            kind = {"SPEED": "boots", "PHASING": "ghost", "IMMUNITY": "shield"}.get(
                entry["type"], entry["type"]
            )
            observed.append(
                (
                    entry["id"] if include_ids else None,
                    kind,
                    entry["limit_type"],
                    entry["limit_amount"],
                    entry["multiplier"],
                )
            )
        else:
            observed.append(
                (
                    entry["id"] if include_ids else None,
                    entry["appearance_name"],
                    entry["key_id"],
                )
            )
    # New pickups have locally predicted IDs until the public metadata exposes
    # them. Compare complete semantics now, then bind exact public IDs below.
    if not include_ids:
        expected, observed = sorted(expected), sorted(observed)
    elif not effects:
        # Consumed coins/gems are inert records. Set-table resize/deletion can
        # reorder their public entries without changing a single resource.
        # Retain key order separately because key consumption is identity-sensitive.
        expected = (
            tuple(e for e in expected if e[1] == "key"),
            tuple(sorted(e for e in expected if e[1] != "key")),
        )
        observed = (
            tuple(e for e in observed if e[1] == "key"),
            tuple(sorted(e for e in observed if e[1] != "key")),
        )
    return tuple(expected), tuple(observed)


def changed_cells(image, previous_image, width, height):
    ih, iw = image.shape[:2]
    if previous_image is None or image.shape != previous_image.shape:
        return {(x, y) for y in range(height) for x in range(width)}
    if (
        image.dtype == previous_image.dtype == np.uint8
        and image.shape[2] == 4
        and image.flags.c_contiguous
        and previous_image.flags.c_contiguous
    ):
        different = image.view(np.uint32).reshape(ih, iw) != previous_image.view(
            np.uint32
        ).reshape(ih, iw)
    else:
        different = np.any(image[:, :, :3] != previous_image[:, :, :3], axis=2)
    pixels = np.flatnonzero(different)
    xs = ((pixels % iw + 1) * width - 1) // iw
    ys = ((pixels // iw + 1) * height - 1) // ih
    return {(int(i % width), int(i // width)) for i in np.unique(ys * width + xs)}


def check_observation(
    observation,
    predicted,
    previous_image,
    *,
    background_image=None,
    actor_template=None,
):
    meta = decode_metadata(observation["info"])
    mismatches = []
    if (meta["width"], meta["height"]) != (predicted.width, predicted.height):
        return ObservationCheck(False, (), (("geometry",),))
    for field in ("turn", "score"):
        if meta[field] != getattr(predicted, field):
            mismatches.append((field, getattr(predicted, field), meta[field]))
    phase = "win" if predicted.win else "lose" if predicted.lose else "ongoing"
    if meta["phase"] != phase:
        mismatches.append(("phase", phase, meta["phase"]))
    actor = _agent(predicted)
    if actor is None:
        return ObservationCheck(False, (), (("agent_count",),))
    if meta["health"] != vars(actor.health):
        mismatches.append(("health", vars(actor.health), meta["health"]))
    for field in ("inventory", "effects"):
        expected, observed = _resource_signatures(
            actor, meta[field], effects=field == "effects"
        )
        if expected != observed:
            mismatches.append((field, expected, observed))
    image = observation["image"]
    width, height = predicted.width, predicted.height
    changed = changed_cells(image, previous_image, width, height)
    position = agent_position(predicted)
    selected = set(changed)
    for x, y in (position,):
        selected.update(
            (nx, ny)
            for nx, ny in ((x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            if 0 <= nx < width and 0 <= ny < height
        )
    for x, col in enumerate(predicted.grid):
        for y, cell in enumerate(col):
            if any(getattr(e, "pushable", None) is not None for e in cell):
                selected.add((x, y))
    scores = classify_initial(observation, tuple(sorted(selected)))
    human_index, box_index = CLASSES.index("human"), CLASSES.index("box")
    full_humans = [
        (x, y)
        for x, y in selected
        if int(scores.probabilities[y, x].argmax()) == human_index
    ]
    disputed_boxes = {
        (x, y)
        for x, y in selected
        if visible_kind(predicted.grid[x][y]) == "box"
        and int(scores.probabilities[y, x].argmax()) != box_index
    }
    regional_cells = set(selected) if full_humans != [position] else disputed_boxes
    if regional_cells:
        regions = classify_subcrops(observation, tuple(sorted(regional_cells)))
        for x, y in regional_cells:
            probabilities = regions.probabilities[y, x]
            kind = int(probabilities.argmax())
            # Subcrops supply actual visual localization evidence, never a
            # label copied from the predicted state or a solvability constraint.
            if probabilities[kind] >= 0.90 and (
                kind == human_index or kind == box_index and (x, y) in disputed_boxes
            ):
                scores.probabilities[y, x] = probabilities
        if previous_image is not None and previous_image.shape == image.shape:
            foregrounds = classify_foregrounds(
                observation,
                previous_image if background_image is None else background_image,
                tuple(sorted(regional_cells)),
            )
            for x, y in regional_cells:
                probabilities = foregrounds.probabilities[y, x]
                kind = int(probabilities.argmax())
                if probabilities[kind] >= 0.90 and (
                    kind == human_index
                    or kind == box_index
                    and (x, y) in disputed_boxes
                ):
                    scores.probabilities[y, x] = probabilities
        matched = locate_actor_template(
            observation, tuple(sorted(selected)), actor_template
        )
        if matched is not None:
            x, y = matched
            # One-hot independent corroboration, not a calibrated CNN posterior.
            scores.probabilities[y, x] = 0
            scores.probabilities[y, x, human_index] = 1
    humans = []
    for x, y in sorted(selected):
        probabilities = scores.probabilities[y, x]
        a, b = map(int, np.argsort(probabilities)[-2:][::-1])
        kind = CLASSES[a]
        if kind == "human":
            humans.append((x, y))
        expected = visible_kind(predicted.grid[x][y])
        if kind != expected:
            mismatches.append(
                ("cell", (x, y), expected, kind, tuple(map(float, probabilities)))
            )
        elif (
            probabilities[a] < MIN_CONFIDENCE
            or probabilities[a] - probabilities[b] < MIN_MARGIN
        ):
            mismatches.append(
                (
                    "uncertain_cell",
                    (x, y),
                    expected,
                    kind,
                    tuple(map(float, probabilities)),
                )
            )
    observed_position = humans[0] if len(humans) == 1 else None
    if observed_position != position:
        mismatches.append(("agent_position", position, observed_position))
    return ObservationCheck(
        not mismatches, tuple(sorted(changed)), tuple(mismatches), scores
    )


def _restore_unconfirmed_pickup(state, before, action, info, labels):
    actor = _agent(state)
    expected = Counter(
        entity_kind(e) for e in (*actor.inventory_list, *actor.status_list)
    )
    meta = decode_metadata(info)
    observed = Counter(e["appearance_name"] for e in meta["inventory"])
    observed.update(
        {"SPEED": "boots", "PHASING": "ghost", "IMMUNITY": "shield"}[e["type"]]
        for e in meta["effects"]
    )
    if sum(expected.values()) <= sum(observed.values()):
        return state, ()  # A type correction is not a failed pickup.
    missing = expected - observed
    unresolved = []
    current = state
    restored = False
    if before is not None and action == Action.PICK_UP:
        point = agent_position(before)
        for entity in before.grid[point[0]][point[1]]:
            kind = entity_kind(entity)
            if getattr(entity, "collectible", None) is None or not missing[kind]:
                continue
            label = labels[point[1]][point[0]]
            visible = (
                label
                if isinstance(label, str)
                else visible_kind(state.grid[point[0]][point[1]])
            )
            if visible not in ("human", "box", kind):
                unresolved.append(("pickup_history_contradiction", point, kind))
                continue
            if not restored:
                current = deepcopy(state)
                restored = True
            current.add(point, deepcopy(entity))
            # A covered item remains a history layer, not an absent detection.
            labels[point[1]][point[0]] = visible
            missing[kind] -= 1
    for kind in ("gem", "coin", "key"):
        if missing[kind]:
            unresolved.append(("inventory_history_unresolved", kind, missing[kind]))
    return current, tuple(unresolved)


def reconcile(observation, hypotheses, check, budget, *, history=None):
    if perf_counter() >= budget.cutoff or any(
        m[0] == "geometry" for m in check.mismatches
    ):
        return ()
    results = []
    for index, hypothesis in enumerate(hypotheses):
        if perf_counter() >= budget.cutoff:
            return ()
        labels = state_labels(hypothesis.state)
        alternatives = {}
        inherited = hypothesis.unresolved
        if check.observed_scores is not None:
            observed = check.observed_scores.probabilities
            resolved = set()
            for y, x in np.argwhere(np.isfinite(observed).all(axis=2)):
                probabilities = observed[y, x]
                a, b = map(int, np.argsort(probabilities)[-2:][::-1])
                kind = CLASSES[a]
                confident = (
                    probabilities[a] >= MIN_CONFIDENCE
                    and probabilities[a] - probabilities[b] >= MIN_MARGIN
                )
                if kind != visible_kind(hypothesis.state.grid[x][y]):
                    labels[y][x] = kind
                if confident:
                    # A visible actor/box does not resolve an item hidden
                    # underneath it. Confident exposed cells do resolve it.
                    if kind not in ("human", "box"):
                        resolved.add((x, y))
                else:
                    alternatives[(x, y)] = (
                        (CLASSES[a], probabilities[a]),
                        (CLASSES[b], probabilities[b]),
                    )
            inherited = tuple(
                entry
                for entry in inherited
                if not (
                    len(entry) >= 3
                    and entry[2] == "ambiguous"
                    and (entry[0], entry[1]) in resolved
                )
            )
        for mismatch in check.mismatches:
            if mismatch[0] not in ("cell", "uncertain_cell"):
                continue
            _, (x, y), _, observed, probabilities = mismatch
            labels[y][x] = observed
            indices = np.argsort(probabilities)[-2:][::-1]
            a, b = map(int, indices)
            if (
                probabilities[a] < MIN_CONFIDENCE
                or probabilities[a] - probabilities[b] < MIN_MARGIN
            ):
                alternatives[(x, y)] = (
                    (CLASSES[a], probabilities[a]),
                    (CLASSES[b], probabilities[b]),
                )
        before = (
            history.hypotheses[index].state
            if history is not None and index < len(history.hypotheses)
            else None
        )
        previous, unresolved = _restore_unconfirmed_pickup(
            hypothesis.state,
            before,
            history.action if history is not None else None,
            observation["info"],
            labels,
        )
        results.extend(
            _enumerate(
                labels,
                observation["info"],
                alternatives,
                previous=previous,
                confidence=hypothesis.confidence,
                inherited=inherited + unresolved,
            )
        )
    unique = {}
    for hypothesis in results:
        key = reference_state_key(hypothesis.state)
        if key not in unique or hypothesis.confidence > unique[key].confidence:
            unique[key] = hypothesis
    ranked = sorted(unique.values(), key=lambda h: -h.confidence)
    if len(ranked) > 4:
        return tuple(
            BoardHypothesis(
                h.state,
                h.confidence,
                h.unresolved + (("hypothesis_limit", len(ranked)),),
            )
            for h in ranked[:4]
        )
    return tuple(ranked)


def common_safe_prefix(hypotheses, actions, budget):
    if not hypotheses or any(fatal_uncertainty(h) for h in hypotheses):
        return ()
    states = [h.state for h in hypotheses]
    prefix = []
    for action in actions:
        if perf_counter() >= budget.cutoff:
            break
        successors = [grid_step(state, action) for state in states]
        if perf_counter() >= budget.cutoff or any(s.lose for s in successors):
            break
        if (
            len({agent_position(s) for s in successors}) != 1
            or len({s.win for s in successors}) != 1
        ):
            break
        damage = [
            _agent(before).health.current_health - _agent(after).health.current_health
            for before, after in zip(states, successors)
        ]
        if any(d > 0 for d in damage) and len(set(damage)) != 1:
            break
        prefix.append(action)
        states = successors
        if all(s.win for s in states):
            break
    return tuple(prefix)
