"""Compact Task 1 state and planner. Only public constructors/API observations used."""

from bisect import bisect_left
from time import perf_counter
from typing import NamedTuple

from grid_adventure.grid import GridState
from grid_adventure.grid import step as grid_step
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES
from grid_adventure.step import Action


class UnsupportedState(ValueError):
    """A public schema outside the characterized standard mechanics."""


class Slot(NamedTuple):
    identity: int
    kind: str
    point: int
    components: tuple
    key: str = ""
    time: int = -1
    usage: int = -1


class Board(NamedTuple):
    width: int
    height: int
    seed: object
    turn_limit: int
    slots: tuple
    agent: int
    cells: tuple
    walls: frozenset
    exits: frozenset
    gems: int
    coins: int
    items: int
    doors: int
    box_slots: tuple
    identities: tuple


class FastState(NamedTuple):
    position: int
    items: int
    doors: int
    boxes: tuple
    inventory: int
    effects: tuple
    health: int
    max_health: int
    turn: int
    win: bool
    lose: bool
    inventory_order: tuple = ()


_FIELDS = {
    "health": ("current_health", "max_health"),
    "key": ("key_id",),
    "locked": ("key_id",),
    "rewardable": ("amount",),
    "damage": ("amount",),
    "speed": ("multiplier",),
    "time_limit": ("amount",),
    "usage_limit": ("amount",),
    "inventory": ("item_ids",),
    "status": ("effect_ids",),
}
_MARKERS = frozenset(
    (
        "agent",
        "blocking",
        "collectible",
        "collidable",
        "exit",
        "pushable",
        "requirable",
        "immunity",
        "phasing",
    )
)
_IGNORED = frozenset(("entity_id", "appearance", "inventory_list", "status_list"))
_SHAPES = {
    "agent": {"agent", "collidable", "health", "inventory", "status"},
    "wall": {"blocking"},
    "box": {"blocking", "pushable"},
    "coin": {"collectible", "rewardable"},
    "gem": {"collectible", "requirable"},
    "key": {"collectible", "key"},
    "door": {"blocking", "locked"},
    "lava": {"collidable", "damage"},
    "exit": {"exit"},
    "open": set(),
    "boots": {"collectible", "speed", "time_limit"},
    "ghost": {"collectible", "phasing", "time_limit"},
    "shield": {"collectible", "immunity", "usage_limit"},
}


def _components(entity):
    result = []
    for name, value in vars(entity).items():
        if value is None or name in _IGNORED:
            continue
        if name not in _FIELDS and name not in _MARKERS:
            raise UnsupportedState("Unknown component: " + name)
        fields = _FIELDS.get(name, ())
        if set(vars(value)) != set(fields):
            raise UnsupportedState("Unknown fields on " + name)
        values = tuple(
            (
                f,
                tuple(sorted(getattr(value, f)))
                if isinstance(getattr(value, f), set)
                else getattr(value, f),
            )
            for f in fields
        )
        result.append((name, values))
    return tuple(sorted(result))


def _kind(components, location):
    names = {name for name, _ in components}
    for kind, shape in _SHAPES.items():
        expected = shape - (
            {"collectible"} if location != "grid" and kind != "key" else set()
        )
        if names == expected:
            return kind
    raise UnsupportedState("Uncharacterized entity shape: " + repr(names))


def _scan(observation):
    for x, column in enumerate(observation.grid):
        for y, cell in enumerate(column):
            for entity in cell:
                yield entity, x + y * observation.width, "grid"
                if getattr(entity, "agent", None) is not None:
                    for item in entity.inventory_list:
                        yield item, -1, "inventory"
                    for effect in entity.status_list:
                        yield effect, -1, "effect"


def _check_grid(s):
    allowed = {
        "width",
        "height",
        "movement",
        "objective",
        "seed",
        "turn",
        "score",
        "step_cost",
        "win",
        "lose",
        "message",
        "turn_limit",
        "grid",
    }
    if not isinstance(s, GridState) or set(vars(s)) - allowed:
        raise UnsupportedState("Unknown observation fields")
    if (
        s.movement != MOVEMENTS["cardinal"]
        or s.objective != OBJECTIVES["collect_gems_and_exit"]
        or s.step_cost != 3
        or not isinstance(s.turn_limit, int)
        or not 1 <= s.turn_limit <= 150
        or not 1 <= s.width <= 15
        or not 1 <= s.height <= 15
        or len(s.grid) != s.width
        or any(len(c) != s.height for c in s.grid)
        or not isinstance(s.score, int)
    ):
        raise UnsupportedState("Unsupported grid configuration")


def encode(observation):
    """Standard cardinal collect-gems/exit schema, integer rewards/counters.

    Every non-None component/field is checked; custom movement/objectives,
    entity extensions, and changed geometry reject to the reference backend.
    Identity slots include inventory/effects, so repairs need not start at zero.
    """
    _check_grid(observation)
    # Simultaneous pickup/overlapping-lock iteration order depends on wrapper
    # entity-table history. These configurations use the verified public backend.
    for column in observation.grid:
        for cell in column:
            if (
                sum(getattr(e, "collectible", None) is not None for e in cell) > 1
                or sum(getattr(e, "locked", None) is not None for e in cell) > 1
            ):
                raise UnsupportedState("Simultaneous pickups or overlapping locks")
    slots = []
    for entity, point, location in _scan(observation):
        components = _components(entity)
        kind = _kind(components, location)
        values = {name: dict(fields) for name, fields in components}
        for field in ("time_limit", "usage_limit"):
            if field in values:
                counter = values[field]["amount"]
                if type(counter) is not int or counter <= 0:
                    raise UnsupportedState("Uncharacterized power counter")
        if (
            "rewardable" in values
            and values["rewardable"]["amount"] != 5
            or "damage" in values
            and values["damage"]["amount"] != 2
            or "speed" in values
            and values["speed"]["multiplier"] != 2
        ):
            raise UnsupportedState("Nonstandard amounts")
        slots.append(
            Slot(
                entity.entity_id,
                kind,
                point,
                components,
                values.get("key", values.get("locked", {})).get("key_id", ""),
                values.get("time_limit", {}).get("amount", -1),
                values.get("usage_limit", {}).get("amount", -1),
            )
        )
    slots.sort(key=lambda slot: slot.identity)
    if len({s.identity for s in slots}) != len(slots):
        raise UnsupportedState("Duplicate entity references")
    agents = [i for i, s in enumerate(slots) if s.kind == "agent"]
    if len(agents) != 1:
        raise UnsupportedState("Expected one agent")

    def mask(kinds):
        return sum(
            1 << i for i, s in enumerate(slots) if s.kind in kinds and s.point >= 0
        )

    identity_slots = {slot.identity: i for i, slot in enumerate(slots)}
    cells = tuple(
        tuple(identity_slots[e.entity_id] for e in observation.grid[x][y])
        for y in range(observation.height)
        for x in range(observation.width)
    )
    board = Board(
        observation.width,
        observation.height,
        observation.seed,
        observation.turn_limit,
        tuple(slots),
        agents[0],
        cells,
        frozenset(s.point for s in slots if s.kind == "wall"),
        frozenset(s.point for s in slots if s.kind == "exit"),
        mask({"gem"}),
        mask({"coin"}),
        mask({"coin", "gem", "key", "boots", "ghost", "shield"}),
        mask({"door"}),
        tuple(i for i, s in enumerate(slots) if s.kind == "box"),
        tuple(slot.identity for slot in slots),
    )
    state, score = read_state(board, observation)
    return board, state, score


def read_state(board, observation):
    _check_grid(observation)
    if (
        observation.width,
        observation.height,
        observation.seed,
        observation.turn_limit,
    ) != (board.width, board.height, board.seed, board.turn_limit):
        raise UnsupportedState("Episode geometry changed")
    lookup = {s.identity: i for i, s in enumerate(board.slots)}
    seen = set()
    items = doors = inventory = 0
    inventory_order = []
    effects = []
    boxes = [-1] * len(board.box_slots)
    position = health = max_health = -1
    for entity, point, location in _scan(observation):
        i = lookup.get(entity.entity_id)
        if i is None or i in seen:
            raise UnsupportedState("Unknown/duplicate entity")
        seen.add(i)
        slot = board.slots[i]
        components = _components(entity)
        kind = _kind(components, location)
        if kind != slot.kind and not (slot.kind == "door" and kind == "open"):
            raise UnsupportedState("Entity kind changed")
        values = dict(components)
        if kind == "door" and values.get("locked") != dict(slot.components).get(
            "locked"
        ):
            raise UnsupportedState("Lock identity changed")
        # The only changing component fields are health, reference sets,
        # effect counters, and collectible/lock removal.
        old = dict(slot.components)
        for name in set(values) | set(old):
            if name not in (
                "health",
                "inventory",
                "status",
                "time_limit",
                "usage_limit",
                "collectible",
                "locked",
                "blocking",
            ) and old.get(name) != values.get(name):
                raise UnsupportedState("Immutable component changed")
        if kind == "agent":
            if location != "grid":
                raise UnsupportedState("Agent off board")
            position = point
            health, max_health = entity.health.current_health, entity.health.max_health
            if entity.inventory.item_ids or entity.status.effect_ids:
                raise UnsupportedState("Inconsistent reference sets")
        elif kind == "box":
            boxes[board.box_slots.index(i)] = point
        elif location == "inventory":
            if kind not in ("coin", "gem", "key"):
                raise UnsupportedState("Unknown carried item")
            inventory |= 1 << i
            inventory_order.append(i)
        elif location == "effect":
            if kind not in ("boots", "ghost", "shield"):
                raise UnsupportedState("Unknown active effect")
            timer = getattr(getattr(entity, "time_limit", None), "amount", -1)
            usage = getattr(getattr(entity, "usage_limit", None), "amount", -1)
            if (
                kind in ("boots", "ghost") and (type(timer) is not int or timer <= 0)
            ) or (kind == "shield" and (type(usage) is not int or usage <= 0)):
                raise UnsupportedState("Invalid counter")
            effects.append((i, timer, usage))
        else:
            if point != slot.point:
                raise UnsupportedState("Static entity moved")
            if kind in ("coin", "gem", "key", "boots", "ghost", "shield"):
                if components != slot.components:
                    raise UnsupportedState("Uncollected item changed")
                items |= 1 << i
            elif kind == "door":
                doors |= 1 << i
    required = {
        i
        for i, s in enumerate(board.slots)
        if s.kind in ("agent", "wall", "box", "exit", "lava", "door", "open")
    }
    if not required <= seen or position < 0 or -1 in boxes:
        raise UnsupportedState("Missing persistent entity")
    return FastState(
        position,
        items,
        doors,
        tuple(boxes),
        inventory,
        tuple(effects),
        health,
        max_health,
        observation.turn,
        observation.win,
        observation.lose,
        tuple(inventory_order),
    ), observation.score


def predicted_key(board, state, score):
    return state, score


def observe_key(board, observation):
    state, score = read_state(board, observation)
    return predicted_key(board, state, score)


_DIRECTIONS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}


def _neighbor(board, point, dx, dy):
    x, y = point % board.width + dx, point // board.width + dy
    return x + y * board.width if 0 <= x < board.width and 0 <= y < board.height else -1


def _reference_set(identities):
    # Public constructor/to_state probes establish incremental union semantics.
    # Rebuilding by set(generator) changes table sizing and resource iteration.
    result = set()
    for identity in identities:
        result = result | {identity}
    return result


def fast_step(board, state, action):
    if action not in tuple(Action):
        raise ValueError("Action is not valid")
    if state.win or state.lose:
        return state, 0
    (
        position,
        items,
        doors,
        boxes,
        inventory,
        effects,
        health,
        max_health,
        turn,
        win,
        lose,
        inventory_order,
    ) = state
    inventory_ids = _reference_set(board.slots[i].identity for i in inventory_order)
    effect_ids = _reference_set(board.slots[i].identity for i, _, _ in effects)
    effect_values = {i: (i, t, u) for i, t, u in effects}
    effects = [effect_values[i] for i in sorted(effect_values)]
    boxes = list(boxes)
    initial_effects = {i for i, _, _ in effects}
    phase = any(board.slots[i].kind == "ghost" for i, _, _ in effects)
    speed = 2 if any(board.slots[i].kind == "boots" for i, _, _ in effects) else 1
    reward = 0

    def contact():
        nonlocal health, win, lose, effect_ids
        # Win/death flags latch even at intermediate speed substeps.
        if position in board.exits and not items & board.gems:
            win = True
        if not phase:
            for i in board.cells[position]:
                if board.slots[i].kind != "lava":
                    continue
                shield = next(
                    (
                        j
                        for j, (slot, _, u) in enumerate(effects)
                        if board.slots[slot].kind == "shield" and u > 0
                    ),
                    None,
                )
                if shield is None:
                    health = max(0, health - 2)
                else:
                    slot, t, u = effects[shield]
                    if u <= 1:
                        effects.pop(shield)
                        effect_ids = effect_ids - {board.slots[slot].identity}
                    else:
                        effects[shield] = slot, t, u - 1
        if health <= 0:
            lose = True

    def blocked(point):
        return point in board.walls or any(
            doors & (1 << i)
            for i in board.cells[point]
            if board.slots[i].kind == "door"
        )

    if action in _DIRECTIONS:
        dx, dy = _DIRECTIONS[action]
        for _ in range(speed):
            target = _neighbor(board, position, dx, dy)
            if target >= 0 and (phase or not blocked(target)):
                if not phase and target in boxes:
                    destination = _neighbor(board, target, dx, dy)
                    if (
                        destination >= 0
                        and destination not in boxes
                        and not blocked(destination)
                    ):
                        boxes = [
                            destination if point == target else point for point in boxes
                        ]
                        position = target
                else:
                    position = target
            contact()
    elif action == Action.PICK_UP:
        picked = sorted(
            board.slots[i].identity for i in board.cells[position] if items & (1 << i)
        )
        for identity in picked:
            i = bisect_left(board.identities, identity)
            items ^= 1 << i
            slot = board.slots[i]
            if slot.kind in ("boots", "ghost", "shield"):
                effects.append((i, slot.time, slot.usage))
                effect_ids = effect_ids | {slot.identity}
            else:
                inventory |= 1 << i
                inventory_ids = inventory_ids | {slot.identity}
                if slot.kind == "coin":
                    reward += 5
        values = {i: (i, t, u) for i, t, u in effects}
        effects = [values[i] for i in sorted(values)]
        phase = any(board.slots[i].kind == "ghost" for i, _, _ in effects)
        contact()
    elif action == Action.USE_KEY:
        for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            target = _neighbor(board, position, dx, dy)
            if target < 0:
                continue
            for i in board.cells[target]:
                if not doors & (1 << i):
                    continue
                key = next(
                    (
                        j
                        for identity in inventory_ids
                        for j in (bisect_left(board.identities, identity),)
                        if board.slots[j].kind == "key"
                        and board.slots[j].key == board.slots[i].key
                    ),
                    None,
                )
                if key is not None:
                    inventory ^= 1 << key
                    inventory_ids = inventory_ids - {board.slots[key].identity}
                    doors ^= 1 << i
        contact()
    else:
        contact()
    values = {}
    for i, t, u in effects:
        if t == 1 and i in initial_effects:
            effect_ids = effect_ids - {board.slots[i].identity}
        else:
            values[i] = (i, t - 1 if t != -1 and i in initial_effects else t, u)
    next_effects = tuple(
        values[bisect_left(board.identities, identity)] for identity in effect_ids
    )
    inventory_order = tuple(
        bisect_left(board.identities, identity) for identity in inventory_ids
    )
    turn += 1
    if turn >= board.turn_limit and not win:
        lose = True
    reward -= 0 if win or health <= 0 else 3
    return FastState(
        position,
        items,
        doors,
        tuple(boxes),
        inventory,
        next_effects,
        health,
        max_health,
        turn,
        win,
        lose,
        inventory_order,
    ), reward


# Frozen P1 reference backend (a51380b9); namespaced, no runtime file import.
"""Bounded Task 1 search using public GridState transitions."""
from collections import deque
from collections.abc import Callable, Hashable
from heapq import heapify, heappop, heappush
from itertools import combinations
from typing import Any, NamedTuple, cast

from grid_adventure.env import ImageObservation
from grid_adventure.step import Action

reference_StateKey = tuple[Hashable, ...]


def reference__freeze(value):
    """Conservative immutable encoding for unknown public component fields."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(reference__freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((reference__freeze(v) for v in value), key=repr))
    if isinstance(value, dict):
        return tuple(sorted((k, reference__freeze(v)) for (k, v) in value.items()))
    if callable(value) and hasattr(value, "__qualname__"):
        return (value.__module__, value.__qualname__)
    return (
        type(value).__module__,
        type(value).__qualname__,
        tuple(sorted((k, reference__freeze(v)) for (k, v) in vars(value).items())),
    )


reference__COMPONENT_FIELDS = {
    "health": ("current_health", "max_health"),
    "key": ("key_id",),
    "locked": ("key_id",),
    "rewardable": ("amount",),
    "damage": ("amount",),
    "cost": ("amount",),
    "speed": ("multiplier",),
    "time_limit": ("amount",),
    "usage_limit": ("amount",),
    "inventory": ("item_ids",),
    "status": ("effect_ids",),
}
reference__MARKERS = frozenset(
    (
        "agent",
        "blocking",
        "collectible",
        "collidable",
        "exit",
        "pushable",
        "requirable",
        "immunity",
        "phasing",
    )
)


def reference__entity_key(entity):
    result = []
    for name, component in vars(entity).items():
        if component is None or name in ("entity_id", "appearance"):
            continue
        if name in ("inventory_list", "status_list"):
            result.append(
                (
                    name,
                    tuple(
                        sorted(
                            (e.entity_id, reference__entity_key(e)) for e in component
                        )
                    ),
                )
            )
        elif name in reference__COMPONENT_FIELDS:
            fields = reference__COMPONENT_FIELDS[name]
            encoded = tuple(
                reference__freeze(getattr(component, field)) for field in fields
            )
            extras = tuple(
                sorted(
                    (k, reference__freeze(v))
                    for (k, v) in vars(component).items()
                    if k not in fields
                )
            )
            result.append((name, (encoded, extras)))
        elif name in reference__MARKERS:
            result.append(
                (
                    name,
                    tuple(
                        sorted(
                            (k, reference__freeze(v))
                            for (k, v) in vars(component).items()
                        )
                    ),
                )
            )
        else:
            result.append((name, reference__freeze(component)))
    return tuple(sorted(result))


def reference_state_key(state: GridState) -> reference_StateKey:
    """Full dynamic state, including overlaps; built-in V1 configurations only."""
    board = tuple(
        (x, y, tuple(sorted(reference__entity_key(e) for e in cell)))
        for (x, column) in enumerate(state.grid)
        for (y, cell) in enumerate(column)
        if cell
    )
    return (
        state.width,
        state.height,
        reference__freeze(state.movement),
        reference__freeze(state.objective),
        state.seed,
        state.turn,
        state.turn_limit,
        state.step_cost,
        state.score,
        state.win,
        state.lose,
        board,
    )


reference_BfsFrontier = deque[tuple[int, GridState]]
reference_HeapFrontier = list[tuple[float, float, int, GridState]]
reference_STATE_CAP = 4096
reference_RECORD_CAP = 32768
reference_PROFILE = False
reference_PRODUCTION_MODE = "guided"
reference_SEARCH_WEIGHT = 3.0
reference_RESOURCE_PROGRESS_WEIGHT = 50.0
reference_INEFFECTIVE_PENALTY = 1000.0


class reference_SearchResult(NamedTuple):
    actions: tuple[Action, ...]
    expected_keys: tuple[reference_StateKey, ...]
    winning: bool
    final_score: float | None
    stop_reason: str
    expanded: int
    simulations: int = 0
    peak_states: int = 0
    peak_records: int = 0
    key_seconds: float | None = None
    simulation_seconds: float | None = None
    heap_seconds: float | None = None


def reference_relaxed_route_length(state: GridState) -> int:
    """Ordering only: Manhattan closest-target distance plus a target MST."""
    agent = None
    gems = []
    exits = []
    for x, column in enumerate(state.grid):
        for y, cell in enumerate(column):
            for e in cell:
                if getattr(e, "agent", None) is not None:
                    agent = (x, y)
                if getattr(e, "requirable", None) is not None:
                    gems.append((x, y))
                if getattr(e, "exit", None) is not None:
                    exits.append((x, y))
    if agent is None or not exits:
        return 0

    def distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    routes = []
    for exit_position in exits:
        targets = gems + [exit_position]
        nearest = min(distance(agent, p) for p in targets)
        reached = [targets.pop()]
        tree = 0
        while targets:
            (cost, index) = min(
                (distance(a, b), i) for a in reached for (i, b) in enumerate(targets)
            )
            tree += cost
            reached.append(targets.pop(index))
        routes.append(nearest + tree)
    return min(routes)


class reference_RouteOrdering:
    """Relaxed geometry/resource estimates for ordering, never transition rules."""

    def __init__(self, state: GridState, *, deadline=float("inf"), clock=perf_counter):
        self.deadline = deadline
        self.clock = clock
        self.hazard_weight = 16.0
        self.box_weight = 16
        (self.width, self.height) = (state.width, state.height)
        self.walls: set[int] = set()
        self.hazards: dict[int, float] = {}
        self.initial_keys = self.initial_doors = 0
        self.resource_progress = 0
        for x, column in enumerate(state.grid):
            for y, cell in enumerate(column):
                point = x + y * self.width
                for entity in cell:
                    locked = getattr(entity, "locked", None)
                    self.initial_doors += locked is not None
                    self.initial_keys += getattr(entity, "key", None) is not None
                    if (
                        getattr(entity, "blocking", None) is not None
                        and locked is None
                        and (getattr(entity, "pushable", None) is None)
                    ):
                        self.walls.add(point)
                    damage = getattr(entity, "damage", None)
                    if damage is not None:
                        self.hazards[point] = damage.amount
        self.maps: dict[tuple, list[float]] = {}
        self.tours: dict[tuple, float] = {}
        self.options: dict[tuple, float] = {}
        self.reachability: dict[tuple, dict[int, set[int]]] = {}

    def neighbors(self, point):
        (x, y) = (point % self.width, point // self.width)
        return (
            nx + ny * self.width
            for (nx, ny) in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            if 0 <= nx < self.width and 0 <= ny < self.height
        )

    def distance(self, origin, target, phase=False, protection=0, obstacles=()):
        if self.clock() >= self.deadline:
            return 0.0
        if not phase and origin in self.walls and (origin != target):
            penalties = dict(obstacles)
            return min(
                (
                    1
                    + self.hazard_weight * self.hazards.get(point, 0) / (1 + protection)
                    + penalties.get(point, 0)
                    + self.distance(point, target, phase, protection, obstacles)
                    for point in self.neighbors(origin)
                    if point not in self.walls
                ),
                default=10000.0,
            )
        key = (target, phase, protection, obstacles)
        if key not in self.maps:
            if len(self.maps) >= 256:
                self.maps.clear()
            distances = [10000.0] * (self.width * self.height)
            distances[target] = 0.0
            queue = [(0.0, target)]
            penalties = dict(obstacles)
            while queue:
                if self.clock() >= self.deadline:
                    return 0.0
                (distance, point) = heappop(queue)
                if distance != distances[point]:
                    continue
                cost = (
                    1
                    + (2 if phase and point in self.walls else 0)
                    + self.hazard_weight * self.hazards.get(point, 0) / (1 + protection)
                    + (penalties.get(point, 0) if not phase else 0)
                )
                for neighbor in self.neighbors(point):
                    if neighbor in self.walls and (not phase):
                        continue
                    candidate = distance + cost
                    if candidate < distances[neighbor]:
                        distances[neighbor] = candidate
                        heappush(queue, (candidate, neighbor))
            self.maps[key] = distances
        return self.maps[key][origin]

    def tour(self, origin, targets, exit_point, phase, protection, obstacles):
        if self.clock() >= self.deadline:
            return 0.0
        if len(targets) > 6:
            points = list(targets) + [exit_point]
            estimate = min(
                self.distance(origin, p, phase, protection, obstacles) for p in points
            ) + len(targets)
            connected = [points.pop()]
            while points:
                if self.clock() >= self.deadline:
                    return 0.0
                (cost, index) = min(
                    (
                        min(
                            self.distance(a, b, phase, protection, obstacles),
                            self.distance(b, a, phase, protection, obstacles),
                        ),
                        i,
                    )
                    for a in connected
                    for (i, b) in enumerate(points)
                )
                estimate += cost
                connected.append(points.pop(index))
            return estimate
        key = (origin, targets, exit_point, phase, protection, obstacles)
        if key not in self.tours:
            if len(self.tours) >= 8192:
                self.tours.clear()
            self.tours[key] = min(
                (
                    self.distance(origin, target, phase, protection, obstacles)
                    + 1
                    + self.tour(
                        target,
                        targets[:i] + targets[i + 1 :],
                        exit_point,
                        phase,
                        protection,
                        obstacles,
                    )
                    for (i, target) in enumerate(targets)
                ),
                default=self.distance(origin, exit_point, phase, protection, obstacles),
            )
        return self.tours[key]

    def future(
        self, origin, targets, exits, phase, protection, speed, obstacles, powers
    ):
        if self.clock() >= self.deadline:
            return 0.0
        key = (
            origin,
            tuple(targets),
            tuple(exits),
            phase,
            protection,
            speed,
            tuple(obstacles),
            tuple(powers),
        )
        if key not in self.options:
            if len(self.options) >= 8192:
                self.options.clear()
            self.options[key] = self._future(
                origin, targets, exits, phase, protection, speed, obstacles, powers
            )
        return self.options[key]

    def ghost_route(
        self, origin, point, steps, targets, exits, protection, speed, obstacles
    ):
        best = 10000.0
        for exit_point in exits:
            region = self.reachable(exit_point, ())
            before = tuple(target for target in targets if target not in region)
            after = tuple(target for target in targets if target in region)
            approach = (
                self.tour(origin, before, point, False, protection, obstacles) / speed
            )
            for landing in region:
                if self.clock() >= self.deadline:
                    return best
                distance = abs(point % self.width - landing % self.width) + abs(
                    point // self.width - landing // self.width
                )
                if distance <= (steps + 1) * speed:
                    best = min(
                        best,
                        approach
                        + 1
                        + distance / speed
                        + self.tour(
                            landing, after, exit_point, False, protection, obstacles
                        )
                        / speed,
                    )
        return best

    def _future(
        self, origin, targets, exits, phase, protection, speed, obstacles, powers
    ):
        estimate = min(
            self.tour(origin, targets, e, phase, protection, obstacles) / speed
            for e in exits
        )
        ghost_count = sum(bool(ghost) for (_, ghost, _, _) in powers)
        for i, (point, ghost, shield, boost) in enumerate(powers):
            if self.clock() >= self.deadline:
                break
            if ghost and ghost_count == 1 and (not phase):
                estimate = min(
                    estimate,
                    self.ghost_route(
                        origin,
                        point,
                        ghost,
                        targets,
                        exits,
                        protection,
                        speed,
                        obstacles,
                    ),
                )
                continue
            (p, defense, velocity) = (
                phase or bool(ghost),
                protection + shield,
                max(speed, boost),
            )
            remaining = min(
                self.tour(point, targets, e, p, defense, obstacles) / velocity
                for e in exits
            )
            if remaining >= 10000:
                remaining = min(
                    remaining,
                    self.future(
                        point,
                        targets,
                        exits,
                        p,
                        defense,
                        velocity,
                        obstacles,
                        powers[:i] + powers[i + 1 :],
                    ),
                )
            estimate = min(
                estimate,
                self.distance(origin, point, phase, protection, obstacles) / speed
                + 1
                + remaining,
            )
        return estimate

    def reachable(self, origin, doors):
        if self.clock() >= self.deadline:
            return {origin}
        key = tuple(doors)
        if key not in self.reachability:
            if len(self.reachability) >= 128:
                self.reachability.clear()
            unseen = set(range(self.width * self.height)) - self.walls - set(doors)
            groups = {}
            while unseen:
                seed = next(iter(unseen))
                reached = {seed}
                unseen.remove(seed)
                queue = deque([seed])
                while queue:
                    for neighbor in self.neighbors(queue.popleft()):
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            reached.add(neighbor)
                            queue.append(neighbor)
                for point in reached:
                    groups[point] = reached
            self.reachability[key] = groups
        groups = self.reachability[key]
        if origin in self.walls:
            return {origin}.union(
                *(groups.get(point, set()) for point in self.neighbors(origin))
            )
        return groups.get(origin, {origin})

    def estimate(self, state: GridState, origin_override=None) -> float:
        if self.clock() >= self.deadline:
            return 0.0
        origin = None
        targets = []
        exits = []
        powers = []
        keys = []
        doors = []
        boxes = []
        phase = False
        phase_steps = 0
        protection = 0
        speed = 1
        inventory = 0
        health = 1
        for x, column in enumerate(state.grid):
            for y, cell in enumerate(column):
                point = x + y * self.width
                for entity in cell:
                    if getattr(entity, "agent", None) is not None:
                        origin = point
                        health = getattr(
                            getattr(entity, "health", None), "current_health", 1
                        )
                        inventory = sum(
                            getattr(item, "key", None) is not None
                            for item in getattr(entity, "inventory_list", ())
                        )
                        for effect in getattr(entity, "status_list", ()):
                            phase |= getattr(effect, "phasing", None) is not None
                            if getattr(effect, "phasing", None) is not None:
                                phase_steps = max(
                                    phase_steps,
                                    getattr(
                                        getattr(effect, "time_limit", None),
                                        "amount",
                                        150,
                                    ),
                                )
                            if getattr(effect, "immunity", None) is not None:
                                protection += getattr(
                                    getattr(effect, "usage_limit", None), "amount", 5
                                )
                            boost = getattr(effect, "speed", None)
                            if boost is not None:
                                speed = max(speed, boost.multiplier)
                    if getattr(entity, "requirable", None) is not None:
                        targets.append(point)
                    if getattr(entity, "exit", None) is not None:
                        exits.append(point)
                    if getattr(entity, "key", None) is not None:
                        keys.append(point)
                    if getattr(entity, "locked", None) is not None:
                        doors.append(point)
                    if getattr(entity, "pushable", None) is not None:
                        boxes.append(point)
                    if getattr(entity, "collectible", None) is not None:
                        ghost = (
                            getattr(getattr(entity, "time_limit", None), "amount", 5)
                            if getattr(entity, "phasing", None) is not None
                            else 0
                        )
                        shield = (
                            getattr(getattr(entity, "usage_limit", None), "amount", 5)
                            if getattr(entity, "immunity", None) is not None
                            else 0
                        )
                        boost = getattr(entity, "speed", None)
                        if ghost or shield or boost is not None:
                            powers.append(
                                (
                                    point,
                                    ghost,
                                    shield,
                                    boost.multiplier if boost is not None else 1,
                                )
                            )
        self.resource_progress = (
            self.initial_keys - len(keys) + self.initial_doors - len(doors)
        )
        if origin_override is not None:
            origin = origin_override
        if origin is None or not exits:
            return 0.0
        if not phase and (not any(ghost for (_, ghost, _, _) in powers)):
            bottlenecks = sum(
                any(
                    target not in self.reachable(origin, (door,))
                    and self.distance(origin, target) < 10000
                    for target in targets + exits
                )
                for door in doors
            )
            if keys and len(keys) <= bottlenecks - inventory:
                targets.extend(point for point in keys if point not in targets)
        remaining = tuple(targets)
        closed = tuple((point, 2) for point in doors) + tuple(
            (point, self.box_weight) for point in boxes
        )
        estimate = self.future(
            origin, remaining, exits, phase, protection, speed, closed, powers
        )
        if not phase and inventory == 0 and doors:
            reachable = self.reachable(origin, tuple(doors))
            if any(
                target not in reachable and self.distance(origin, target) < 10000
                for target in targets + exits
            ):
                via_keys = [
                    self.distance(origin, key, phase, protection, closed)
                    + 1
                    + self.future(
                        key, remaining, exits, phase, protection, speed, closed, powers
                    )
                    - 1
                    for key in keys
                    if key in reachable
                ]
                if via_keys:
                    estimate = min(via_keys)
        shields = [(point, shield) for (point, _, shield, _) in powers if shield]
        if shields and (not phase) and (not any(ghost for (_, ghost, _, _) in powers)):
            critical_hits = 0
            for hazard, damage in self.hazards.items():
                if self.clock() >= self.deadline:
                    break
                if (
                    hazard != origin
                    and damage > 0
                    and any(
                        target not in self.reachable(origin, (hazard,))
                        and self.distance(origin, target) < 10000
                        for target in targets + exits
                    )
                ):
                    critical_hits += 1
            damage = max(self.hazards.values(), default=1)
            required = (
                critical_hits - protection - max(0, int((health - 1) / max(1, damage)))
            )
            if required > 0:
                for count in range(1, len(shields) + 1):
                    routes = []
                    for subset in combinations(shields, count):
                        charges = sum(amount for (_, amount) in subset)
                        if charges >= required:
                            stops = remaining + tuple(
                                point for (point, _) in subset if point not in remaining
                            )
                            routes.append(
                                min(
                                    self.tour(
                                        origin,
                                        stops,
                                        e,
                                        False,
                                        protection + charges,
                                        closed,
                                    )
                                    / speed
                                    for e in exits
                                )
                            )
                    if routes:
                        estimate = min(routes)
                        break
        if phase:
            landings = []
            for point in range(self.width * self.height):
                if self.clock() >= self.deadline:
                    break
                if point in self.walls:
                    continue
                distance = abs(point % self.width - origin % self.width) + abs(
                    point // self.width - origin // self.width
                )
                if distance <= (phase_steps + 1) * speed:
                    landings.append(
                        distance / speed
                        + self.future(
                            point,
                            remaining,
                            exits,
                            False,
                            protection,
                            speed,
                            closed,
                            powers,
                        )
                    )
            estimate = min(landings, default=10000.0)
        return max(0.0, estimate - inventory)

    def actions(self, state):
        origin = None
        pickup = False
        key = False
        for x, column in enumerate(state.grid):
            for y, cell in enumerate(column):
                if any(getattr(e, "agent", None) is not None for e in cell):
                    origin = x + y * self.width
                    pickup = any(
                        getattr(e, "collectible", None) is not None for e in cell
                    )
        if origin is None:
            return tuple(Action)
        for point in self.neighbors(origin):
            key |= any(
                getattr(e, "locked", None) is not None
                for e in state.grid[point % self.width][point // self.width]
            )
        directions = {
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
        }
        ranked = []
        for i, action in enumerate(Action):
            value = 10001.0
            if action in directions:
                (dx, dy) = directions[action]
                x = origin % self.width + dx
                y = origin // self.width + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    value = self.estimate(state, x + y * self.width)
            elif action == Action.PICK_UP and pickup:
                value = -2.0
            elif action == Action.USE_KEY and key:
                value = -1.0
            ranked.append((value, i, action))
        return tuple(action for (_, _, action) in sorted(ranked))


def reference_search(
    start: GridState,
    *,
    deadline: float,
    mode: str = "bfs",
    clock: Callable[[], float] = perf_counter,
) -> reference_SearchResult:
    """Search within an absolute deadline and retained-state/record caps."""
    if mode not in ("bfs", "best_first", "guided"):
        raise ValueError(f"Unsupported mode: {mode}")
    expanded = simulations = 0
    peak_states = 1
    key_seconds = simulation_seconds = heap_seconds = 0.0
    incumbent = None
    refining = False
    nodes = []

    def key_of(s):
        nonlocal key_seconds
        if not reference_PROFILE:
            return reference_state_key(s)
        t = perf_counter()
        key = reference_state_key(s)
        key_seconds += perf_counter() - t
        return key

    def finish(reason):
        (actions, keys) = ([], [])
        if incumbent is not None:
            index = incumbent
            horizon = max(0, (start.turn_limit or 150) - start.turn)
            for _ in range(horizon + 1):
                (parent, action, key, _, _) = nodes[index]
                keys.append(key)
                if parent == -1:
                    break
                actions.append(action)
                index = parent
            else:
                raise RuntimeError("Parent chain exceeds turn horizon")
            actions.reverse()
            keys.reverse()
        return reference_SearchResult(
            tuple(actions),
            tuple(keys),
            incumbent is not None,
            nodes[incumbent][4] if incumbent is not None else None,
            reason,
            expanded,
            simulations,
            peak_states,
            len(nodes),
            key_seconds if reference_PROFILE else None,
            simulation_seconds if reference_PROFILE else None,
            heap_seconds if reference_PROFILE else None,
        )

    if clock() >= deadline:
        return finish("deadline")
    first_key = key_of(start)
    nodes.append((-1, None, first_key, 0, start.score))
    if start.win:
        incumbent = 0
        return finish("win")
    if start.lose:
        return finish("exhausted")
    ordering = (
        reference_RouteOrdering(start, deadline=deadline, clock=clock)
        if mode == "guided"
        else None
    )

    def priority(state, depth):
        if ordering is not None and (not refining):
            estimate = ordering.estimate(state)
            return (
                depth
                + reference_SEARCH_WEIGHT * estimate
                - reference_RESOURCE_PROGRESS_WEIGHT * ordering.resource_progress
            )
        if ordering is not None:
            return -state.score / max(1, state.step_cost) + ordering.estimate(state)
        return depth + reference_relaxed_route_length(state)

    frontier = (
        deque([(0, start)])
        if mode == "bfs"
        else [(priority(start, 0), -start.score, 0, start)]
    )
    visited = {first_key}
    pending: dict[int, tuple[tuple[Action, ...], int]] = {}
    ineffective: set[int] = set()
    horizon = max(0, (start.turn_limit or 150) - start.turn)
    while frontier:
        if clock() >= deadline:
            return finish("deadline")
        if mode == "bfs":
            (index, current) = cast(reference_BfsFrontier, frontier).popleft()
        else:
            t = perf_counter() if reference_PROFILE else 0.0
            (current_priority, _, index, current) = heappop(
                cast(reference_HeapFrontier, frontier)
            )
            if reference_PROFILE:
                heap_seconds += perf_counter() - t
        depth = nodes[index][3]
        if depth >= horizon:
            del current
            continue
        if ordering is not None:
            (options, cursor) = pending.pop(index, ((), 0))
            if not options:
                options = tuple(Action) if refining else ordering.actions(current)
                expanded += 1
            actions = (options[cursor],)
            if cursor + 1 < len(options):
                pending[index] = (options, cursor + 1)
                heappush(
                    cast(reference_HeapFrontier, frontier),
                    (current_priority, -current.score, index, current),
                )
        else:
            expanded += 1
            actions = tuple(Action)
        for action in actions:
            if clock() >= deadline:
                return finish("deadline")
            if len(nodes) >= reference_RECORD_CAP:
                return finish("record_cap")
            retained = len(frontier) + (1 if current is start else 2)
            if retained + 1 > reference_STATE_CAP:
                return finish("state_cap")
            peak_states = max(peak_states, retained + 1)
            t = perf_counter() if reference_PROFILE else 0.0
            successor = grid_step(current, action)
            if reference_PROFILE:
                simulation_seconds += perf_counter() - t
            simulations += 1
            expired = clock() >= deadline
            if not successor.lose:
                key = key_of(successor)
                if key not in visited:
                    visited.add(key)
                    child = len(nodes)
                    nodes.append((index, action, key, depth + 1, successor.score))
                    previous = nodes[index][2]
                    if (
                        ordering is not None
                        and key[11] == previous[11]
                        and (key[:5] == previous[:5])
                        and (key[6:8] == previous[6:8])
                        and (key[9:11] == previous[9:11])
                    ):
                        ineffective.add(child)
                    if successor.win:
                        if incumbent is None or successor.score > nodes[incumbent][4]:
                            incumbent = child
                        if mode == "bfs":
                            return finish("deadline" if expired else "win")
                        if ordering is not None and (not refining):
                            if expired:
                                return finish("deadline")
                            refining = True
                            ordering.hazard_weight = 0.0
                            ordering.box_weight = 8
                            ordering.maps.clear()
                            ordering.tours.clear()
                            ordering.options.clear()
                            heap = cast(reference_HeapFrontier, frontier)
                            for position, (_, _, node, saved) in enumerate(heap):
                                if clock() >= deadline:
                                    return finish("deadline")
                                heap[position] = (
                                    priority(saved, nodes[node][3])
                                    + (
                                        reference_INEFFECTIVE_PENALTY
                                        if node in ineffective
                                        else 0
                                    ),
                                    -saved.score,
                                    node,
                                    saved,
                                )
                            if heap:
                                del saved
                            heapify(heap)
                    elif mode == "bfs":
                        cast(reference_BfsFrontier, frontier).append((child, successor))
                    else:
                        t = perf_counter() if reference_PROFILE else 0.0
                        heappush(
                            cast(reference_HeapFrontier, frontier),
                            (
                                priority(successor, depth + 1)
                                + (
                                    reference_INEFFECTIVE_PENALTY
                                    if child in ineffective
                                    else 0
                                ),
                                -successor.score,
                                child,
                                successor,
                            ),
                        )
                        if reference_PROFILE:
                            heap_seconds += perf_counter() - t
            del successor
            if expired or clock() >= deadline:
                return finish("deadline")
        del current
    return finish("exhausted")


reference_INITIAL_SEARCH_SECONDS = 3.0
reference_SEARCH_CUTOFF_SECONDS = 3.5


class reference_Agent:
    def __init__(self):
        self._constructed = perf_counter()
        self._initial_deadline = self._constructed + reference_INITIAL_SEARCH_SECONDS
        self._cutoff = self._constructed + reference_SEARCH_CUTOFF_SECONDS
        self._tried = False
        self._actions = ()
        self._keys = ()
        self._next = 0
        self._stats: dict[str, Any] = {
            "search_calls": 0,
            "cache_hits": 0,
            "expanded": 0,
            "simulations": 0,
            "peak_states": 0,
            "peak_records": 0,
        }

    def step(self, state: GridState | ImageObservation) -> Action:
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if not self._actions and self._tried and (perf_counter() >= self._cutoff):
            return Action.WAIT
        observed = reference_state_key(state)
        if self._next < len(self._actions) and observed == self._keys[self._next]:
            action = self._actions[self._next]
            self._next += 1
            self._stats["cache_hits"] += 1
            return action
        self._actions = self._keys = ()
        self._next = 0
        deadline = self._cutoff if self._tried else self._initial_deadline
        self._tried = True
        if perf_counter() >= deadline:
            return Action.WAIT
        result = reference_search(
            state, deadline=deadline, mode=reference_PRODUCTION_MODE
        )
        self._stats["search_calls"] += 1
        for name in ("expanded", "simulations"):
            self._stats[name] += getattr(result, name)
        for name in ("peak_states", "peak_records"):
            self._stats[name] = max(self._stats[name], getattr(result, name))
        self._stats["stop_reason"] = result.stop_reason
        for name in ("key_seconds", "simulation_seconds", "heap_seconds"):
            value = getattr(result, name)
            if value is not None:
                self._stats[name] = self._stats.get(name, 0.0) + value
        if result.winning and result.actions:
            self._actions = result.actions
            self._keys = result.expected_keys
            self._next = 1
            return self._actions[0]
        return Action.WAIT

    def info(self) -> dict[str, str]:
        return {key: str(value) for (key, value) in self._stats.items()}


class CompactOrdering(reference_RouteOrdering):
    def __init__(self, board, *, deadline=float("inf"), clock=perf_counter):
        self.board = board
        self.deadline, self.clock = deadline, clock
        self.hazard_weight, self.box_weight = 16.0, 16
        self.width, self.height = board.width, board.height
        self.walls = set(board.walls)
        self.hazards = {s.point: 2 for s in board.slots if s.kind == "lava"}
        self.initial_keys = sum(s.kind == "key" and s.point >= 0 for s in board.slots)
        self.initial_doors = board.doors.bit_count()
        self.resource_progress = 0
        self.maps, self.tours, self.options, self.reachability = {}, {}, {}, {}

    def estimate(self, state, origin_override=None):
        if self.clock() >= self.deadline:
            return 0.0
        board = self.board
        origin = state.position
        health = state.health
        targets = []
        exits = sorted(board.exits)
        powers = []
        keys = []
        doors = []
        boxes = list(state.boxes)
        phase = False
        phase_steps = protection = 0
        speed = 1
        inventory = sum(
            bool(state.inventory & (1 << i)) and s.kind == "key"
            for i, s in enumerate(board.slots)
        )
        for i, t, u in state.effects:
            kind = board.slots[i].kind
            if kind == "ghost":
                phase = True
                phase_steps = max(phase_steps, t)
            elif kind == "shield":
                protection += u
            elif kind == "boots":
                speed = 2
        # P1 scanned columns, then rows, not entity IDs.
        for x in range(board.width):
            for y in range(board.height):
                point = x + y * board.width
                for i in board.cells[point]:
                    slot = board.slots[i]
                    if state.items & (1 << i):
                        if slot.kind == "gem":
                            targets.append(point)
                        elif slot.kind == "key":
                            keys.append(point)
                        elif slot.kind in ("ghost", "shield", "boots"):
                            powers.append(
                                (
                                    point,
                                    slot.time if slot.kind == "ghost" else 0,
                                    slot.usage if slot.kind == "shield" else 0,
                                    2 if slot.kind == "boots" else 1,
                                )
                            )
                    if state.doors & (1 << i):
                        doors.append(point)
        self.resource_progress = (
            self.initial_keys - len(keys) + self.initial_doors - len(doors)
        )
        if origin_override is not None:
            origin = origin_override
        if origin is None or not exits:
            return 0.0
        if not phase and not any(ghost for _, ghost, _, _ in powers):
            bottlenecks = sum(
                any(
                    target not in self.reachable(origin, (door,))
                    and self.distance(origin, target) < 10000
                    for target in targets + exits
                )
                for door in doors
            )
            if keys and len(keys) <= bottlenecks - inventory:
                targets.extend(point for point in keys if point not in targets)
        remaining = tuple(targets)
        closed = tuple((point, 2) for point in doors) + tuple(
            (point, self.box_weight) for point in boxes
        )
        estimate = self.future(
            origin, remaining, exits, phase, protection, speed, closed, powers
        )
        if not phase and inventory == 0 and doors:
            reachable = self.reachable(origin, tuple(doors))
            if any(
                target not in reachable and self.distance(origin, target) < 10000
                for target in targets + exits
            ):
                via_keys = [
                    self.distance(origin, key, phase, protection, closed)
                    + 1
                    + self.future(
                        key, remaining, exits, phase, protection, speed, closed, powers
                    )
                    - 1
                    for key in keys
                    if key in reachable
                ]
                if via_keys:
                    estimate = min(via_keys)
        shields = [(point, shield) for point, _, shield, _ in powers if shield]
        if shields and not phase and not any(ghost for _, ghost, _, _ in powers):
            critical_hits = 0
            for hazard, damage in self.hazards.items():
                if self.clock() >= self.deadline:
                    break
                if (
                    hazard != origin
                    and damage > 0
                    and any(
                        target not in self.reachable(origin, (hazard,))
                        and self.distance(origin, target) < 10000
                        for target in targets + exits
                    )
                ):
                    critical_hits += 1
            damage = max(self.hazards.values(), default=1)
            required = (
                critical_hits - protection - max(0, int((health - 1) / max(1, damage)))
            )
            if required > 0:
                for count in range(1, len(shields) + 1):
                    routes = []
                    for subset in combinations(shields, count):
                        charges = sum(amount for _, amount in subset)
                        if charges >= required:
                            stops = remaining + tuple(
                                point for point, _ in subset if point not in remaining
                            )
                            routes.append(
                                min(
                                    self.tour(
                                        origin,
                                        stops,
                                        e,
                                        False,
                                        protection + charges,
                                        closed,
                                    )
                                    / speed
                                    for e in exits
                                )
                            )
                    if routes:
                        estimate = min(routes)
                        break
        if phase:
            landings = []
            for point in range(self.width * self.height):
                if self.clock() >= self.deadline:
                    break
                if point in self.walls:
                    continue
                distance = abs(point % self.width - origin % self.width) + abs(
                    point // self.width - origin // self.width
                )
                if distance <= (phase_steps + 1) * speed:
                    landings.append(
                        distance / speed
                        + self.future(
                            point,
                            remaining,
                            exits,
                            False,
                            protection,
                            speed,
                            closed,
                            powers,
                        )
                    )
            estimate = min(landings, default=10000.0)
        return max(0.0, estimate - inventory)

    def actions(self, state):
        origin = state.position
        board = self.board
        pickup = any(state.items & (1 << i) for i in board.cells[origin])
        key = any(
            state.doors & (1 << i)
            for point in self.neighbors(origin)
            for i in board.cells[point]
        )
        directions = {
            Action.LEFT: (-1, 0),
            Action.RIGHT: (1, 0),
            Action.UP: (0, -1),
            Action.DOWN: (0, 1),
        }
        ranked = []
        for i, action in enumerate(Action):
            value = 10001.0
            if action in directions:
                dx, dy = directions[action]
                x = origin % self.width + dx
                y = origin // self.width + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    value = self.estimate(state, x + y * self.width)
            elif action == Action.PICK_UP and pickup:
                value = -2.0
            elif action == Action.USE_KEY and key:
                value = -1.0
            ranked.append((value, i, action))
        return tuple(action for _, _, action in sorted(ranked))


STATE_CAP = 4096
RECORD_CAP = 32768
PROFILE = False
PRODUCTION_MODE = "p2b"
INITIAL_SEARCH_SECONDS = 3.0
SEARCH_CUTOFF_SECONDS = 3.5


class Verification(NamedTuple):
    status: str
    actual_score: object
    actual_keys: tuple


class FastResult(NamedTuple):
    actions: tuple
    expected_keys: tuple
    winning: bool
    final_score: object
    stop_reason: str
    expansions: int
    transitions: int
    time_to_first_win: object
    peak_states: int
    peak_records: int


def verify_plan(
    board, observation, actions, predicted_keys, *, deadline, clock=perf_counter
):
    if len(predicted_keys) != len(actions) + 1:
        return Verification("invalid", None, ())
    actual = []
    current = observation
    for index in range(len(actions) + 1):
        if clock() >= deadline:
            return Verification("deadline", None, tuple(actual))
        try:
            key = observe_key(board, current)
        except UnsupportedState:
            return Verification("invalid", None, tuple(actual))
        actual.append(key)
        if key != predicted_keys[index]:
            return Verification("invalid", None, tuple(actual))
        if index < len(actions):
            current = grid_step(current, actions[index])
            if clock() >= deadline:
                return Verification("deadline", None, tuple(actual))
    if not current.win or current.lose:
        return Verification("invalid", None, tuple(actual))
    return Verification("verified", current.score, tuple(actual))


def movement_lower_bound(board, state):
    """Obstacle-free Manhattan MST / permanent speed two; pickups cost zero.

    Every winning path connects agent, remaining gems and an exit. Its movement
    length dominates this tree; each action moves at most two cells, including
    phasing and boots. Multiple gems on one cell may collapse in this relaxation.
    """
    gems = {
        s.point
        for i, s in enumerate(board.slots)
        if state.items & board.gems & (1 << i)
    }
    best = None
    for exit_point in board.exits:
        points = gems | {state.position, exit_point}
        connected = {points.pop()}
        total = 0
        while points:
            distance, point = min(
                (
                    abs(a % board.width - b % board.width)
                    + abs(a // board.width - b // board.width),
                    b,
                )
                for a in connected
                for b in points
            )
            total += distance
            connected.add(point)
            points.remove(point)
        bound = (total + 1) // 2
        best = bound if best is None else min(best, bound)
    return 0 if best is None else best


def reward_upper_bound(board, state, score, *, mst=True):
    """Optimistic final reward; valid only for the checked standard schema.

    A winning continuation can collect at most all remaining 5-point coins.
    Of at least L future actions, at most its last is free. No dominance across
    turns, effects or health is used. Death states never enter the frontier.
    """
    if state.win:
        return score
    remaining_coins = (state.items & board.coins).bit_count()
    lower = movement_lower_bound(board, state) if mst else 0
    return score + 5 * remaining_coins - 3 * max(0, lower - 1)


def search(
    board, start, score, *, deadline, mode, verify_candidate, clock=perf_counter
):
    """Frozen P1 deferred policy on primitives; incumbents must pass replay."""
    if mode not in ("p2a", "p2b"):
        raise ValueError("Unsupported mode: " + mode)
    began = clock()
    expansions = transitions = 0
    peak_states = 1
    incumbent = None
    first_win = None
    refining = False
    nodes = []  # parent, action, state, score, depth

    def finish(reason):
        return FastResult(
            incumbent[0] if incumbent else (),
            incumbent[1] if incumbent else (),
            incumbent is not None,
            incumbent[2] if incumbent else None,
            reason,
            expansions,
            transitions,
            first_win,
            peak_states,
            len(nodes),
        )

    def candidate(index):
        nonlocal incumbent, first_win
        actions, keys = [], []
        while index != -1:
            parent, action, state, value, _ = nodes[index]
            keys.append(predicted_key(board, state, value))
            if parent != -1:
                actions.append(action)
            index = parent
        actions = tuple(reversed(actions))
        keys = tuple(reversed(keys))
        verified = verify_candidate(actions, keys)
        if verified.status == "verified" and (
            incumbent is None or verified.actual_score > incumbent[2]
        ):
            incumbent = (actions, verified.actual_keys, verified.actual_score)
            if first_win is None:
                first_win = clock() - began
        return verified.status

    if clock() >= deadline:
        return finish("deadline")
    nodes.append((-1, None, start, score, 0))
    if start.win:
        status = candidate(0)
        return finish("win" if status == "verified" else status)
    if start.lose:
        return finish("exhausted")
    ordering = CompactOrdering(board, deadline=deadline, clock=clock)

    def priority(state, value, depth):
        estimate = ordering.estimate(state)
        if not refining:
            return depth + 3.0 * estimate - 50.0 * ordering.resource_progress
        return -value / 3 + estimate

    frontier = [(priority(start, score, 0), -score, 0, start, score)]
    visited = {(start, score)}
    pending = {}
    ineffective = set()
    horizon = max(0, board.turn_limit - start.turn)
    while frontier:
        if clock() >= deadline:
            return finish("deadline")
        current_priority, _, index, current, value = heappop(frontier)
        depth = nodes[index][4]
        if (
            mode == "p2b"
            and incumbent is not None
            and reward_upper_bound(board, current, value) <= incumbent[2]
        ):
            pending.pop(index, None)
            continue
        if depth >= horizon:
            continue
        options, cursor = pending.pop(index, ((), 0))
        if not options:
            options = tuple(Action) if refining else ordering.actions(current)
            expansions += 1
        action = options[cursor]
        if cursor + 1 < len(options):
            pending[index] = (options, cursor + 1)
            heappush(frontier, (current_priority, -value, index, current, value))
        if clock() >= deadline:
            return finish("deadline")
        if len(nodes) >= RECORD_CAP:
            return finish("record_cap")
        retained = len(frontier) + (1 if current is start else 2)
        if retained + 1 > STATE_CAP:
            return finish("state_cap")
        peak_states = max(peak_states, retained + 1)
        successor, delta = fast_step(board, current, action)
        next_score = value + delta
        transitions += 1
        expired = clock() >= deadline
        key = predicted_key(board, successor, next_score)
        if (
            not successor.lose
            and key not in visited
            and not (
                mode == "p2b"
                and incumbent is not None
                and reward_upper_bound(board, successor, next_score) <= incumbent[2]
            )
        ):
            visited.add(key)
            child = len(nodes)
            nodes.append((index, action, successor, next_score, depth + 1))
            if successor._replace(turn=current.turn) == current:
                ineffective.add(child)
            if successor.win:
                if incumbent is None or next_score > incumbent[2]:
                    status = candidate(child)
                    if status in ("invalid", "deadline"):
                        return finish(status)
                if not refining:
                    if expired:
                        return finish("deadline")
                    refining = True
                    ordering.hazard_weight, ordering.box_weight = 0.0, 8
                    ordering.maps.clear()
                    ordering.tours.clear()
                    ordering.options.clear()
                    for position, (_, _, node, saved, saved_score) in enumerate(
                        frontier
                    ):
                        if clock() >= deadline:
                            return finish("deadline")
                        frontier[position] = (
                            priority(saved, saved_score, nodes[node][4])
                            + (1000.0 if node in ineffective else 0),
                            -saved_score,
                            node,
                            saved,
                            saved_score,
                        )
                    heapify(frontier)
            else:
                heappush(
                    frontier,
                    (
                        priority(successor, next_score, depth + 1)
                        + (1000.0 if child in ineffective else 0),
                        -next_score,
                        child,
                        successor,
                        next_score,
                    ),
                )
        if expired or clock() >= deadline:
            return finish("deadline")
    return finish("exhausted")


class ReferenceAgent(reference_Agent):
    def step_until(self, observation, deadline):
        self._cutoff = min(self._cutoff, deadline)
        self._initial_deadline = min(self._initial_deadline, deadline)
        return self.step(observation)


class Agent:
    def __init__(self):
        self._constructed = perf_counter()
        self._initial_deadline = self._constructed + INITIAL_SEARCH_SECONDS
        self._cutoff = self._constructed + SEARCH_CUTOFF_SECONDS
        self._tried = False
        self._board = None
        self._actions = self._keys = ()
        self._next = 0
        self._reference = None
        self._stats: dict[str, Any] = {
            "search_calls": 0,
            "cache_hits": 0,
            "transitions": 0,
            "expansions": 0,
            "peak_states": 0,
            "peak_records": 0,
        }

    def _fallback(self, state):
        self._stats["backend"] = "reference"
        if self._reference is None:
            self._reference = ReferenceAgent()
        deadline = self._cutoff if self._tried else self._initial_deadline
        self._tried = True
        return self._reference.step_until(state, deadline)

    def step(self, state):
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if self._reference is not None:
            return self._fallback(state)
        try:
            if (
                self._next < len(self._actions)
                and observe_key(self._board, state) == self._keys[self._next]
            ):
                action = self._actions[self._next]
                self._next += 1
                self._stats["cache_hits"] += 1
                return action
            self._actions = self._keys = ()
            self._next = 0
            deadline = self._cutoff if self._tried else self._initial_deadline
            self._tried = True
            if perf_counter() >= deadline:
                return Action.WAIT
            board, start, score = encode(state)
            self._board = board
            result = search(
                board,
                start,
                score,
                deadline=deadline,
                mode=PRODUCTION_MODE,
                verify_candidate=lambda actions, keys: verify_plan(
                    board, state, actions, keys, deadline=self._cutoff
                ),
            )
            self._stats["search_calls"] += 1
            self._stats["stop_reason"] = result.stop_reason
            self._stats["time_to_first_win"] = result.time_to_first_win
            for name in ("transitions", "expansions"):
                self._stats[name] += getattr(result, name)
            for name in ("peak_states", "peak_records"):
                self._stats[name] = max(self._stats[name], getattr(result, name))
            if result.winning and result.actions:
                self._actions, self._keys = result.actions, result.expected_keys
                self._next = 1
                return result.actions[0]
            if result.stop_reason == "invalid":
                return self._fallback(state)
        except UnsupportedState:
            self._stats["stop_reason"] = "unsupported"
            return self._fallback(state)
        return Action.WAIT

    def info(self):
        return {key: str(value) for key, value in self._stats.items()}
