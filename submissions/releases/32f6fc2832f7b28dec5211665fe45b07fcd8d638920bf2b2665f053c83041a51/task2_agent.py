# V4 standalone; planning is conditional on observed visual evidence.
"""P2 candidate from frozen P2a e478c1cc; isolated from P2 and V3 releases."""

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
    keys: int


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
    key_mask = sum(1 << i for i, s in enumerate(slots) if s.kind == "key")
    # With <=1 key, a lock has zero or one matching carried identity. Coins
    # and gems cannot affect this choice. Competing keys retain the old guard.
    if (
        key_mask.bit_count() > 1
        and sum(s.kind in ("key", "coin", "gem") for s in slots) > 4
    ):
        raise UnsupportedState("Large keyed inventory iteration history")
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
        key_mask,
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
    if board.keys.bit_count() <= 1 or not board.keys & (inventory | items):
        inventory_order.sort()
    return FastState(
        position,
        items,
        doors,
        tuple(boxes),
        inventory,
        tuple(sorted(effects)),
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
    effect_values = {i: (i, t, u) for i, t, u in effects}
    effects = [effect_values[i] for i in sorted(effect_values)]
    boxes = list(boxes)
    initial_effects = {i for i, _, _ in effects}
    phase = any(board.slots[i].kind == "ghost" for i, _, _ in effects)
    speed = 2 if any(board.slots[i].kind == "boots" for i, _, _ in effects) else 1
    reward = 0

    def contact():
        nonlocal health, win, lose
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
                if target in boxes:
                    destination = _neighbor(board, target, dx, dy)
                    if (
                        destination >= 0
                        and destination not in boxes
                        and not blocked(destination)
                        and not any(
                            board.slots[i].kind == "lava"
                            for i in board.cells[destination]
                        )
                    ):
                        boxes = [
                            destination if point == target else point for point in boxes
                        ]
                        position = target
                    elif phase:
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
        if not (t == 1 and i in initial_effects):
            values[i] = (i, t - 1 if t != -1 and i in initial_effects else t, u)
    next_effects = tuple(values[i] for i in sorted(values))
    inventory_order = tuple(
        bisect_left(board.identities, identity) for identity in inventory_ids
    )
    if board.keys.bit_count() <= 1 or not board.keys & (inventory | items):
        inventory_order = tuple(sorted(inventory_order))
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
    dispatch_cutoff: float | None = None,
) -> reference_SearchResult:
    """Search within an absolute deadline and retained-state/record caps."""
    if mode not in ("bfs", "best_first", "guided"):
        raise ValueError(f"Unsupported mode: {mode}")
    began = clock()
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
                        if dispatch_cutoff is not None:
                            # Search cost includes public transitions, state keys
                            # and ordering. Leave room for independent replay,
                            # then the caller's observation/dispatch overhead.
                            per_step = max(0.0, clock() - began) / max(1, simulations)
                            reserve = 0.1 + 4.0 * nodes[incumbent][3] * per_step
                            deadline = min(deadline, dispatch_cutoff - reserve)
                            if ordering is not None:
                                ordering.deadline = min(ordering.deadline, deadline)
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


def reference_verify_plan(
    start, actions, expected_keys, *, deadline, clock=perf_counter
):
    """Replay a complete reference incumbent before installing its cache."""
    if len(expected_keys) != len(actions) + 1:
        return Verification("invalid", None, ())
    current = start
    actual = []
    for index in range(len(actions) + 1):
        if clock() >= deadline:
            return Verification("deadline", None, ())
        key = reference_state_key(current)
        actual.append(key)
        if key != expected_keys[index]:
            return Verification("invalid", None, ())
        if index < len(actions):
            current = grid_step(current, actions[index])
    if clock() >= deadline:
        return Verification("deadline", None, ())
    if not current.win or current.lose:
        return Verification("invalid", None, ())
    return Verification("verified", current.score, tuple(actual))


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
            if perf_counter() >= self._cutoff:
                return Action.WAIT
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
            state,
            deadline=deadline,
            mode=reference_PRODUCTION_MODE,
            dispatch_cutoff=self._cutoff,
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
        if perf_counter() >= self._cutoff:
            return Action.WAIT
        if result.winning and result.actions:
            before = perf_counter()
            verified = reference_verify_plan(
                state,
                result.actions,
                result.expected_keys,
                deadline=self._cutoff,
                clock=perf_counter,
            )
            self._stats["verification_status"] = verified.status
            self._stats["verification_seconds"] = perf_counter() - before
            if verified.status != "verified" or perf_counter() >= self._cutoff:
                return Action.WAIT
            self._actions = result.actions
            self._keys = verified.actual_keys
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
        # An uncollected ghost is not proof that every lock can be bypassed
        # before expiry. Keep key bottlenecks in primitive action ordering.
        if not phase:
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
PRODUCTION_MODE = "p2a"
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


def primitive_search(
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


# P3 resource-event rescue. Heuristic caps are truncations, not dominance.
EVENT_FRONTIER_CAP = 1024
EVENT_RECORD_CAP = 8192
EVENT_DEPTH = 24
EVENT_ALTERNATIVES = 4


def event_query(board, start, score, goal, *, deadline, clock=perf_counter):
    """Reach a resource event using all primitive actions and exact endpoints."""
    if clock() >= deadline:
        return (), 0, 0, 0
    kind, identity = goal
    ordering = CompactOrdering(board, deadline=deadline, clock=clock)
    target = board.slots[identity].point if kind in ("item", "door") else identity
    records = [(-1, None, start, score, 0)]
    seen = {(start, score)}
    endpoints = []
    transitions = 0
    peak_frontier = 1

    def complete(state):
        if state.win:
            return True
        if kind == "item":
            return not state.items & (1 << identity)
        if kind == "door":
            return not state.doors & (1 << identity)
        if kind == "box":
            return state.boxes != start.boxes
        return False

    def distance(state):
        phase = any(board.slots[i].kind == "ghost" for i, _, _ in state.effects)
        protection = sum(
            u for i, _, u in state.effects if board.slots[i].kind == "shield"
        )
        closed = tuple(
            (s.point, 2) for i, s in enumerate(board.slots) if state.doors & (1 << i)
        )
        closed += tuple((p, 8) for p in state.boxes)
        return ordering.distance(state.position, target, phase, protection, closed)

    frontier = [(distance(start), 0)]
    while frontier and clock() < deadline:
        _, index = heappop(frontier)
        _, _, state, value, depth = records[index]
        if complete(state):
            actions = []
            cursor = index
            while records[cursor][0] != -1:
                parent, action, _, _, _ = records[cursor]
                actions.append(action)
                cursor = parent
            endpoints.append((state, value, tuple(reversed(actions))))
            if len(endpoints) >= EVENT_ALTERNATIVES:
                break
            continue
        if state.lose or depth >= min(EVENT_DEPTH, board.turn_limit - start.turn):
            continue
        for action in Action:
            if clock() >= deadline:
                return tuple(endpoints), transitions, len(records), peak_frontier
            successor, delta = fast_step(board, state, action)
            transitions += 1
            key = (successor, value + delta)
            if successor.lose or key in seen:
                continue
            if len(records) >= EVENT_RECORD_CAP or len(frontier) >= EVENT_FRONTIER_CAP:
                return tuple(endpoints), transitions, len(records), peak_frontier
            seen.add(key)
            child = len(records)
            records.append((index, action, successor, value + delta, depth + 1))
            rank = depth + 1 + 3 * distance(successor)
            if complete(successor):
                rank = -100000 + depth
            heappush(frontier, (rank, child))
            peak_frontier = max(peak_frontier, len(frontier))
    return tuple(endpoints), transitions, len(records), peak_frontier


def event_search(
    board, start, score, *, deadline, mode, verify_candidate, clock=perf_counter
):
    """Bounded resource-event frontier; only public-verified primitive wins exit."""
    began = clock()
    nodes = [(start, score, ())]
    frontier = [(0, 0)]
    seen = {(start, score)}
    transitions = expansions = 0
    peak_states = peak_records = 1
    ordering = CompactOrdering(board, deadline=deadline, clock=clock)

    def finish(reason):
        return FastResult(
            (),
            (),
            False,
            None,
            reason,
            expansions,
            transitions,
            None,
            peak_states,
            peak_records,
        )

    while frontier and clock() < deadline:
        _, index = heappop(frontier)
        state, value, prefix = nodes[index]
        goals = [
            ("item", i) for i, slot in enumerate(board.slots) if state.items & (1 << i)
        ]
        goals += [
            ("door", i) for i, slot in enumerate(board.slots) if state.doors & (1 << i)
        ]
        goals += [("box", point) for point in state.boxes]
        goals += [("exit", point) for point in board.exits]

        # Near resources first; goal completion itself is never coordinate arrival.
        def goal_rank(goal, state=state):
            kind, identity = goal
            target = (
                board.slots[identity].point if kind in ("item", "door") else identity
            )
            distance = abs(target % board.width - state.position % board.width) + abs(
                target // board.width - state.position // board.width
            )
            return distance + (
                8 if kind == "item" and board.slots[identity].kind == "coin" else 0
            )

        goals.sort(key=goal_rank)
        expansions += 1
        for goal in goals:
            if clock() >= deadline:
                return finish("deadline")
            endpoints, used, records, local_peak = event_query(
                board, state, value, goal, deadline=deadline, clock=clock
            )
            transitions += used
            peak_records = max(peak_records, len(nodes) + records)
            peak_states = max(peak_states, len(frontier) + local_peak)
            for endpoint, next_score, suffix in endpoints:
                key = (endpoint, next_score)
                if key in seen:
                    continue
                seen.add(key)
                actions = prefix + suffix
                if endpoint.win:
                    # Rebuild all predicted prefixes from the original observation.
                    current, running_score = start, score
                    keys = [predicted_key(board, current, running_score)]
                    for action in actions:
                        if clock() >= deadline:
                            return finish("deadline")
                        current, delta = fast_step(board, current, action)
                        running_score += delta
                        keys.append(predicted_key(board, current, running_score))
                    verified = verify_candidate(actions, tuple(keys))
                    if verified.status == "verified":
                        return FastResult(
                            actions,
                            verified.actual_keys,
                            True,
                            verified.actual_score,
                            "event_win",
                            expansions,
                            transitions,
                            clock() - began,
                            peak_states,
                            peak_records,
                        )
                    return finish(verified.status)
                if (
                    len(nodes) >= EVENT_RECORD_CAP
                    or len(frontier) >= EVENT_FRONTIER_CAP
                ):
                    return finish("event_cap")
                child = len(nodes)
                nodes.append((endpoint, next_score, actions))
                estimate = ordering.estimate(endpoint)
                progress = (
                    board.gems & ~endpoint.items
                ).bit_count() + ordering.resource_progress
                rank = len(actions) + 3 * estimate - 50 * progress
                heappush(frontier, (rank, child))
    return finish("deadline" if clock() >= deadline else "exhausted")


def search(
    board, start, score, *, deadline, mode, verify_candidate, clock=perf_counter
):
    """Preserve the qualified primitive result; rescue only a capped nonwinner."""
    began = clock()
    result = primitive_search(
        board,
        start,
        score,
        deadline=deadline,
        mode=mode,
        verify_candidate=verify_candidate,
        clock=clock,
    )
    if (
        result.winning
        or result.stop_reason not in ("state_cap", "record_cap", "exhausted")
        or clock() >= deadline
    ):
        return result
    rescued = event_search(
        board,
        start,
        score,
        deadline=deadline,
        mode=mode,
        verify_candidate=verify_candidate,
        clock=clock,
    )
    return rescued._replace(
        transitions=result.transitions + rescued.transitions,
        expansions=result.expansions + rescued.expansions,
        peak_states=max(result.peak_states, rescued.peak_states),
        peak_records=max(result.peak_records, rescued.peak_records),
        time_to_first_win=clock() - began if rescued.winning else None,
    )


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
            self._reference._constructed = self._constructed
            self._reference._initial_deadline = self._initial_deadline
            self._reference._tried = perf_counter() >= self._initial_deadline
        self._reference._cutoff = min(self._reference._cutoff, self._cutoff)
        self._reference._initial_deadline = min(
            self._reference._initial_deadline,
            self._initial_deadline,
            self._reference._cutoff,
        )
        self._tried = True
        return self._reference.step(state)

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
                if perf_counter() >= self._cutoff:
                    return Action.WAIT
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
                mode=getattr(self, "_mode", PRODUCTION_MODE),
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
            if perf_counter() >= self._cutoff:
                return Action.WAIT
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


class PlannerAdapter(Agent):
    """Structural absolute budget: started_at, initial_deadline, cutoff.

    Caller reserves execution/perception outside cutoff. To preserve V3's
    allocation, pass reference_initial_deadline=min(budget.initial_deadline,
    budget.started_at + 2.0). This module has no vision dependency.
    """

    def __init__(self, budget, mode="p2a", reference_initial_deadline=None):
        if mode not in ("p2a", "p2b"):
            raise ValueError("Unknown planner mode")
        super().__init__()
        self.budget, self._mode = budget, mode
        self._constructed = budget.started_at
        self._cutoff = budget.cutoff
        self._initial_deadline = min(budget.initial_deadline, self._cutoff)
        self._reference_initial_deadline = min(
            self._initial_deadline,
            self._initial_deadline
            if reference_initial_deadline is None
            else reference_initial_deadline,
        )

    def _fallback(self, state):
        self._stats["backend"] = "reference"
        if self._reference is None:
            self._reference = ReferenceAgent()
            self._reference._constructed = self._constructed
            self._reference._initial_deadline = self._reference_initial_deadline
            # A late constructor must enter recovery immediately if the
            # original initial interval is already spent; never renew it.
            self._reference._tried = perf_counter() >= min(
                self._initial_deadline, self._reference_initial_deadline
            )
        backend = self._reference
        backend._initial_deadline = min(
            backend._initial_deadline,
            self._initial_deadline,
            self._reference_initial_deadline,
        )
        backend._cutoff = min(self._cutoff, self.budget.cutoff)
        self._tried = True
        return backend.step(state)

    def step(self, state):
        self._cutoff = min(self._cutoff, self.budget.cutoff)
        self._initial_deadline = min(
            self._initial_deadline, self.budget.initial_deadline, self._cutoff
        )
        if perf_counter() >= self._cutoff:
            return Action.WAIT
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if self._reference is not None:
            backend = self._reference
            if (
                backend._next < len(backend._actions)
                and reference_state_key(state) != backend._keys[backend._next]
            ):
                self._rebind_reference_suffix(state)
        elif self._next < len(self._actions):
            try:
                agrees = observe_key(self._board, state) == self._keys[self._next]
            except UnsupportedState:
                agrees = False
            if not agrees:
                self._rebind_suffix(state)
        if perf_counter() >= self._cutoff:
            return Action.WAIT
        return super().step(state)

    def _rebind_reference_suffix(self, state):
        backend = self._reference
        if backend is None:
            return
        suffix = backend._actions[backend._next :]
        current = state
        keys = [reference_state_key(current)]
        for action in suffix:
            if perf_counter() >= self._cutoff:
                return
            current = grid_step(current, action)
            keys.append(reference_state_key(current))
        if perf_counter() >= self._cutoff or not current.win or current.lose:
            return
        backend._actions, backend._keys, backend._next = suffix, tuple(keys), 0
        self._stats["reference_suffix_rebindings"] = (
            self._stats.get("reference_suffix_rebindings", 0) + 1
        )

    def _rebind_suffix(self, state):
        suffix = self._actions[self._next :]
        try:
            board, current, score = encode(state)
            keys = [predicted_key(board, current, score)]
            for action in suffix:
                if perf_counter() >= self._cutoff:
                    return
                current, delta = fast_step(board, current, action)
                score += delta
                keys.append(predicted_key(board, current, score))
            verified = verify_plan(
                board, state, suffix, tuple(keys), deadline=self._cutoff
            )
            if verified.status != "verified" or perf_counter() >= self._cutoff:
                return
            self._board = board
            self._actions, self._keys, self._next = suffix, verified.actual_keys, 0
            self._stats["suffix_rebindings"] = (
                self._stats.get("suffix_rebindings", 0) + 1
            )
        except UnsupportedState:
            return


class EpisodeBudget(NamedTuple):
    started_at: float
    initial_deadline: float
    cutoff: float


class Task3Agent(PlannerAdapter):
    """Explicit structured Task 3 entrypoint; budget never follows geometry."""

    def __init__(self, budget=None, mode="p2a"):
        if budget is None:
            began = perf_counter()
            budget = EpisodeBudget(began, began + 11.0, began + 12.0)
        super().__init__(budget, mode=mode)

StructuredAgent = Agent
BackendPlannerAdapter = PlannerAdapter
P3_REFERENCE_STATE_KEY = reference_state_key

# ---- v4_types ----
"""V3 runtime records. This module deliberately has no vision-library import."""

from typing import NamedTuple

from grid_adventure.grid import GridState
from grid_adventure.step import Action


class EpisodeBudget(NamedTuple):
    started_at: float
    initial_deadline: float
    cutoff: float


class CellScores(NamedTuple):
    probabilities: object
    width: int
    height: int


class BoardHypothesis(NamedTuple):
    state: GridState
    confidence: float
    unresolved: tuple


class ObservationCheck(NamedTuple):
    consistent: bool
    changed_cells: tuple
    mismatches: tuple
    observed_scores: object = None


class Emission(NamedTuple):
    turn: int | None
    action: Action
    reason: str
    prediction_valid: bool


class PerceptionEvidence(NamedTuple):
    scores: CellScores
    disagreements: tuple
    policy: str


class PerceptionPolicy(NamedTuple):
    temperature: float
    min_confidence: float
    min_margin: float
    view: str

# ---- v4_bridge ----
"""Public image/metadata contract, without access to an evaluation environment.

Grid-only IDs occupy negative local slots. IDs explicitly exposed in public
resource metadata are retained after pickup: changing their integer values can
change Python-set key consumption order. No unobserved engine IDs are read.
"""

from copy import deepcopy
from dataclasses import replace
from typing import Any

from grid_adventure.entities import (
    AgentEntity,
    BoxEntity,
    CoinEntity,
    ExitEntity,
    GemEntity,
    KeyEntity,
    LavaEntity,
    LockedDoorEntity,
    PhasingPowerUpEntity,
    ShieldPowerUpEntity,
    SpeedPowerUpEntity,
    UnlockedDoorEntity,
    WallEntity,
)
from grid_adventure.grid import GridState
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES



CLASSES = (
    "boots",
    "box",
    "coin",
    "exit",
    "floor",
    "gem",
    "ghost",
    "human",
    "key",
    "lava",
    "locked",
    "opened",
    "shield",
    "wall",
)
ENTITY_TYPES: dict[str, Any] = dict(
    zip(
        CLASSES,
        (
            SpeedPowerUpEntity,
            BoxEntity,
            CoinEntity,
            ExitEntity,
            None,
            GemEntity,
            PhasingPowerUpEntity,
            AgentEntity,
            KeyEntity,
            LavaEntity,
            LockedDoorEntity,
            UnlockedDoorEntity,
            ShieldPowerUpEntity,
            WallEntity,
        ),
    )
)
EFFECT_TYPES = {"SPEED": "boots", "PHASING": "ghost", "IMMUNITY": "shield"}


def split_cells(observation):
    image = observation["image"]
    config = observation["info"]["config"]
    width, height = config["width"], config["height"]
    ih, iw = image.shape[:2]
    if width <= 0 or height <= 0 or iw < width or ih < height:
        raise ValueError("Invalid image/grid geometry")
    return [
        [
            image[
                y * ih // height : (y + 1) * ih // height,
                x * iw // width : (x + 1) * iw // width,
                :,
            ]
            for x in range(width)
        ]
        for y in range(height)
    ]


def decode_metadata(info):
    config, status, agent = info["config"], info["status"], info["agent"]
    objective = config["objective"]
    if objective == "collect_exit":
        objective = "collect_gems_and_exit"
    if objective not in OBJECTIVES or config["movement"] not in MOVEMENTS:
        raise ValueError("Unsupported image configuration")
    return {
        **{key: config[key] for key in ("width", "height", "movement")},
        "objective": objective,
        "seed": None if config["seed"] == -1 else config["seed"],
        "turn_limit": None if config["turn_limit"] == -1 else config["turn_limit"],
        **{key: status[key] for key in ("turn", "score", "phase")},
        "health": deepcopy(agent["health"]),
        "inventory": deepcopy(agent["inventory"]),
        "effects": deepcopy(agent["effects"]),
    }


def _resource(entry, *, effect=False):
    kind = EFFECT_TYPES[entry["type"]] if effect else entry["appearance_name"]
    constructor = ENTITY_TYPES[kind]
    if constructor is None or kind not in (
        "boots",
        "ghost",
        "shield",
        "coin",
        "gem",
        "key",
    ):
        raise ValueError("Unsupported resource type")
    entity = constructor(entity_id=entry["id"])
    if not isinstance(entry["id"], int) or entry["id"] < 0:
        raise ValueError("Unsupported public resource ID")
    if kind == "key":
        entity.key = replace(entity.key, key_id=entry["key_id"])
    else:
        entity.collectible = None
    if effect:
        field = {"TIME": "time_limit", "USAGE": "usage_limit"}[entry["limit_type"]]
        setattr(
            entity, field, replace(getattr(entity, field), amount=entry["limit_amount"])
        )
        if kind == "boots":
            entity.speed = replace(entity.speed, multiplier=entry["multiplier"])
    return entity


def reconstruct(labels, info, previous=None):
    """Build a conditional hypothesis; tuple labels assert observed full layers.

    A single actor/box label on a later frame does not establish the covered
    layers. A same-turn predicted previous state supplies corroborated history;
    without it the ambiguity is returned explicitly.
    """
    meta = decode_metadata(info)
    width, height = meta["width"], meta["height"]
    if len(labels) != height or any(len(row) != width for row in labels):
        raise ValueError("Labels do not match metadata geometry")
    if previous is not None and (
        (previous.width, previous.height) != (width, height)
        or previous.turn != meta["turn"]
    ):
        raise ValueError("History must be the same-turn prediction")
    state = GridState(
        width=width,
        height=height,
        movement=MOVEMENTS[meta["movement"]],
        objective=OBJECTIVES[meta["objective"]],
        seed=meta["seed"],
        turn_limit=meta["turn_limit"],
        step_cost=3,
        turn=meta["turn"],
        score=meta["score"],
        win=meta["phase"] == "win",
        lose=meta["phase"] == "lose",
        message=info.get("message") or None,
    )
    unresolved = []
    old_ids = (
        [e.entity_id for col in previous.grid for cell in col for e in cell]
        if previous
        else []
    )
    next_id = min([-1] + old_ids) - 1
    agents: list[AgentEntity] = []
    for y, row in enumerate(labels):
        for x, label in enumerate(row):
            explicit_layers = isinstance(label, (tuple, list))
            kinds = tuple(label) if explicit_layers else (label,)
            old = list(previous.grid[x][y]) if previous else []
            old = deepcopy(old)
            for kind in kinds:
                if kind not in ENTITY_TYPES:
                    unresolved.append((x, y, "unknown_class", kind))
                    continue
                if kind == "floor":
                    continue
                constructor = ENTITY_TYPES[kind]
                match = next((e for e in old if isinstance(e, constructor)), None)
                if match is None and kind == "human" and previous is not None:
                    # There is one actor. A visually established unexpected
                    # position must preserve its local identity during repair.
                    match = next(
                        (
                            deepcopy(e)
                            for col in previous.grid
                            for cell in col
                            for e in cell
                            if getattr(e, "agent", None) is not None
                        ),
                        None,
                    )
                if match is not None:
                    if match in old:
                        old.remove(match)
                    entity = match
                else:
                    entity = constructor(entity_id=next_id)
                    next_id -= 1
                state.add((x, y), entity)
                if kind == "human":
                    agents.append(entity)
            if not explicit_layers and any(k in ("human", "box") for k in kinds):
                if previous is None and meta["turn"] > 0:
                    unresolved.append((x, y, "covered_layers_without_history"))
                elif previous is not None:
                    for entity in old:
                        state.add((x, y), entity)
    if len(agents) != 1:
        unresolved.append(("agent_count", len(agents)))
    for agent in agents:
        agent.health = replace(agent.health, **meta["health"])
        agent.inventory_list = [_resource(entry) for entry in meta["inventory"]]
        agent.status_list = [_resource(entry, effect=True) for entry in meta["effects"]]
    return BoardHypothesis(state, 1.0 if not unresolved else 0.0, tuple(unresolved))

# ---- v4_model ----
_NUMPY_MODEL_B64 = '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'

class _NumpyCNN:
    numpy_inference = True

    def __init__(self, weights):
        self._cache = {}
        self.weights = weights
        self.kernels = [
            weights[f"features.{i}.weight"].transpose(0, 2, 3, 1)
            .reshape(weights[f"features.{i}.weight"].shape[0], -1).T.copy()
            for i in (0, 2, 4, 6)
        ]

    def eval(self):
        return self

    def __call__(self, images):
        import numpy as np

        if images.ndim != 4 or images.shape[1:] != (3, 48, 48):
            raise ValueError("Expected NCHW 48px RGB crops")
        if not len(images):
            return np.empty((0, 14), np.float32)
        # Exact FP32 input bytes only; never reuse inferred labels by position.
        keys = [row.tobytes() for row in images]
        missing = {}
        for i, key in enumerate(keys):
            if key not in self._cache and key not in missing:
                missing[key] = i
        fresh = {}
        if missing:
            logits = self._infer(images[list(missing.values())])
            fresh = {key: row.copy() for key, row in zip(missing, logits)}
        # Assemble before eviction, including boards larger than the cache.
        result = np.stack([fresh[key] if key in fresh else self._cache[key] for key in keys])
        for key, row in fresh.items():
            if len(self._cache) >= 1024:
                del self._cache[next(iter(self._cache))]
            self._cache[key] = row
        return result

    def _infer(self, images):
        import numpy as np

        if images.ndim != 4 or images.shape[1:] != (3, 48, 48):
            raise ValueError("Expected NCHW 48px RGB crops")
        results = []
        # Bound intermediate im2col memory independently of board size.
        for offset in range(0, len(images), 32):
            x = images[offset:offset + 32].transpose(0, 2, 3, 1)
            for i, stride, kernel in zip((0, 2, 4, 6), (1, 2, 2, 2), self.kernels):
                x = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
                windows = np.lib.stride_tricks.sliding_window_view(
                    x, (3, 3), axis=(1, 2)
                )[:, ::stride, ::stride]
                n, h, w, c, _, _ = windows.shape
                rows = windows.transpose(0, 1, 2, 4, 5, 3).reshape(n * h * w, 9 * c)
                x = (rows @ kernel + self.weights[f"features.{i}.bias"]).reshape(n, h, w, -1)
                np.maximum(x, 0, out=x)
            x = x.reshape(len(x), 2, 3, 2, 3, 96).mean(axis=(2, 4))
            x = x.transpose(0, 3, 1, 2).reshape(len(x), 384)
            x = x @ self.weights["head.1.weight"].T + self.weights["head.1.bias"]
            np.maximum(x, 0, out=x)
            results.append(x @ self.weights["head.3.weight"].T + self.weights["head.3.bias"])
        return np.concatenate(results) if results else np.empty((0, 14), np.float32)


def get_model(device="cpu", dtype=None):
    import base64
    import io
    import numpy as np

    if device != "cpu" or dtype not in (None, "float32", np.float32):
        raise ValueError("Only FP32 CPU inference is supported")
    data = base64.b64decode(_NUMPY_MODEL_B64)
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        weights = {key: archive[key] for key in archive.files}
    return _NumpyCNN(weights)

TEMPERATURE = 1.8340080864093418

POLICY = PerceptionPolicy(TEMPERATURE, 0.9, 0.1, "full")

# ---- v4_perception ----
"""Lazy CPU perception: runtime model bytes are supplied by the V3 package."""

from types import SimpleNamespace




# Mutate one shared object so pasted function-scoped notebook cells and module
# execution use the same configuration; rebinding globals splits those scopes.
_perception_runtime = SimpleNamespace(
    loader=None,
    model=None,
    temperature=1.0,
    policy=PerceptionPolicy(1.0, 0.85, 0.20, "full"),
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


def classify(observation, cells=None):
    import numpy as np
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

perception_policy = current_policy

# ---- v4_tracking ----
"""Conditional board tracking with independent image and public metadata checks."""

from collections import Counter
from copy import deepcopy
from itertools import product
from time import perf_counter
from typing import NamedTuple

import numpy as np
from grid_adventure.grid import step as grid_step
from grid_adventure.step import Action















class PredictionHistory(NamedTuple):
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
            any(entity_kind(e) == "exit" for e in cell)
            for col in hypothesis.state.grid
            for cell in col
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


def initial_hypotheses(evidence, info, *, previous=None):
    return _initial_hypotheses_policy(
        evidence, info, perception_policy(), previous=previous
    )


def _initial_hypotheses_policy(evidence, info, policy, *, previous=None):
    scores = evidence.scores
    disagreements = {(x, y): (a, b) for x, y, a, b in evidence.disagreements}
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
                (x, y) in disagreements
                or values[y, x, a] < policy.min_confidence
                or values[y, x, a] - values[y, x, b] < policy.min_margin
            ):
                if (x, y) in disagreements:
                    a, b = (CLASSES.index(k) for k in disagreements[(x, y)])
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


def repeated_ambiguity(check, hypotheses):
    """Identical pixels with the same retained alternatives add no contradiction.

    This preserves a verified planner only; reconciliation and all-hypothesis
    action verification still run with the independent measured probabilities.
    """
    if not hypotheses or not check.mismatches:
        return False
    changed = set(check.changed_cells)
    for mismatch in check.mismatches:
        if mismatch[0] != "uncertain_cell":
            return False
        _, point, _, _, probabilities = mismatch
        if point in changed:
            return False
        options = {CLASSES[int(i)] for i in np.argsort(probabilities)[-2:]}
        if check.observed_scores is not None:
            for x, y, a, b in check.observed_scores.disagreements:
                if (x, y) == point:
                    options = {a, b}
        if any(
            not any(
                len(entry) == 4
                and entry[:2] == point
                and entry[2] == "ambiguous"
                and set(entry[3]) == options
                for entry in h.unresolved
            )
            for h in hypotheses
        ):
            return False
    return True


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
    evidence = classify_evidence(observation, tuple(sorted(selected)))
    scores = evidence.scores
    foreground_confirmed = set()
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
    actor_probabilities = scores.probabilities[position[1], position[0]]
    actor_ranking = np.sort(actor_probabilities)
    uncertain_actor = (
        actor_ranking[-1] < perception_policy().min_confidence
        or actor_ranking[-1] - actor_ranking[-2] < perception_policy().min_margin
        or any((x, y) == position for x, y, _, _ in evidence.disagreements)
    )
    regional_cells = (
        set(selected)
        if full_humans != [position] or uncertain_actor
        else disputed_boxes
    )
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
                foreground_confirmed.add((x, y))
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
                    foreground_confirmed.add((x, y))
        matched = locate_actor_template(
            observation, tuple(sorted(selected)), actor_template
        )
        if matched is not None:
            x, y = matched
            # One-hot independent corroboration, not a calibrated CNN posterior.
            scores.probabilities[y, x] = 0
            scores.probabilities[y, x, human_index] = 1
            foreground_confirmed.add((x, y))
    disagreements = {
        (x, y) for x, y, _, _ in evidence.disagreements
    } - foreground_confirmed
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
            (x, y) in disagreements
            or probabilities[a] < perception_policy().min_confidence
            or probabilities[a] - probabilities[b] < perception_policy().min_margin
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
        not mismatches,
        tuple(sorted(changed)),
        tuple(mismatches),
        evidence._replace(
            disagreements=tuple(
                d
                for d in evidence.disagreements
                if (d[0], d[1]) not in foreground_confirmed
            )
        ),
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
    # Independent image and metadata checks already corroborated this complete
    # same-turn prediction. Rebuilding it changes entity identities and forces
    # needless replay of the remaining plan after otherwise ordinary moves.
    if (
        check.consistent
        and not check.mismatches
        and check.observed_scores is not None
        and len(hypotheses) == 1
        and not hypotheses[0].unresolved
        and hypotheses[0].state.turn == observation["info"]["status"]["turn"]
    ):
        return hypotheses
    results = []
    for index, hypothesis in enumerate(hypotheses):
        if perf_counter() >= budget.cutoff:
            return ()
        labels = state_labels(hypothesis.state)
        alternatives = {}
        inherited = hypothesis.unresolved
        if check.observed_scores is not None:
            observed = check.observed_scores.scores.probabilities
            disagreements = {
                (x, y): (a, b) for x, y, a, b in check.observed_scores.disagreements
            }
            resolved = set()
            for y, x in np.argwhere(np.isfinite(observed).all(axis=2)):
                probabilities = observed[y, x]
                a, b = map(int, np.argsort(probabilities)[-2:][::-1])
                kind = CLASSES[a]
                confident = (
                    (x, y) not in disagreements
                    and probabilities[a] >= perception_policy().min_confidence
                    and probabilities[a] - probabilities[b]
                    >= perception_policy().min_margin
                )
                if kind != visible_kind(hypothesis.state.grid[x][y]):
                    labels[y][x] = kind
                if confident:
                    # A visible actor/box does not resolve an item hidden
                    # underneath it. Confident exposed cells do resolve it.
                    if kind not in ("human", "box"):
                        resolved.add((x, y))
                else:
                    if (x, y) in disagreements:
                        a, b = (CLASSES.index(k) for k in disagreements[(x, y)])
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
            if (x, y) not in alternatives and (
                probabilities[a] < perception_policy().min_confidence
                or probabilities[a] - probabilities[b] < perception_policy().min_margin
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


def common_safe_prefix(
    hypotheses, actions, budget, *, successor_cache=None, transition=None
):
    if not hypotheses or any(fatal_uncertainty(h) for h in hypotheses):
        return ()
    states = [h.state for h in hypotheses]
    prefix = []
    for action in actions:
        if perf_counter() >= budget.cutoff:
            break
        successors = [(transition or grid_step)(state, action) for state in states]
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
        if successor_cache is not None:
            successor_cache.append(tuple(successors))
        states = successors
        if all(s.win for s in states):
            break
    return tuple(prefix)

# ---- v4_search ----
"""V4 search derived from frozen P3, with bounded post-win refinement.

Parent bytes remain untouched. Only V4 image planning uses these functions.
A first fast incumbent is publicly verified before starting its 25ms refinement
window; reference incumbents are still independently replayed before dispatch.
"""

from grid_adventure.grid import _specialize_single as v4_specialize_entity
from grid_adventure.grid import step as v4_public_step
from grid_universe.grid.convert import _entity_object_from_state as v4_entity_from_state
from grid_universe.grid.convert import to_state as v4_to_state
from grid_universe.grid.entity import FIELD_TO_COMPONENT as V4_COMPONENT_FIELDS
from grid_universe.step import step as v4_native_step


































def v4_reference_transition(state, action):
    before = v4_to_state(state)
    # Reference-bearing worlds retain the complete public conversion path.
    if before.portal or before.pathfinding:
        return v4_public_step(state, action)
    after = v4_native_step(before, action, agent_id=next(iter(before.agent)))
    if after.portal or after.pathfinding:
        return v4_public_step(state, action)

    def static_ids(native):
        candidates = set(native.blocking) | set(native.exit)
        for name in V4_COMPONENT_FIELDS:
            if name not in ("blocking", "exit", "appearance"):
                candidates.difference_update(getattr(native, name))
        return candidates

    reusable = static_ids(before) & static_ids(after)
    original = {
        e.entity_id: e for column in state.grid for cell in column for e in cell
    }
    result = GridState(
        **{
            name: getattr(after, name)
            for name in (
                "width",
                "height",
                "movement",
                "objective",
                "seed",
                "step_cost",
                "turn",
                "score",
                "turn_limit",
                "win",
                "lose",
                "message",
            )
        }
    )
    for eid in sorted(after.position):
        position = after.position[eid]
        if not (0 <= position.x < after.width and 0 <= position.y < after.height):
            continue
        if (
            eid in reusable
            and eid in original
            and all(
                getattr(before, name).get(eid) == getattr(after, name).get(eid)
                for name in ("blocking", "exit", "appearance", "position")
            )
        ):
            # Search only reads these entities. Public independent winning
            # replay still creates its own fully specialized states.
            entity = original[eid]
        else:
            entity = v4_specialize_entity(v4_entity_from_state(after, eid))
            for field in ("inventory_list", "status_list"):
                nested = getattr(entity, field, None)
                if nested:
                    setattr(entity, field, [v4_specialize_entity(e) for e in nested])
        result.add((position.x, position.y), entity)
    return result


def v4_primitive_search(
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
    ordering = None
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
        nonlocal incumbent, first_win, deadline
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
                deadline = min(deadline, clock() + (0.15 if board.coins else 0.025))
                if ordering is not None:
                    ordering.deadline = min(ordering.deadline, deadline)
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


def v4_search(
    board, start, score, *, deadline, mode, verify_candidate, clock=perf_counter
):
    """Preserve the qualified primitive result; rescue only a capped nonwinner."""
    began = clock()
    result = v4_primitive_search(
        board,
        start,
        score,
        deadline=deadline,
        mode=mode,
        verify_candidate=verify_candidate,
        clock=clock,
    )
    if (
        result.winning
        or result.stop_reason not in ("state_cap", "record_cap", "exhausted")
        or clock() >= deadline
    ):
        return result
    rescued = event_search(
        board,
        start,
        score,
        deadline=deadline,
        mode=mode,
        verify_candidate=verify_candidate,
        clock=clock,
    )
    return rescued._replace(
        transitions=result.transitions + rescued.transitions,
        expansions=result.expansions + rescued.expansions,
        peak_states=max(result.peak_states, rescued.peak_states),
        peak_records=max(result.peak_records, rescued.peak_records),
        time_to_first_win=clock() - began if rescued.winning else None,
    )


def v4_reference_search(
    start: GridState,
    *,
    deadline: float,
    mode: str = "bfs",
    clock: Callable[[], float] = perf_counter,
    dispatch_cutoff: float | None = None,
) -> reference_SearchResult:
    """Search within an absolute deadline and retained-state/record caps."""
    if mode not in ("bfs", "best_first", "guided"):
        raise ValueError(f"Unsupported mode: {mode}")
    began = clock()
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
            successor = v4_reference_transition(current, action)
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
                        if incumbent is None:
                            deadline = min(deadline, clock() + 0.025)
                            if ordering is not None:
                                ordering.deadline = min(ordering.deadline, deadline)
                        if incumbent is None or successor.score > nodes[incumbent][4]:
                            incumbent = child
                        if dispatch_cutoff is not None:
                            # Search cost includes public transitions, state keys
                            # and ordering. Leave room for independent replay,
                            # then the caller's observation/dispatch overhead.
                            per_step = max(0.0, clock() - began) / max(1, simulations)
                            reserve = 0.1 + 4.0 * nodes[incumbent][3] * per_step
                            deadline = min(deadline, dispatch_cutoff - reserve)
                            if ordering is not None:
                                ordering.deadline = min(ordering.deadline, deadline)
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

# ---- v4_planner ----
"""V4 binding to the user-selected, validated frozen P3 backend.

The standalone packager vendors the exact parent bytes identified below.
This module does not read source files or import any vision library at runtime.
"""

from time import perf_counter




















BACKEND_SOURCE = "agents/v4_backend.py"
BACKEND_SHA256 = "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4"
__all__ = [
    "BACKEND_SHA256",
    "BACKEND_SOURCE",
    "PlannerAdapter",
    "StructuredAgent",
    "reference_state_key",
]


class _V4ReplayCache:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._verified_edges = {}

    def _verify_cached(self, state, actions, expected_keys, key_of, deadline):
        if len(expected_keys) != len(actions) + 1:
            return Verification("invalid", None, ())
        current, actual, edges = state, [], {}
        previous_reference_key = None
        for index in range(len(actions) + 1):
            if perf_counter() >= deadline:
                return Verification("deadline", None, ())
            try:
                key = (
                    previous_reference_key
                    if key_of is reference_state_key
                    and previous_reference_key is not None
                    else key_of(current)
                )
            except UnsupportedState:
                return Verification("invalid", None, ())
            actual.append(key)
            if key != expected_keys[index]:
                return Verification("invalid", None, ())
            if index < len(actions):
                action = actions[index]
                successor = grid_step(current, action)
                if len(actions) <= 256:
                    source_reference_key = (
                        key
                        if key_of is reference_state_key
                        else previous_reference_key
                        if previous_reference_key is not None
                        else reference_state_key(current)
                    )
                    previous_reference_key = reference_state_key(successor)
                    edges[(id(current), action)] = (
                        current,
                        source_reference_key,
                        successor,
                        previous_reference_key,
                    )
                current = successor
        if perf_counter() >= deadline:
            return Verification("deadline", None, ())
        if not current.win or current.lose:
            return Verification("invalid", None, ())
        # Only a fully verified winning replay may populate the cache.
        self._verified_edges = edges
        return Verification("verified", current.score, tuple(actual))

    def predict(self, state, action):
        owner = getattr(self, "_reference", None) or self
        entry = owner._verified_edges.get((id(state), action))
        if (
            entry is not None
            and entry[0] is state
            and reference_state_key(state) == entry[1]
            and reference_state_key(entry[2]) == entry[3]
        ):
            return entry[2]
        return grid_step(state, action)


class V4ReferenceAgent(_V4ReplayCache, ReferenceAgent):
    def step(self, state: GridState | ImageObservation) -> Action:
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if not self._actions and self._tried and (perf_counter() >= self._cutoff):
            return Action.WAIT
        observed = reference_state_key(state)
        if self._next < len(self._actions) and observed == self._keys[self._next]:
            if perf_counter() >= self._cutoff:
                return Action.WAIT
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
        result = v4_reference_search(
            state,
            deadline=deadline,
            mode=reference_PRODUCTION_MODE,
            dispatch_cutoff=self._cutoff,
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
        if perf_counter() >= self._cutoff:
            return Action.WAIT
        if result.winning and result.actions:
            before = perf_counter()
            verified = self._verify_cached(
                state,
                result.actions,
                result.expected_keys,
                reference_state_key,
                deadline=self._cutoff,
            )
            self._stats["verification_status"] = verified.status
            self._stats["verification_seconds"] = perf_counter() - before
            if verified.status != "verified" or perf_counter() >= self._cutoff:
                return Action.WAIT
            self._actions = result.actions
            self._keys = verified.actual_keys
            self._next = 1
            return self._actions[0]
        return Action.WAIT


class PlannerAdapter(_V4ReplayCache, BackendPlannerAdapter):
    _execution_cutoff: float

    def __init__(self, budget: EpisodeBudget, mode: str = "p2b"):
        super().__init__(
            budget,
            mode=mode,
            reference_initial_deadline=min(
                budget.initial_deadline, budget.started_at + 2.0
            ),
        )
        # Cold perception may have spent the initial interval before this
        # adapter exists. Enter recovery now; never create a new deadline.
        self._tried = perf_counter() >= self._initial_deadline

    def step(self, state):
        # After search closes, only dispatch an existing matching verified suffix.
        if perf_counter() >= self._cutoff:
            if perf_counter() >= getattr(self, "_execution_cutoff", self._cutoff):
                return Action.WAIT
            active = self._reference if self._reference is not None else self
            if active._next >= len(active._actions):
                return Action.WAIT
            try:
                key = (
                    reference_state_key(state)
                    if self._reference is not None
                    else observe_key(self._board, state)
                )
            except UnsupportedState:
                return Action.WAIT
            if key != active._keys[active._next]:
                return Action.WAIT
            action = active._actions[active._next]
            active._next += 1
            active._stats["cache_hits"] += 1
            return action
        self._cutoff = min(self._cutoff, self.budget.cutoff)
        self._initial_deadline = min(
            self._initial_deadline, self.budget.initial_deadline, self._cutoff
        )
        if perf_counter() >= self._cutoff:
            return Action.WAIT
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if self._reference is not None:
            backend = self._reference
            if (
                backend._next < len(backend._actions)
                and reference_state_key(state) != backend._keys[backend._next]
            ):
                self._rebind_reference_suffix(state)
        elif self._next < len(self._actions):
            try:
                agrees = observe_key(self._board, state) == self._keys[self._next]
            except UnsupportedState:
                agrees = False
            if not agrees:
                self._rebind_suffix(state)
        if perf_counter() >= self._cutoff:
            return Action.WAIT
        return self._search_step(state)

    def _search_step(self, state):
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if self._reference is not None:
            return self._fallback(state)
        try:
            if (
                self._next < len(self._actions)
                and observe_key(self._board, state) == self._keys[self._next]
            ):
                if perf_counter() >= self._cutoff:
                    return Action.WAIT
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
            result = v4_search(
                board,
                start,
                score,
                deadline=deadline,
                mode=getattr(self, "_mode", PRODUCTION_MODE),
                verify_candidate=lambda actions, keys: self._verify_cached(
                    state,
                    actions,
                    keys,
                    lambda current: observe_key(board, current),
                    self._cutoff,
                ),
            )
            self._stats["search_calls"] += 1
            self._stats["stop_reason"] = result.stop_reason
            self._stats["time_to_first_win"] = result.time_to_first_win
            for name in ("transitions", "expansions"):
                self._stats[name] += getattr(result, name)
            for name in ("peak_states", "peak_records"):
                self._stats[name] = max(self._stats[name], getattr(result, name))
            if perf_counter() >= self._cutoff:
                return Action.WAIT
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

    def _fallback(self, state):
        self._stats["backend"] = "reference"
        if self._reference is None:
            self._reference = V4ReferenceAgent()
            self._reference._constructed = self._constructed
            self._reference._initial_deadline = self._reference_initial_deadline
            # A late constructor must enter recovery immediately if the
            # original initial interval is already spent; never renew it.
            self._reference._tried = perf_counter() >= min(
                self._initial_deadline, self._reference_initial_deadline
            )
        backend = self._reference
        backend._initial_deadline = min(
            backend._initial_deadline,
            self._initial_deadline,
            self._reference_initial_deadline,
        )
        backend._cutoff = min(self._cutoff, self.budget.cutoff)
        self._tried = True
        return backend.step(state)

# ---- v4 ----
"""V4: conditional image tracking over the explicitly selected frozen P3 backend."""

from time import perf_counter
from typing import Any

from grid_adventure.grid import GridState
from grid_adventure.grid import step as grid_step
from grid_adventure.step import Action























class PerceptionUncertainty(ValueError):
    """The observation has no complete bounded reconstruction for visualization."""


class Agent:
    def __init__(self):
        started = perf_counter()
        self._budget = EpisodeBudget(started, started + 3.0, started + 4.65)
        self._planning_budget = EpisodeBudget(started, started + 3.0, started + 3.5)
        self._structured = StructuredAgent()
        self._structured._constructed = started
        self._structured._initial_deadline = self._budget.initial_deadline
        self._structured._cutoff = self._planning_budget.cutoff
        self._planner = None
        self._hypotheses = ()
        self._previous_image = None
        self._background_image = None
        self._actor_template = None
        self._history = None
        self._pending_recovery = False
        self._last_emission = None
        self._emission_observation = None
        self._fallback_hypotheses = ()
        self._stats: dict[str, Any] = {
            "input_mode": "unset",
            "visual_discrepancies": 0,
            "metadata_discrepancies": 0,
            "repairs": 0,
            "recoveries": 0,
            "actions": 0,
            "hypotheses": 0,
            "reason": "not_started",
            "conditional_on_perception": True,
        }
        configure(get_model, TEMPERATURE, policy=POLICY)

    def parse(self, observation):
        try:
            turn = observation["info"]["status"]["turn"]
            history = self._hypotheses + (
                self._history.hypotheses if self._history is not None else ()
            )
            previous = next((h.state for h in history if h.state.turn == turn), None)
            if previous is None:
                hypotheses = initial_hypotheses(
                    classify_evidence(observation), observation["info"]
                )
            else:
                matching = tuple(h for h in self._hypotheses if h.state.turn == turn)
                if not matching:
                    matching = tuple(h for h in history if h.state.turn == turn)
                check = check_observation(
                    observation,
                    previous,
                    self._previous_image,
                    background_image=self._background_image,
                    actor_template=self._actor_template,
                )
                hypotheses = reconcile(
                    observation, matching, check, self._budget, history=self._history
                )
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            raise PerceptionUncertainty(str(error)) from error
        complete = [h for h in hypotheses if not fatal_uncertainty(h)]
        if not complete:
            raise PerceptionUncertainty("No complete bounded visual reconstruction")
        return complete[0].state

    def step(self, observation):
        if isinstance(observation, GridState):
            self._stats["input_mode"] = "gridstate"
            return self._structured.step(observation)
        self._stats["input_mode"] = "image"
        self._emission_observation = observation
        self._fallback_hypotheses = self._hypotheses
        turn = self._observation_turn(observation)
        try:
            return self._step_image(observation)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            return self._emit_action(
                Action.WAIT,
                turn=turn,
                hypotheses=self._fallback_hypotheses,
                reason="unsupported_observation:" + type(error).__name__,
            )
        finally:
            self._emission_observation = None
            self._fallback_hypotheses = ()

    @staticmethod
    def _observation_turn(observation):
        try:
            turn = observation["info"]["status"]["turn"]
            return turn if type(turn) is int and turn >= 0 else None
        except (KeyError, TypeError):
            return None

    def _emit_action(
        self, action, *, turn, hypotheses, reason, verified_successors=None
    ):
        """Commit only the prediction of the action actually returned.

        Failed replay drops continuity and the executable suffix together. A
        conditional WAIT prediction is never counted as independently observed.
        """
        observation = self._emission_observation
        terminal = reason == "terminal"
        if reason == "cutoff" and observation is not None:
            try:
                terminal = observation["info"]["status"]["phase"] in ("win", "lose")
            except (KeyError, TypeError):
                terminal = False
            if terminal:
                reason = "terminal"
        usable = (
            not terminal
            and turn is not None
            and hypotheses
            and perf_counter() < self._budget.cutoff
            and all(
                h.state.turn == turn and not fatal_uncertainty(h) for h in hypotheses
            )
        )
        if usable:
            try:
                if observation is None:
                    raise TypeError("Missing emission frame")
                config = observation["info"]["config"]
                width, height = config["width"], config["height"]
                usable = (
                    type(width) is int
                    and type(height) is int
                    and all(
                        (h.state.width, h.state.height) == (width, height)
                        for i, h in enumerate(hypotheses)
                    )
                )
            except (KeyError, TypeError):
                usable = False
        predictions = ()
        background = previous_image = actor_template = history = None
        if usable:
            try:
                # All fallible work is local until the final assignment block.
                predictions = tuple(
                    BoardHypothesis(
                        (
                            verified_successors[i]
                            if verified_successors is not None
                            else grid_step(h.state, action)
                        ),
                        h.confidence,
                        h.unresolved,
                    )
                    for i, h in enumerate(hypotheses)
                )
                if perf_counter() >= self._budget.cutoff:
                    predictions = ()
                elif observation is not None:
                    previous_image = observation["image"].copy()
                    background = (
                        previous_image.copy()
                        if self._background_image is None
                        else self._background_image.copy()
                    )
                    actor_template = self._actor_template
                    if self._background_image is None:
                        state = hypotheses[0].state
                        floors = [
                            (x, y)
                            for x, col in enumerate(state.grid)
                            for y, cell in enumerate(col)
                            if not cell
                        ]
                        actor_template = make_actor_template(
                            observation, agent_position(state), floors
                        )
                    else:
                        current = split_cells(observation)
                        tiles = split_cells({**observation, "image": background})
                        state = hypotheses[0].state
                        for y in range(state.height):
                            for x in range(state.width):
                                if visible_kind(state.grid[x][y]) not in (
                                    "human",
                                    "box",
                                ):
                                    tiles[y][x][:] = current[y][x]
                    history = PredictionHistory(hypotheses, action)
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                predictions = ()
                action = Action.WAIT
                reason = "emission_failure:" + type(error).__name__
        valid = bool(predictions)
        # A verified WAIT with no cached plan still consumed an initial search
        # attempt. Retain that attempt so the next observed state can use the
        # original recovery interval instead of restarting an expired interval.
        if not valid or reason not in ("conditional_plan", "planner_no_verified_plan"):
            self._planner = None
        if not valid:
            previous_image = background = actor_template = history = None
            self._pending_recovery = False
        self._hypotheses = predictions
        self._history = history
        self._previous_image = previous_image
        self._background_image = background
        self._actor_template = actor_template
        self._last_emission = Emission(turn, action, reason, valid)
        self._stats["reason"] = reason
        if not terminal:
            self._stats["actions"] += 1
        return action

    def _step_image(self, observation):
        if perf_counter() >= self._budget.cutoff:
            return self._emit_action(
                Action.WAIT,
                turn=self._observation_turn(observation),
                hypotheses=(),
                reason="cutoff",
            )
        if observation["info"]["status"]["phase"] in ("win", "lose"):
            return self._emit_action(
                Action.WAIT,
                turn=self._observation_turn(observation),
                hypotheses=(),
                reason="terminal",
            )
        turn = self._observation_turn(observation)
        if turn is None:
            raise ValueError("Invalid observation turn")
        if self._hypotheses and any(h.state.turn != turn for h in self._hypotheses):
            self._hypotheses = ()
            self._fallback_hypotheses = ()
            self._history = None
            self._planner = None
            self._previous_image = None
            self._background_image = None
            self._actor_template = None
        repairing = False
        if not self._hypotheses:
            hypotheses = initial_hypotheses(
                classify_evidence(observation), observation["info"]
            )
        else:
            check = check_observation(
                observation,
                self._hypotheses[0].state,
                self._previous_image,
                background_image=self._background_image,
                actor_template=self._actor_template,
            )
            if not check.consistent and not repeated_ambiguity(check, self._hypotheses):
                self._fallback_hypotheses = ()
                repairing = True
                self._stats["visual_discrepancies"] += sum(
                    m[0] in ("cell", "uncertain_cell", "agent_position")
                    for m in check.mismatches
                )
                self._stats["metadata_discrepancies"] += sum(
                    m[0] not in ("cell", "uncertain_cell", "agent_position")
                    for m in check.mismatches
                )
                self._stats["repairs"] += 1
                self._planner = None
            elif self._pending_recovery:
                self._stats["recoveries"] += 1
                self._pending_recovery = False
            hypotheses = reconcile(
                observation,
                self._hypotheses,
                check,
                self._budget,
                history=self._history,
            )
        self._fallback_hypotheses = hypotheses
        self._stats["hypotheses"] = len(hypotheses)
        if not hypotheses or any(fatal_uncertainty(h) for h in hypotheses):
            return self._emit_action(
                Action.WAIT,
                turn=turn,
                hypotheses=hypotheses,
                reason="unresolved_perception",
            )
        if perf_counter() >= self._budget.cutoff:
            return self._emit_action(
                Action.WAIT,
                turn=self._observation_turn(observation),
                hypotheses=(),
                reason="cutoff",
            )
        if self._planner is None:
            self._planner = PlannerAdapter(self._planning_budget)
            self._planner._execution_cutoff = self._budget.cutoff
            if repairing:
                self._planner._tried = True
        action = self._planner.step(hypotheses[0].state)
        planner_info = self._planner.info()
        self._stats["planner_stop_reason"] = planner_info.get("stop_reason", "cached")
        self._stats["planner_backend"] = (
            "reference" if self._planner._reference is not None else "fast"
        )
        successor_cache = []
        prefix = common_safe_prefix(
            hypotheses,
            (action,),
            self._budget,
            successor_cache=successor_cache,
            transition=self._planner.predict,
        )
        if not prefix:
            return self._emit_action(
                Action.WAIT,
                turn=turn,
                hypotheses=hypotheses,
                reason="hypothesis_disagreement_or_cutoff",
            )
        if repairing:
            self._pending_recovery = True
        active = (
            self._planner._reference
            if self._planner._reference is not None
            else self._planner
        )
        reason = (
            "planner_no_verified_plan"
            if action == Action.WAIT and not active._actions
            else "conditional_plan"
        )
        return self._emit_action(
            action,
            turn=turn,
            hypotheses=hypotheses,
            reason=reason,
            verified_successors=successor_cache[0],
        )

    def info(self):
        return {
            "name": "V4 recovery development",
            "backend_sha256": BACKEND_SHA256,
            **{key: str(value) for key, value in self._stats.items()},
        }
