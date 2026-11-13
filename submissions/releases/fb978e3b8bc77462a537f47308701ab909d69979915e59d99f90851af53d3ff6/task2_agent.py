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
# V4 selected model and policy; generated loader bytes retained.
TEMPERATURE = 1.8340080864093418
def get_model(device: str = "cpu", dtype: str | None = None):
    """
    Return a TorchScript model loaded from an embedded, base64-encoded compressed blob.
    Self-contained: no need for the original Python class.

    Args:
        device: Where to map the model (e.g., "cpu", "cuda", "cuda:0").
        dtype: Optional dtype to convert parameters/buffers to (e.g., "float32", "float16").
    """
    import base64, io, torch
    import zlib as _z; _decomp = _z.decompress
    _blob_b64 = "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"
    _raw = _decomp(base64.b64decode(_blob_b64))
    buf = io.BytesIO(_raw)
    m = torch.jit.load(buf, map_location=device)
    if dtype is not None:
        dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        for p in m.parameters():
            p.data = p.data.to(dt)
        for b in m.buffers():
            b.data = b.data.to(dt)
    m.eval()
    return m


POLICY = PerceptionPolicy(TEMPERATURE, 0.9, 0.1, 'full')

# ---- v4_perception ----
"""Lazy CPU perception: runtime model bytes are supplied by the V3 package."""




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

# ---- v4_planner ----
"""V4 binding to the user-selected, validated frozen P3 backend.

The standalone packager vendors the exact parent bytes identified below.
This module does not read source files or import any vision library at runtime.
"""






BACKEND_SOURCE = "agents/v4_backend.py"
BACKEND_SHA256 = "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4"
__all__ = [
    "BACKEND_SHA256",
    "BACKEND_SOURCE",
    "PlannerAdapter",
    "StructuredAgent",
    "reference_state_key",
]


class PlannerAdapter(BackendPlannerAdapter):
    def __init__(self, budget: EpisodeBudget, mode: str = "p2a"):
        super().__init__(
            budget,
            mode=mode,
            reference_initial_deadline=min(
                budget.initial_deadline, budget.started_at + 2.0
            ),
        )

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
        self._budget = EpisodeBudget(started, started + 3.0, started + 3.5)
        self._structured = StructuredAgent()
        self._structured._constructed = started
        self._structured._initial_deadline = self._budget.initial_deadline
        self._structured._cutoff = self._budget.cutoff
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

    def _emit_action(self, action, *, turn, hypotheses, reason):
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
                        for h in hypotheses
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
                        grid_step(h.state, action), h.confidence, h.unresolved
                    )
                    for h in hypotheses
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
        if not valid or reason != "conditional_plan":
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
            self._planner = PlannerAdapter(self._budget)
            if repairing:
                self._planner._tried = True
        action = self._planner.step(hypotheses[0].state)
        planner_info = self._planner.info()
        self._stats["planner_stop_reason"] = planner_info.get("stop_reason", "cached")
        self._stats["planner_backend"] = (
            "reference" if self._planner._reference is not None else "fast"
        )
        prefix = common_safe_prefix(hypotheses, (action,), self._budget)
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
            action, turn=turn, hypotheses=hypotheses, reason=reason
        )

    def info(self):
        return {
            "name": "V4 recovery development",
            "backend_sha256": BACKEND_SHA256,
            **{key: str(value) for key, value in self._stats.items()},
        }
