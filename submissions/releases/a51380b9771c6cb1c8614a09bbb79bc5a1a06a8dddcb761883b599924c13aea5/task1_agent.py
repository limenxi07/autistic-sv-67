"""Bounded Task 1 search using public GridState transitions."""

from collections import deque
from collections.abc import Callable, Hashable
from heapq import heapify, heappop, heappush
from itertools import combinations
from time import perf_counter
from typing import Any, NamedTuple, cast

from grid_adventure.env import ImageObservation
from grid_adventure.grid import GridState
from grid_adventure.grid import step as grid_step
from grid_adventure.step import Action

StateKey = tuple[Hashable, ...]


def _freeze(value):
    """Conservative immutable encoding for unknown public component fields."""
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if callable(value) and hasattr(value, "__qualname__"):
        return (value.__module__, value.__qualname__)
    return (
        type(value).__module__,
        type(value).__qualname__,
        tuple(sorted((k, _freeze(v)) for k, v in vars(value).items())),
    )


# Carried entities retain reference IDs and counters; keys and locks use key_id.
# Preserve unknown fields; omit appearance, board IDs and messages.
# Missing and None components are equivalent in the public wrappers.
_COMPONENT_FIELDS = {
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


def _entity_key(entity):
    result = []
    for name, component in vars(entity).items():
        if component is None or name in ("entity_id", "appearance"):
            continue
        if name in ("inventory_list", "status_list"):
            result.append(
                (name, tuple(sorted((e.entity_id, _entity_key(e)) for e in component)))
            )
        elif name in _COMPONENT_FIELDS:
            fields = _COMPONENT_FIELDS[name]
            encoded = tuple(_freeze(getattr(component, field)) for field in fields)
            extras = tuple(
                sorted(
                    (k, _freeze(v))
                    for k, v in vars(component).items()
                    if k not in fields
                )
            )
            result.append((name, (encoded, extras)))
        elif name in _MARKERS:
            result.append(
                (
                    name,
                    tuple(sorted((k, _freeze(v)) for k, v in vars(component).items())),
                )
            )
        else:
            result.append((name, _freeze(component)))
    return tuple(sorted(result))


def state_key(state: GridState) -> StateKey:
    """Full dynamic state, including overlaps; built-in V1 configurations only."""
    board = tuple(
        (x, y, tuple(sorted(_entity_key(e) for e in cell)))
        for x, column in enumerate(state.grid)
        for y, cell in enumerate(column)
        if cell
    )
    return (
        state.width,
        state.height,
        _freeze(state.movement),
        _freeze(state.objective),
        state.seed,
        state.turn,
        state.turn_limit,
        state.step_cost,
        state.score,
        state.win,
        state.lose,
        board,
    )


BfsFrontier = deque[tuple[int, GridState]]
HeapFrontier = list[tuple[float, float, int, GridState]]

STATE_CAP = 4096
RECORD_CAP = 32768
PROFILE = False
PRODUCTION_MODE = "guided"
SEARCH_WEIGHT = 3.0
RESOURCE_PROGRESS_WEIGHT = 50.0
INEFFECTIVE_PENALTY = 1000.0


class SearchResult(NamedTuple):
    actions: tuple[Action, ...]
    expected_keys: tuple[StateKey, ...]
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


def relaxed_route_length(state: GridState) -> int:
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
            cost, index = min(
                (distance(a, b), i) for a in reached for i, b in enumerate(targets)
            )
            tree += cost
            reached.append(targets.pop(index))
        routes.append(nearest + tree)
    return min(routes)


class RouteOrdering:
    """Relaxed geometry/resource estimates for ordering, never transition rules."""

    def __init__(self, state: GridState, *, deadline=float("inf"), clock=perf_counter):
        self.deadline = deadline
        self.clock = clock
        self.hazard_weight = 16.0
        self.box_weight = 16
        self.width, self.height = state.width, state.height
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
                        and getattr(entity, "pushable", None) is None
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
        x, y = point % self.width, point // self.width
        return (
            nx + ny * self.width
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            if 0 <= nx < self.width and 0 <= ny < self.height
        )

    def distance(self, origin, target, phase=False, protection=0, obstacles=()):
        if self.clock() >= self.deadline:
            return 0.0
        if not phase and origin in self.walls and origin != target:
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
                distance, point = heappop(queue)
                if distance != distances[point]:
                    continue
                cost = (
                    1
                    + (2 if phase and point in self.walls else 0)
                    + self.hazard_weight * self.hazards.get(point, 0) / (1 + protection)
                    + (penalties.get(point, 0) if not phase else 0)
                )
                for neighbor in self.neighbors(point):
                    if neighbor in self.walls and not phase:
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
            points = list(targets) + ([exit_point])
            estimate = min(
                self.distance(origin, p, phase, protection, obstacles) for p in points
            ) + len(targets)
            connected = [points.pop()]
            while points:
                if self.clock() >= self.deadline:
                    return 0.0
                cost, index = min(
                    (
                        min(
                            self.distance(a, b, phase, protection, obstacles),
                            self.distance(b, a, phase, protection, obstacles),
                        ),
                        i,
                    )
                    for a in connected
                    for i, b in enumerate(points)
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
                    for i, target in enumerate(targets)
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
        ghost_count = sum(bool(ghost) for _, ghost, _, _ in powers)
        for i, (point, ghost, shield, boost) in enumerate(powers):
            if self.clock() >= self.deadline:
                break
            if ghost and ghost_count == 1 and not phase:
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
            p, defense, velocity = (
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


def search(
    start: GridState,
    *,
    deadline: float,
    mode: str = "bfs",
    clock: Callable[[], float] = perf_counter,
) -> SearchResult:
    """Search within an absolute deadline and retained-state/record caps."""
    if mode not in ("bfs", "best_first", "guided"):
        raise ValueError(f"Unsupported mode: {mode}")
    expanded = simulations = 0
    peak_states = 1
    key_seconds = simulation_seconds = heap_seconds = 0.0
    incumbent = None
    refining = False
    # Records: parent index, incoming action, complete key, depth, final score.
    nodes = []

    def key_of(s):
        nonlocal key_seconds
        if not PROFILE:
            return state_key(s)
        t = perf_counter()
        key = state_key(s)
        key_seconds += perf_counter() - t
        return key

    def finish(reason):
        actions, keys = [], []
        if incumbent is not None:
            index = incumbent
            horizon = max(0, (start.turn_limit or 150) - start.turn)
            for _ in range(horizon + 1):
                parent, action, key, _, _ = nodes[index]
                keys.append(key)
                if parent == -1:
                    break
                actions.append(action)
                index = parent
            else:
                raise RuntimeError("Parent chain exceeds turn horizon")
            actions.reverse()
            keys.reverse()
        return SearchResult(
            tuple(actions),
            tuple(keys),
            incumbent is not None,
            nodes[incumbent][4] if incumbent is not None else None,
            reason,
            expanded,
            simulations,
            peak_states,
            len(nodes),
            key_seconds if PROFILE else None,
            simulation_seconds if PROFILE else None,
            heap_seconds if PROFILE else None,
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
        RouteOrdering(start, deadline=deadline, clock=clock)
        if mode == "guided"
        else None
    )

    def priority(state, depth):
        if ordering is not None and not refining:
            estimate = ordering.estimate(state)
            return (
                depth
                + SEARCH_WEIGHT * estimate
                - RESOURCE_PROGRESS_WEIGHT * ordering.resource_progress
            )
        if ordering is not None:
            return -state.score / max(1, state.step_cost) + ordering.estimate(state)
        return depth + relaxed_route_length(state)

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
            index, current = cast(BfsFrontier, frontier).popleft()
        else:
            t = perf_counter() if PROFILE else 0.0
            current_priority, _, index, current = heappop(cast(HeapFrontier, frontier))
            if PROFILE:
                heap_seconds += perf_counter() - t
        depth = nodes[index][3]
        if depth >= horizon:
            del current
            continue
        if ordering is not None:
            options, cursor = pending.pop(index, ((), 0))
            if not options:
                options = tuple(Action) if refining else ordering.actions(current)
                expanded += 1
            actions = (options[cursor],)
            if cursor + 1 < len(options):
                pending[index] = (options, cursor + 1)
                heappush(
                    cast(HeapFrontier, frontier),
                    (current_priority, -current.score, index, current),
                )
        else:
            expanded += 1
            actions = tuple(Action)
        for action in actions:
            if clock() >= deadline:
                return finish("deadline")
            if len(nodes) >= RECORD_CAP:
                return finish("record_cap")
            # Count start, current and the transient successor.
            retained = len(frontier) + (1 if current is start else 2)
            if retained + 1 > STATE_CAP:
                return finish("state_cap")
            peak_states = max(peak_states, retained + 1)
            t = perf_counter() if PROFILE else 0.0
            successor = grid_step(current, action)
            if PROFILE:
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
                        and key[:5] == previous[:5]
                        and key[6:8] == previous[6:8]
                        and key[9:11] == previous[9:11]
                    ):
                        ineffective.add(child)
                    if successor.win:
                        if incumbent is None or successor.score > nodes[incumbent][4]:
                            incumbent = child
                        if mode == "bfs":
                            return finish("deadline" if expired else "win")
                        if ordering is not None and not refining:
                            if expired:
                                return finish("deadline")
                            refining = True
                            ordering.hazard_weight = 0.0
                            ordering.box_weight = 8
                            ordering.maps.clear()
                            ordering.tours.clear()
                            ordering.options.clear()
                            heap = cast(HeapFrontier, frontier)
                            for position, (_, _, node, saved) in enumerate(heap):
                                if clock() >= deadline:
                                    return finish("deadline")
                                heap[position] = (
                                    priority(saved, nodes[node][3])
                                    + (
                                        INEFFECTIVE_PENALTY
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
                        cast(BfsFrontier, frontier).append((child, successor))
                    else:
                        t = perf_counter() if PROFILE else 0.0
                        heappush(
                            cast(HeapFrontier, frontier),
                            (
                                priority(successor, depth + 1)
                                + (INEFFECTIVE_PENALTY if child in ineffective else 0),
                                -successor.score,
                                child,
                                successor,
                            ),
                        )
                        if PROFILE:
                            heap_seconds += perf_counter() - t
            del successor
            if expired or clock() >= deadline:
                return finish("deadline")
        del current
    return finish("exhausted")


INITIAL_SEARCH_SECONDS = 3.0
SEARCH_CUTOFF_SECONDS = 3.5


class Agent:
    def __init__(self):
        self._constructed = perf_counter()
        self._initial_deadline = self._constructed + INITIAL_SEARCH_SECONDS
        self._cutoff = self._constructed + SEARCH_CUTOFF_SECONDS
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
        # Avoid repeated state scans once planning is exhausted.
        if not self._actions and self._tried and perf_counter() >= self._cutoff:
            return Action.WAIT
        observed = state_key(state)
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
        result = search(state, deadline=deadline, mode=PRODUCTION_MODE)
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
        return {key: str(value) for key, value in self._stats.items()}
