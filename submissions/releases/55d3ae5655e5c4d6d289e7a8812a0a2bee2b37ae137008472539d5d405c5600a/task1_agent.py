"""Bounded Task 1 search using public GridState transitions."""

from collections import deque
from collections.abc import Callable, Hashable
from heapq import heappop, heappush
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


# Carried entities retain inventory/effect counters; keys and locks use key_id.
# Preserve unknown fields and raw references; omit appearance, IDs and messages.
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
            result.append((name, tuple(sorted(_entity_key(e) for e in component))))
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
HeapFrontier = list[tuple[int, float, int, GridState]]

STATE_CAP = 4096
RECORD_CAP = 32768
PROFILE = False
PRODUCTION_MODE = "best_first"


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


def search(
    start: GridState,
    *,
    deadline: float,
    mode: str = "bfs",
    clock: Callable[[], float] = perf_counter,
) -> SearchResult:
    """Search within an absolute deadline and retained-state/record caps."""
    if mode not in ("bfs", "best_first"):
        raise ValueError(f"Unsupported mode: {mode}")
    expanded = simulations = 0
    peak_states = 1
    key_seconds = simulation_seconds = heap_seconds = 0.0
    incumbent = None
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
    frontier = (
        deque([(0, start)])
        if mode == "bfs"
        else [(relaxed_route_length(start), -start.score, 0, start)]
    )
    visited = {first_key}
    horizon = max(0, (start.turn_limit or 150) - start.turn)
    while frontier:
        if clock() >= deadline:
            return finish("deadline")
        if mode == "bfs":
            index, current = cast(BfsFrontier, frontier).popleft()
        else:
            t = perf_counter() if PROFILE else 0.0
            _, _, index, current = heappop(cast(HeapFrontier, frontier))
            if PROFILE:
                heap_seconds += perf_counter() - t
        depth = nodes[index][3]
        if depth >= horizon:
            del current
            continue
        expanded += 1
        for action in Action:
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
                    if successor.win:
                        if incumbent is None or successor.score > nodes[incumbent][4]:
                            incumbent = child
                        if mode == "bfs":
                            return finish("deadline" if expired else "win")
                    elif mode == "bfs":
                        cast(BfsFrontier, frontier).append((child, successor))
                    else:
                        t = perf_counter() if PROFILE else 0.0
                        heappush(
                            cast(HeapFrontier, frontier),
                            (
                                depth + 1 + relaxed_route_length(successor),
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
