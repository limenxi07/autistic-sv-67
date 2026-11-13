"""Task 1 planner. All dynamics use the public GridState transition API."""
from __future__ import annotations

from collections.abc import Hashable
from grid_adventure.grid import GridState, step as grid_step
from grid_adventure.env import ImageObservation
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
    if callable(value) and hasattr(value, '__qualname__'):
        return (value.__module__, value.__qualname__)
    return (type(value).__module__, type(value).__qualname__,
            tuple(sorted((k, _freeze(v)) for k, v in vars(value).items())))


# Public runtime schema, observed via constructors and black-box pickup traces:
# markers affect collisions/objectives; numeric fields affect transitions/reward.
# inventory.item_ids / status.effect_ids are empty in normal GridState wrappers;
# preserve them conservatively if populated. The semantic carried entities are
# inventory_list/status_list, including key.key_id, time_limit.amount,
# usage_limit.amount and speed.multiplier. Entity numeric IDs are cosmetic in
# these wrappers; key_id supplies the matching identity for keys and locks.
# Appearance (sprite/priority), state.message and entity_id are omitted. Generic
# Entity objects have many None fields; absent and None components are equivalent.
# Unknown non-None gameplay fields are retained rather than silently discarded.
_COMPONENT_FIELDS = {
    'health': ('current_health', 'max_health'),
    'key': ('key_id',), 'locked': ('key_id',),
    'rewardable': ('amount',), 'damage': ('amount',), 'cost': ('amount',),
    'speed': ('multiplier',), 'time_limit': ('amount',), 'usage_limit': ('amount',),
    'inventory': ('item_ids',), 'status': ('effect_ids',),
}
_MARKERS = frozenset(('agent', 'blocking', 'collectible', 'collidable', 'exit',
                      'pushable', 'requirable', 'immunity', 'phasing'))


def _entity_key(entity):
    result = []
    for name, component in vars(entity).items():
        if component is None or name in ('entity_id', 'appearance'):
            continue
        if name in ('inventory_list', 'status_list'):
            result.append((name, tuple(sorted((_entity_key(e) for e in component)))))
        elif name in _COMPONENT_FIELDS:
            fields = _COMPONENT_FIELDS[name]
            encoded = tuple(_freeze(getattr(component, field)) for field in fields)
            extras = tuple(sorted((k, _freeze(v)) for k, v in vars(component).items()
                                  if k not in fields))
            result.append((name, (encoded, extras)))
        elif name in _MARKERS:
            result.append((name, tuple(sorted((k, _freeze(v)) for k, v in vars(component).items()))))
        else:
            result.append((name, _freeze(component)))
    return tuple(sorted(result))


def state_key(state: GridState) -> StateKey:
    """Exact score/turns and semantic entities, including overlapping contents.

    Movement and objective include their public functions as module/qualname;
    P1 supports the documented built-in configurations, not arbitrary closures.
    Seed is retained conservatively. Tests cover the supported V1 schema only.
    """
    board = tuple((x, y, tuple(sorted(_entity_key(e) for e in cell)))
                  for x, column in enumerate(state.grid)
                  for y, cell in enumerate(column) if cell)
    return (state.width, state.height, _freeze(state.movement), _freeze(state.objective),
            state.seed, state.turn, state.turn_limit, state.step_cost,
            state.score, state.win, state.lose, board)


from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

STATE_CAP = 4096
RECORD_CAP = 32768
PROFILE = False
PRODUCTION_MODE = 'bfs'


@dataclass(frozen=True, slots=True)
class SearchResult:
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


def search(start: GridState, *, deadline: float, mode: str = 'bfs',
           clock: Callable[[], float] = perf_counter) -> SearchResult:
    """Bounded reference-simulator search with compact parent records.

    Retained-state count includes start, the popped current state, and the
    transient successor. Only frontier entries retain other simulator states.
    A winning successor is saved even when its simulation crosses the deadline.
    """
    if mode != 'bfs':
        raise ValueError(f'Unsupported mode: {mode}')
    expanded = simulations = 0
    peak_states = 1
    key_seconds = simulation_seconds = 0.
    incumbent = None
    # Records: parent index, incoming action, complete key, depth, final score.
    nodes = []

    def key_of(s):
        nonlocal key_seconds
        if not PROFILE: return state_key(s)
        t = perf_counter(); key = state_key(s); key_seconds += perf_counter()-t
        return key

    def finish(reason):
        actions, keys = [], []
        if incumbent is not None:
            index = incumbent
            horizon = max(0, (start.turn_limit or 150) - start.turn)
            for _ in range(horizon+1):
                parent, action, key, _, _ = nodes[index]
                keys.append(key)
                if parent == -1: break
                actions.append(action); index = parent
            else:
                raise RuntimeError('Parent chain exceeds turn horizon')
            actions.reverse(); keys.reverse()
        return SearchResult(tuple(actions), tuple(keys), incumbent is not None,
                            nodes[incumbent][4] if incumbent is not None else None,
                            reason, expanded, simulations, peak_states, len(nodes),
                            key_seconds if PROFILE else None,
                            simulation_seconds if PROFILE else None,
                            0. if PROFILE else None)

    if clock() >= deadline: return finish('deadline')
    first_key = key_of(start)
    nodes.append((-1, None, first_key, 0, start.score))
    if start.win:
        incumbent = 0
        return finish('win')
    if start.lose: return finish('exhausted')
    frontier = deque([(0, start)])
    visited = {first_key}
    horizon = max(0, (start.turn_limit or 150) - start.turn)
    while frontier:
        if clock() >= deadline: return finish('deadline')
        index, current = frontier.popleft()
        depth = nodes[index][3]
        if depth >= horizon:
            del current
            continue
        expanded += 1
        for action in Action:
            if clock() >= deadline: return finish('deadline')
            if len(nodes) >= RECORD_CAP: return finish('record_cap')
            retained = len(frontier) + (1 if current is start else 2)
            if retained + 1 > STATE_CAP: return finish('state_cap')
            peak_states = max(peak_states, retained+1)
            t = perf_counter() if PROFILE else 0.
            successor = grid_step(current, action)
            if PROFILE: simulation_seconds += perf_counter()-t
            simulations += 1
            expired = clock() >= deadline
            if not successor.lose:
                key = key_of(successor)
                if key not in visited:
                    visited.add(key)
                    child = len(nodes)
                    nodes.append((index, action, key, depth+1, successor.score))
                    if successor.win:
                        incumbent = child
                        return finish('deadline' if expired else 'win')
                    frontier.append((child, successor))
            del successor
            if expired or clock() >= deadline: return finish('deadline')
        del current
    return finish('exhausted')


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
        self._stats = dict(search_calls=0, cache_hits=0, expanded=0, simulations=0,
                           peak_states=0, peak_records=0)

    def step(self, state: GridState | ImageObservation) -> Action:
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        # Bounded fallback after planning failure: avoid scanning a board for each
        # remaining WAIT once there is neither a plan nor repair time available.
        if not self._actions and self._tried and perf_counter() >= self._cutoff:
            return Action.WAIT
        observed = state_key(state)
        if self._next < len(self._actions) and observed == self._keys[self._next]:
            action = self._actions[self._next]
            self._next += 1
            self._stats['cache_hits'] += 1
            return action
        self._actions = self._keys = ()
        self._next = 0
        deadline = self._cutoff if self._tried else self._initial_deadline
        self._tried = True
        if perf_counter() >= deadline:
            return Action.WAIT
        result = search(state, deadline=deadline, mode=PRODUCTION_MODE)
        self._stats['search_calls'] += 1
        for name in ('expanded', 'simulations'):
            self._stats[name] += getattr(result, name)
        for name in ('peak_states', 'peak_records'):
            self._stats[name] = max(self._stats[name], getattr(result, name))
        self._stats['stop_reason'] = result.stop_reason
        for name in ('key_seconds', 'simulation_seconds', 'heap_seconds'):
            value = getattr(result, name)
            if value is not None: self._stats[name] = self._stats.get(name, 0.) + value
        if result.winning and result.actions:
            self._actions = result.actions
            self._keys = result.expected_keys
            self._next = 1
            return self._actions[0]
        return Action.WAIT

    def info(self) -> dict[str, str]:
        return {key: str(value) for key, value in self._stats.items()}
