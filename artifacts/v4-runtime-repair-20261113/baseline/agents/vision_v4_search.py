"""V4 search with bounded refinement after the first verified win.

Fast routes receive a 25 ms refinement window; reference routes are replayed
before dispatch. The frozen P3 backend remains unchanged.
"""

from grid_adventure.grid import _specialize_single as v4_specialize_entity
from grid_adventure.grid import step as v4_public_step
from grid_universe.grid.convert import _entity_object_from_state as v4_entity_from_state
from grid_universe.grid.convert import to_state as v4_to_state
from grid_universe.grid.entity import FIELD_TO_COMPONENT as V4_COMPONENT_FIELDS
from grid_universe.step import step as v4_native_step

from agents.vision_v4_backend import (
    RECORD_CAP,
    STATE_CAP,
    Action,
    Board,
    Callable,
    CompactOrdering,
    FastResult,
    GridState,
    Slot,
    UnsupportedState,
    _check_grid,
    _components,
    _kind,
    _scan,
    cast,
    deque,
    event_search,
    fast_step,
    heapify,
    heappop,
    heappush,
    perf_counter,
    predicted_key,
    read_state,
    reference__entity_key,
    reference__freeze,
    reference_BfsFrontier,
    reference_HeapFrontier,
    reference_INEFFECTIVE_PENALTY,
    reference_PROFILE,
    reference_RECORD_CAP,
    reference_relaxed_route_length,
    reference_RESOURCE_PROGRESS_WEIGHT,
    reference_RouteOrdering,
    reference_SEARCH_WEIGHT,
    reference_SearchResult,
    reference_STATE_CAP,
    reward_upper_bound,
)


def v4_encode_proposal(observation):
    """Encode a compact search proposal for supported standard mechanics.

    Preserve inventory/effect identities. Replay each proposed route against the
    public engine before use; reject unsupported components.
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
    # Large keyed inventories are proposals only: the compact simulator may
    # choose a different carried key because public set iteration has history.
    # PlannerAdapter must replay every state and reject any mismatch before use.
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


def v4_state_key(state):
    """The complete reference key, with no recursive work for plain scenery."""
    board = []
    for x, column in enumerate(state.grid):
        for y, cell in enumerate(column):
            if not cell:
                continue
            entities = []
            for entity in cell:
                fields = vars(entity)
                simple = False
                if (
                    len(fields) == 3
                    and "entity_id" in fields
                    and "appearance" in fields
                ):
                    for name in ("blocking", "exit"):
                        component = fields.get(name)
                        if component is not None and not vars(component):
                            entities.append(((name, ()),))
                            simple = True
                            break
                if not simple:
                    entities.append(reference__entity_key(entity))
            if len(entities) > 1:
                entities.sort()
            board.append((x, y, tuple(entities)))
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
        tuple(board),
    )


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
            # Search, replay and tracking only read these entities. Eligibility
            # excludes all dynamic components; key guards detect later mutation.
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


def vision_v4_search(
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
            return v4_state_key(s)
        t = perf_counter()
        key = v4_state_key(s)
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
