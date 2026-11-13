"""Connect V4 tracking to the frozen P3 planner and verify routes before dispatch."""

from time import perf_counter

from agents.vision_v4_backend import (
    PRODUCTION_MODE,
    Action,
    GridState,
    ImageObservation,
    ReferenceAgent,
    UnsupportedState,
    Verification,
    encode,
    grid_step,
    reference_PRODUCTION_MODE,
    reference_state_key,
)
from agents.vision_v4_backend import Agent as StructuredAgent
from agents.vision_v4_backend import (
    PlannerAdapter as BackendPlannerAdapter,
)
from agents.vision_v4_search import (
    v4_encode_proposal,
    v4_observe_key,
    v4_reference_search,
    v4_reference_transition,
    v4_state_key,
    vision_v4_search,
)
from agents.vision_v4_types import EpisodeBudget

BACKEND_SOURCE = "agents/vision_v4_backend.py"
BACKEND_SHA256 = "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4"
__all__ = [
    "BACKEND_SHA256",
    "BACKEND_SOURCE",
    "PlannerAdapter",
    "StructuredAgent",
    "reference_state_key",
]


class _V4ReplayCache:
    """Cache independently verified transitions for safe route reuse."""

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
                    if key_of is v4_state_key and previous_reference_key is not None
                    else key_of(current)
                )
            except UnsupportedState:
                return Verification("invalid", None, ())
            actual.append(key)
            if key != expected_keys[index]:
                return Verification("invalid", None, ())
            if index < len(actions):
                action = actions[index]
                # Re-simulate the complete route with the public native engine.
                # Its converter retains unchanged scenery; every dynamic entity
                # is rebuilt and every expected state is still checked below.
                successor = v4_reference_transition(current, action)
                if len(actions) <= 256:
                    source_reference_key = (
                        key
                        if key_of is v4_state_key
                        else previous_reference_key
                        if previous_reference_key is not None
                        else v4_state_key(current)
                    )
                    previous_reference_key = v4_state_key(successor)
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
        # Cache only independently verified winning routes.
        self._verified_edges = edges
        return Verification("verified", current.score, tuple(actual))

    def predict(self, state, action):
        owner = getattr(self, "_reference", None) or self
        entry = owner._verified_edges.get((id(state), action))
        if (
            entry is not None
            and entry[0] is state
            and v4_state_key(state) == entry[1]
            and v4_state_key(entry[2]) == entry[3]
        ):
            return entry[2]
        return grid_step(state, action)


class V4ReferenceAgent(_V4ReplayCache, ReferenceAgent):
    """Run reference search using the original episode deadlines."""

    def step(self, state: GridState | ImageObservation) -> Action:
        if not isinstance(state, GridState) or state.win or state.lose:
            return Action.WAIT
        if not self._actions and self._tried and (perf_counter() >= self._cutoff):
            return Action.WAIT
        observed = v4_state_key(state)
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
                v4_state_key,
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
    """Verify compact proposals and reserve time to execute the accepted route."""

    _execution_cutoff: float

    def __init__(self, budget: EpisodeBudget, mode: str = "p2b"):
        super().__init__(
            budget,
            mode=mode,
            reference_initial_deadline=min(
                budget.initial_deadline, budget.started_at + 2.0
            ),
        )
        # If perception spent the initial window, recover on the same clock.
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
                    v4_state_key(state)
                    if self._reference is not None
                    else v4_observe_key(self._board, state)
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
            if backend._next < len(backend._actions):
                if v4_state_key(state) == backend._keys[backend._next]:
                    if perf_counter() >= self._cutoff:
                        return Action.WAIT
                    action = backend._actions[backend._next]
                    backend._next += 1
                    backend._stats["cache_hits"] += 1
                    return action
                self._rebind_reference_suffix(state)
        elif self._next < len(self._actions):
            try:
                agrees = v4_observe_key(self._board, state) == self._keys[self._next]
            except UnsupportedState:
                agrees = False
            if agrees:
                if perf_counter() >= self._cutoff:
                    return Action.WAIT
                action = self._actions[self._next]
                self._next += 1
                self._stats["cache_hits"] += 1
                return action
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
                and v4_observe_key(self._board, state) == self._keys[self._next]
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
            speculative = False
            try:
                board, start, score = encode(state)
            except UnsupportedState as error:
                if str(error) != "Large keyed inventory iteration history":
                    raise
                board, start, score = v4_encode_proposal(state)
                speculative = True
                # Reserve the existing reference window if no compact route
                # verifies. Replay retains the original absolute cutoff.
                deadline = min(deadline, perf_counter() + 0.75)
            self._board = board
            result = vision_v4_search(
                board,
                start,
                score,
                deadline=deadline,
                mode=getattr(self, "_mode", PRODUCTION_MODE),
                verify_candidate=lambda actions, keys: self._verify_cached(
                    state,
                    actions,
                    keys,
                    lambda current: v4_observe_key(board, current),
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
            if speculative or result.stop_reason == "invalid":
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
            # Late construction enters recovery without extending the deadline.
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
