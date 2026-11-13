"""Boss-mode entry point with a 15-second budget for either observation format."""

from time import perf_counter

from grid_adventure.grid import GridState
from grid_adventure.step import Action

from agents.vision_v4 import Agent
from agents.vision_v4_backend import (
    Task3Agent,
    UnsupportedState,
    observe_key,
    reference_state_key,
)
from agents.vision_v4_types import EpisodeBudget


class BossStructuredAgent(Task3Agent):
    """Reserve the end of the episode for dispatch of an already verified route."""

    def __init__(self, budget, execution_cutoff):
        super().__init__(budget)
        self._execution_cutoff = execution_cutoff

    def step(self, state):
        now = perf_counter()
        if now < min(self._cutoff, self.budget.cutoff):
            active = self._reference if self._reference is not None else self
            actions, index = active._actions, active._next
            action = super().step(state)
            if action != Action.WAIT:
                return action
            now = perf_counter()
            current = self._reference if self._reference is not None else self
            if (
                now < min(self._cutoff, self.budget.cutoff)
                or current is not active
                or active._actions is not actions
                or active._next != index
            ):
                return action
            # The frozen delegate may cross its cutoff before dispatch. Retry
            # this observation only if it did not replace or consume the route;
            # a verified WAIT action must never consume a second action here.
        if (
            now >= self._execution_cutoff
            or not isinstance(state, GridState)
            or state.win
            or state.lose
        ):
            return Action.WAIT
        # Do not enter the frozen adapter after its planning deadline: it may
        # search or replay a suffix to rebind mismatching resource identities.
        active = self._reference if self._reference is not None else self
        if active._next >= len(active._actions):
            return Action.WAIT
        try:
            observed = (
                reference_state_key(state)
                if self._reference is not None
                else observe_key(self._board, state)
            )
        except UnsupportedState:
            return Action.WAIT
        if (
            observed != active._keys[active._next]
            or perf_counter() >= self._execution_cutoff
        ):
            return Action.WAIT
        action = active._actions[active._next]
        active._next += 1
        active._stats["cache_hits"] += 1
        return action


class BossAgent(Agent):
    """Give Task 3 longer deadlines while retaining the original construction time."""

    def __init__(self):
        super().__init__()
        # Include cold setup time; only this entry point selects the boss budget.
        started = self._budget.started_at
        self._budget = EpisodeBudget(started, started + 11.0, started + 14.65)
        self._planning_budget = EpisodeBudget(started, started + 11.0, started + 12.0)
        self._structured = BossStructuredAgent(
            self._planning_budget, self._budget.cutoff
        )
