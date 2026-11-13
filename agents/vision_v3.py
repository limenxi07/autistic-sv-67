"""Image-tracking agent (V3), backed by the P2-next planner."""

from time import perf_counter
from typing import Any

from grid_adventure.grid import GridState
from grid_adventure.grid import step as grid_step
from grid_adventure.step import Action

from agents.vision_v3_bridge import split_cells
from agents.vision_v3_model import TEMPERATURE, get_model
from agents.vision_v3_perception import (
    classify_initial,
    configure,
    make_actor_template,
)
from agents.vision_v3_planner import (
    BACKEND_SHA256,
    PlannerAdapter,
    StructuredAgent,
)
from agents.vision_v3_tracking import (
    PredictionHistory,
    agent_position,
    check_observation,
    common_safe_prefix,
    fatal_uncertainty,
    initial_hypotheses,
    reconcile,
    visible_kind,
)
from agents.vision_v3_types import BoardHypothesis, EpisodeBudget


class PerceptionUncertainty(ValueError):
    """The observation has no complete bounded reconstruction for visualization."""


class Agent:
    """Classify observations, track board hypotheses and request verified actions."""
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
        configure(get_model, TEMPERATURE)

    def parse(self, observation):
        try:
            turn = observation["info"]["status"]["turn"]
            history = self._hypotheses + (
                self._history.hypotheses if self._history is not None else ()
            )
            previous = next((h.state for h in history if h.state.turn == turn), None)
            hypotheses = initial_hypotheses(
                classify_initial(observation), observation["info"], previous=previous
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
        try:
            return self._step_image(observation)
        except (ValueError, KeyError, TypeError, RuntimeError) as error:
            self._stats["reason"] = "unsupported_observation:" + type(error).__name__
            return Action.WAIT

    def _step_image(self, observation):
        if perf_counter() >= self._budget.cutoff:
            self._stats["reason"] = "cutoff"
            return Action.WAIT
        if observation["info"]["status"]["phase"] in ("win", "lose"):
            self._stats["reason"] = "terminal"
            return Action.WAIT
        repairing = False
        if not self._hypotheses:
            hypotheses = initial_hypotheses(
                classify_initial(observation), observation["info"]
            )
        else:
            check = check_observation(
                observation,
                self._hypotheses[0].state,
                self._previous_image,
                background_image=self._background_image,
                actor_template=self._actor_template,
            )
            if not check.consistent:
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
        self._stats["hypotheses"] = len(hypotheses)
        if not hypotheses or any(fatal_uncertainty(h) for h in hypotheses):
            self._stats["reason"] = "unresolved_perception"
            return Action.WAIT
        if perf_counter() >= self._budget.cutoff:
            self._stats["reason"] = "cutoff"
            return Action.WAIT
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
            self._stats["reason"] = "hypothesis_disagreement_or_cutoff"
            return Action.WAIT
        if repairing:
            self._pending_recovery = True
        self._history = PredictionHistory(hypotheses, action)
        if self._background_image is None:
            self._background_image = observation["image"].copy()
            state = hypotheses[0].state
            floors = [
                (x, y)
                for x, col in enumerate(state.grid)
                for y, cell in enumerate(col)
                if not cell
            ]
            self._actor_template = make_actor_template(
                observation, agent_position(state), floors
            )
        else:
            current = split_cells(observation)
            background = split_cells({**observation, "image": self._background_image})
            state = hypotheses[0].state
            for y in range(state.height):
                for x in range(state.width):
                    if visible_kind(state.grid[x][y]) not in ("human", "box"):
                        background[y][x][:] = current[y][x]
        self._hypotheses = tuple(
            BoardHypothesis(grid_step(h.state, action), h.confidence, h.unresolved)
            for h in hypotheses
        )
        self._previous_image = observation["image"].copy()
        self._stats["actions"] += 1
        active = (
            self._planner._reference
            if self._planner._reference is not None
            else self._planner
        )
        self._stats["reason"] = (
            "planner_no_verified_plan"
            if action == Action.WAIT and not active._actions
            else "conditional_plan"
        )
        return action

    def info(self):
        return {
            "name": "V3 vision",
            "backend_sha256": BACKEND_SHA256,
            **{key: str(value) for key, value in self._stats.items()},
        }
