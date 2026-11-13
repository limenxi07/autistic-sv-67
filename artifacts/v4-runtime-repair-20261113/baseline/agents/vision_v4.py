"""Image-recovery agent (V4), backed by the frozen P3 planner."""

from time import perf_counter
from typing import Any

from grid_adventure.grid import GridState
from grid_adventure.grid import step as grid_step
from grid_adventure.step import Action

from agents.vision_v4_bridge import split_cells
from agents.vision_v4_model import POLICY, TEMPERATURE, get_model
from agents.vision_v4_perception import (
    classify_evidence,
    configure,
    make_actor_template,
)
from agents.vision_v4_planner import (
    BACKEND_SHA256,
    PlannerAdapter,
    StructuredAgent,
)
from agents.vision_v4_tracking import (
    PredictionHistory,
    agent_position,
    check_observation,
    common_safe_prefix,
    fatal_uncertainty,
    initial_hypotheses,
    reconcile,
    repeated_ambiguity,
    visible_kind,
)
from agents.vision_v4_types import (
    BoardHypothesis,
    Emission,
    EpisodeBudget,
)


class PerceptionUncertainty(ValueError):
    """The observation has no complete bounded reconstruction for visualization."""


class Agent:
    """Track observations and recover planning without restarting the episode clock."""
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
