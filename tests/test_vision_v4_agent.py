import unittest
from copy import deepcopy
from unittest.mock import patch

from grid_adventure.entities import ExitEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.vision_v4 import Agent, PerceptionUncertainty
from tests.planner_p1_fixtures import corridor
from tests.vision_v3_fixtures import vision_observation


class RecoveryTests(unittest.TestCase):
    def test_safe_dispatch_reuses_publicly_verified_transition(self):
        observation = vision_observation(corridor())
        agent = Agent()
        with (
            patch(
                "agents.vision_v4.grid_step", wraps=step
            ) as emission_step,
            patch(
                "agents.vision_v4_tracking.grid_step", wraps=step
            ) as safety_step,
        ):
            self.assertEqual(agent.step(observation), Action.RIGHT)
        self.assertEqual(emission_step.call_count + safety_step.call_count, 0)
        self.assertTrue(agent._last_emission.prediction_valid)

    def test_cold_perception_uses_recovery_without_an_extra_wait(self):
        from agents.vision_v4_perception import classify_evidence

        state = corridor()
        observation = vision_observation(state)
        # Measure real image evidence before the simulated cold-start interval.
        Agent()
        evidence = classify_evidence(observation)
        agent = Agent()
        budget = tuple(agent._budget)
        now = agent._budget.started_at

        def slow_perception(_observation):
            nonlocal now
            now = agent._budget.initial_deadline + 0.05
            return evidence

        with (
            patch(
                "agents.vision_v4.perf_counter", side_effect=lambda: now
            ),
            patch(
                "agents.vision_v4_tracking.perf_counter",
                side_effect=lambda: now,
            ),
            patch(
                "agents.vision_v4_backend.perf_counter",
                side_effect=lambda: now,
            ),
            patch(
                "agents.vision_v4_planner.perf_counter",
                side_effect=lambda: now,
            ),
            patch(
                "agents.vision_v4.classify_evidence",
                side_effect=slow_perception,
            ),
        ):
            self.assertEqual(agent.step(observation), Action.RIGHT)
            self.assertEqual(agent.info()["reason"], "conditional_plan")
            assert agent._last_emission is not None
            self.assertTrue(agent._last_emission.prediction_valid)
            state = step(state, Action.RIGHT)
            self.assertEqual(agent.step(vision_observation(state)), Action.RIGHT)
            self.assertEqual(agent.info()["reason"], "conditional_plan")
        self.assertEqual(tuple(agent._budget), budget)

    def test_planless_wait_retains_attempt_without_cached_actions(self):
        agent = Agent()
        state = corridor()
        agent.step(vision_observation(state))
        planner = agent._planner
        assert planner is not None
        planner._actions = planner._keys = ()
        planner._next = 0
        agent._emission_observation = vision_observation(state)
        # The first observed state is available in the committed history.
        assert agent._history is not None
        hypotheses = agent._history.hypotheses
        agent._emit_action(
            Action.WAIT,
            turn=state.turn,
            hypotheses=hypotheses,
            reason="planner_no_verified_plan",
        )
        self.assertIs(agent._planner, planner)
        assert agent._last_emission is not None
        self.assertTrue(agent._last_emission.prediction_valid)

    def test_repeated_unchanged_ambiguity_keeps_verified_planner(self):
        from grid_adventure.entities import BoxEntity

        from agents.vision_v4_types import ObservationCheck
        from tests.planner_p1_fixtures import layout

        agent = Agent()
        state = layout(width=5)
        state.add((0, 0), BoxEntity())
        state = step(state, agent.step(vision_observation(state)))
        planner = agent._planner
        agent._hypotheses = tuple(
            h._replace(
                unresolved=h.unresolved + ((0, 0, "ambiguous", ("box", "wall")),)
            )
            for h in agent._hypotheses
        )
        p = [0.0] * 14
        from agents.vision_v4_bridge import CLASSES

        p[CLASSES.index("box")] = 0.7
        p[CLASSES.index("wall")] = 0.3
        check = ObservationCheck(
            False,
            ((0, 1), (1, 1)),
            (("uncertain_cell", (0, 0), "box", "box", tuple(p)),),
        )
        from agents.vision_v4_tracking import repeated_ambiguity

        self.assertTrue(repeated_ambiguity(check, agent._hypotheses))
        self.assertFalse(
            repeated_ambiguity(
                check._replace(changed_cells=((0, 0),)), agent._hypotheses
            )
        )
        self.assertFalse(
            repeated_ambiguity(
                check._replace(mismatches=(("score", 0, -1),)), agent._hypotheses
            )
        )
        self.assertFalse(
            repeated_ambiguity(
                check, tuple(h._replace(unresolved=()) for h in agent._hypotheses)
            )
        )
        with patch(
            "agents.vision_v4.check_observation", return_value=check
        ):
            agent.step(vision_observation(state))
        self.assertIs(agent._planner, planner)
        self.assertEqual(agent.info()["repairs"], "0")

    def test_wait_fallback_predicts_actual_wait(self):
        agent = Agent()
        state = corridor()
        action = agent.step(vision_observation(state))
        state = step(state, action)
        with patch(
            "agents.vision_v4.common_safe_prefix", return_value=()
        ):
            result = agent.step(vision_observation(state))
        self.assertEqual(result, Action.WAIT)
        assert agent._last_emission is not None
        self.assertEqual(agent._last_emission.action, result)
        assert agent._last_emission is not None
        self.assertTrue(agent._last_emission.prediction_valid)
        self.assertEqual(agent._hypotheses[0].state.turn, state.turn + 1)
        self.assertEqual(agent._hypotheses[0].state.score, step(state, result).score)
        self.assertEqual(agent.info()["actions"], "2")
        self.assertIsNone(agent._planner)

    def test_exception_then_valid_frame_does_not_loop_on_stale_turn(self):
        from grid_adventure.entities import GemEntity

        state = corridor()
        state.add((1, 0), GemEntity())
        agent = Agent()
        budget = tuple(agent._budget)
        state = step(state, agent.step(vision_observation(state)))
        with patch(
            "agents.vision_v4.check_observation",
            side_effect=ValueError("one bad frame"),
        ):
            action = agent.step(vision_observation(state))
        self.assertEqual(action, Action.WAIT)
        self.assertEqual(agent._hypotheses[0].state.turn, state.turn + 1)
        state = step(state, action)
        action = agent.step(vision_observation(state))
        self.assertEqual(action, Action.PICK_UP)
        self.assertEqual(agent._hypotheses[0].state.turn, state.turn + 1)
        self.assertEqual(agent.info()["actions"], "3")
        self.assertEqual(tuple(agent._budget), budget)
        self.assertNotIn("unsupported_observation", agent.info()["reason"])

    def test_cutoff_emits_without_more_model_or_grid_calls(self):
        agent = Agent()
        with (
            patch(
                "agents.vision_v4.perf_counter",
                return_value=agent._budget.cutoff + 1,
            ),
            patch(
                "agents.vision_v4.classify_evidence",
                side_effect=AssertionError("late model"),
            ),
            patch(
                "agents.vision_v4.grid_step",
                side_effect=AssertionError("late replay"),
            ),
        ):
            self.assertEqual(agent.step(vision_observation(corridor())), Action.WAIT)
        assert agent._last_emission is not None
        self.assertFalse(agent._last_emission.prediction_valid)
        self.assertEqual(agent._hypotheses, ())
        self.assertIsNone(agent._planner)
        self.assertEqual(agent.info()["actions"], "1")

    def test_partial_repair_exception_does_not_commit(self):
        state = corridor()
        agent = Agent()
        state = step(state, agent.step(vision_observation(state)))
        before = deepcopy(agent._hypotheses)
        with patch(
            "agents.vision_v4.visible_kind",
            side_effect=RuntimeError("emission failed"),
        ):
            action = agent.step(vision_observation(state))
        self.assertEqual(action, Action.WAIT)
        assert agent._last_emission is not None
        self.assertFalse(agent._last_emission.prediction_valid)
        self.assertEqual(agent._hypotheses, ())
        self.assertIsNone(agent._history)
        self.assertIsNone(agent._planner)
        self.assertEqual(before[0].state.turn, state.turn)

    def test_known_covered_exit_parses_without_mutation(self):
        from grid_adventure.entities import GemEntity

        state = corridor()
        state.add((1, 0), GemEntity())
        # Reach exit with a remaining required gem so it is nonterminal.
        agent = Agent()
        first = agent.step(vision_observation(state))
        state = step(state, first)
        # Force actual and predicted supported move onto exit.
        observation = vision_observation(state)
        with patch(
            "agents.vision_v4.PlannerAdapter.step",
            return_value=Action.RIGHT,
        ):
            action = agent.step(observation)
        state = step(state, action)
        self.assertFalse(state.win)
        before = (agent.info(), agent._hypotheses, agent._history, agent._last_emission)
        parsed = agent.parse(vision_observation(state))
        self.assertTrue(any(isinstance(e, ExitEntity) for e in parsed.grid[2][0]))
        self.assertEqual(
            (agent.info(), agent._hypotheses, agent._history, agent._last_emission),
            before,
        )

    def test_unknown_covered_layers_remain_unresolved(self):
        agent = Agent()
        state = step(corridor(), Action.WAIT)
        with self.assertRaises(PerceptionUncertainty):
            agent.parse(vision_observation(state))

    def test_invalid_turn_breaks_continuity(self):
        agent = Agent()
        state = corridor()
        agent.step(vision_observation(state))
        bad = vision_observation(state)
        bad["info"]["status"]["turn"] = None
        self.assertEqual(agent.step(bad), Action.WAIT)
        assert agent._last_emission is not None
        self.assertFalse(agent._last_emission.prediction_valid)
        self.assertEqual(agent._hypotheses, ())
        self.assertIsNone(agent._planner)

    def test_terminal_emission_does_not_advance_world(self):
        agent = Agent()
        state = step(step(corridor(), Action.RIGHT), Action.RIGHT)
        with patch(
            "agents.vision_v4.grid_step",
            side_effect=AssertionError("terminal replay"),
        ):
            self.assertEqual(agent.step(vision_observation(state)), Action.WAIT)
        assert agent._last_emission is not None
        self.assertEqual(agent._last_emission.reason, "terminal")
        assert agent._last_emission is not None
        self.assertFalse(agent._last_emission.prediction_valid)
        self.assertEqual(agent.info()["actions"], "0")

    def test_unsupported_frame_then_later_visible_frame_drops_stale_history(self):
        agent = Agent()
        state = corridor()
        state = step(state, agent.step(vision_observation(state)))
        # Missing actual turn makes advancing a prediction impossible.
        observation = vision_observation(state)
        del observation["info"]["status"]["turn"]
        action = agent.step(observation)
        self.assertEqual(action, Action.WAIT)
        self.assertEqual(agent._hypotheses, ())
        self.assertIsNone(agent._history)
        state = step(state, action)
        agent.step(vision_observation(state))
        self.assertNotIn("unsupported_observation", agent.info()["reason"])
        # An unknown covered layer remains unresolved; no synthetic continuity.
        self.assertEqual(agent._hypotheses, ())

    def test_exception_after_contradiction_cannot_advance_disproved_state(self):
        from agents.vision_v4_types import ObservationCheck

        state = corridor()
        agent = Agent()
        state = step(state, agent.step(vision_observation(state)))
        check = ObservationCheck(False, (), (("agent_position", (1, 0), (0, 0)),))
        with (
            patch(
                "agents.vision_v4.check_observation", return_value=check
            ),
            patch(
                "agents.vision_v4.reconcile",
                side_effect=RuntimeError("partial repair"),
            ),
        ):
            self.assertEqual(agent.step(vision_observation(state)), Action.WAIT)
        self.assertEqual(agent._hypotheses, ())
        self.assertIsNone(agent._history)
        self.assertIsNone(agent._planner)
        assert agent._last_emission is not None
        self.assertFalse(agent._last_emission.prediction_valid)

    def test_terminal_after_cutoff_is_not_counted_as_an_emitted_transition(self):
        agent = Agent()
        state = step(step(corridor(), Action.RIGHT), Action.RIGHT)
        with (
            patch(
                "agents.vision_v4.perf_counter",
                return_value=agent._budget.cutoff + 1,
            ),
            patch(
                "agents.vision_v4.grid_step",
                side_effect=AssertionError("terminal transition"),
            ),
        ):
            self.assertEqual(agent.step(vision_observation(state)), Action.WAIT)
        assert agent._last_emission is not None
        self.assertEqual(agent._last_emission.reason, "terminal")
        self.assertEqual(agent.info()["actions"], "0")

    def test_invalid_geometry_fallback_does_not_replay_old_board(self):
        agent = Agent()
        state = corridor()
        state = step(state, agent.step(vision_observation(state)))
        observation = vision_observation(state)
        observation["info"]["config"]["width"] = 999
        with (
            patch(
                "agents.vision_v4.check_observation",
                side_effect=RuntimeError("bad frame"),
            ),
            patch(
                "agents.vision_v4.grid_step",
                side_effect=AssertionError("invalid geometry replay"),
            ),
        ):
            self.assertEqual(agent.step(observation), Action.WAIT)
        self.assertEqual(agent._hypotheses, ())
        assert agent._last_emission is not None
        self.assertFalse(agent._last_emission.prediction_valid)
