import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import vision_v4_backend as backend
from agents.vision_v4_planner import PlannerAdapter, StructuredAgent
from agents.vision_v4_types import EpisodeBudget


class BackendTests(unittest.TestCase):
    def test_basic_verified_win_prunes_uncompetitive_branches(self):
        from time import perf_counter

        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        start = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(start, start + 3, start + 3.5))
        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["A0 Basic Movement"]()
        planner.step(state)
        self.assertTrue(planner._actions)
        self.assertLess(planner._stats["transitions"], 100)
        for action in planner._actions:
            state = backend.grid_step(state, action)
        self.assertTrue(state.win)
        self.assertEqual(state.score, -9)

    def test_refinement_stops_soon_after_verified_win(self):
        from agents.vision_v4_search import v4_primitive_search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B2 The Vault"]()
        board, start, score = backend.encode(state)
        now = 0.0
        verified_at = []

        def clock():
            nonlocal now
            now += 0.00001
            return now

        def verify(actions, keys):
            result = backend.verify_plan(
                board, state, actions, keys, deadline=float("inf")
            )
            if result.status == "verified":
                verified_at.append(now)
            return result

        post_win_deadlines = []

        class SlowRefinementOrdering(backend.CompactOrdering):
            def estimate(self, *args, **kwargs):
                nonlocal now
                if verified_at and not post_win_deadlines:
                    post_win_deadlines.append(self.deadline)
                    now = min(self.deadline, now + 1.0)
                return super().estimate(*args, **kwargs)

        with patch("agents.vision_v4_search.CompactOrdering", SlowRefinementOrdering):
            result = v4_primitive_search(
                board,
                start,
                score,
                deadline=100,
                mode="p2b",
                verify_candidate=verify,
                clock=clock,
            )
        self.assertTrue(post_win_deadlines)
        self.assertTrue(result.winning)
        self.assertLess(now - verified_at[0], 0.152)
        self.assertEqual(
            backend.verify_plan(
                board,
                state,
                result.actions,
                result.expected_keys,
                deadline=float("inf"),
            ).status,
            "verified",
        )

    def test_backend_matches_frozen_bytes(self):
        raw = Path("agents/vision_v4_backend.py").read_bytes()
        self.assertEqual(raw, Path("artifacts/p3/frozen/task1_agent.py").read_bytes())
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4",
        )

    def test_default_structured_dispatch_is_p3_agent(self):
        self.assertIs(StructuredAgent, backend.Agent)
        self.assertIsNot(StructuredAgent, backend.Task3Agent)

    def test_adapter_preserves_origin_and_two_second_reference_cap(self):
        with patch.object(backend, "perf_counter", return_value=10):
            adapter = PlannerAdapter(EpisodeBudget(10, 13, 13.5))
        self.assertEqual(tuple(adapter.budget), (10, 13, 13.5))
        self.assertEqual(adapter._reference_initial_deadline, 12)
        self.assertEqual(adapter._constructed, 10)
        self.assertEqual(adapter._initial_deadline, 13)
        self.assertEqual(adapter._cutoff, 13.5)

    def test_reidentification_rebinds_verified_suffix(self):
        from tests import test_vision_v3_harness as fixtures

        with patch.object(fixtures, "PlannerAdapter", PlannerAdapter):
            fixtures.HarnessTests(
                "test_public_pickup_reidentification_retains_verified_suffix"
            ).test_public_pickup_reidentification_retains_verified_suffix()
            # Exercise reference rebinding explicitly: large keyed worlds now
            # get a verified compact proposal before reaching this backend.
            with patch(
                "agents.vision_v4_planner.v4_encode_proposal",
                side_effect=backend.UnsupportedState("Force reference coverage"),
            ):
                fixtures.HarnessTests(
                    "test_reference_suffix_survives_public_resource_reidentification"
                ).test_reference_suffix_survives_public_resource_reidentification()

    def test_late_fallback_never_renews_budget(self):
        from tests import test_vision_v3_harness as fixtures

        with patch.object(fixtures, "PlannerAdapter", PlannerAdapter):
            fixtures.HarnessTests(
                "test_planner_never_resets_original_deadline"
            ).test_planner_never_resets_original_deadline()

    def test_b4_dispatch_and_failed_replay(self):
        from tests import test_hierarchical_p3_dispatch as fixtures

        with patch.object(fixtures, "hierarchical_p3", backend):
            fixtures.DispatchTests(
                "test_long_verified_route_finishes_before_original_cutoff_on_slow_cpu"
            ).test_long_verified_route_finishes_before_original_cutoff_on_slow_cpu()
            fixtures.DispatchTests(
                "test_reference_replay_rejection_never_installs_partial_cache"
            ).test_reference_replay_rejection_never_installs_partial_cache()


class ControlTests(unittest.TestCase):
    def test_control_is_deterministic_and_binds_p3(self):
        import tempfile

        from tools.benchmark_planner_p1 import load_agent
        from tools.diagnose_vision_v4 import package_p3_control

        with tempfile.TemporaryDirectory() as d:
            a = package_p3_control(Path(d) / "a")
            b = package_p3_control(Path(d) / "b")
            self.assertEqual(a.read_bytes(), b.read_bytes())
            agent = load_agent(str(a))()
            self.assertEqual(agent._structured.__class__.__name__, "Agent")
            ns = agent._step_image.__globals__
            self.assertIs(ns["reference_state_key"], ns["P3_REFERENCE_STATE_KEY"])
            self.assertTrue(
                issubclass(ns["PlannerAdapter"], ns["BackendPlannerAdapter"])
            )
            self.assertIn("class Task3Agent", a.read_text())
            self.assertNotIsInstance(agent._structured, ns["Task3Agent"])


class TransitionReuseTests(unittest.TestCase):
    def test_verified_transitions_reuse_exact_objects_and_reject_changed_state(self):
        from copy import deepcopy
        from time import perf_counter

        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        action = planner.step(state)
        with patch.object(vision_v4_planner, "grid_step", wraps=backend.grid_step) as simulate:
            successor = planner.predict(state, action)
            self.assertEqual(simulate.call_count, 0)
            self.assertEqual(
                reference_key(successor),
                reference_key(backend.grid_step(state, action)),
            )
            modified = deepcopy(state)
            modified.score -= 3
            other = planner.predict(modified, action)
            self.assertEqual(simulate.call_count, 1)
            self.assertEqual(other.score, successor.score - 3)
        next_action = planner.step(successor)
        with patch.object(
            vision_v4_planner, "grid_step", side_effect=AssertionError("Repeated replay")
        ):
            self.assertTrue(planner.predict(successor, next_action).win)

    def test_mutated_source_or_successor_is_not_reused(self):
        from time import perf_counter

        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        action = planner.step(state)
        successor = planner.predict(state, action)
        state.score -= 3
        with patch.object(vision_v4_planner, "grid_step", wraps=backend.grid_step) as simulate:
            self.assertEqual(planner.predict(state, action).score, successor.score - 3)
            self.assertEqual(simulate.call_count, 1)
        state.score += 3
        successor.score -= 9
        with patch.object(vision_v4_planner, "grid_step", wraps=backend.grid_step) as simulate:
            self.assertNotEqual(planner.predict(state, action).score, successor.score)
            self.assertEqual(simulate.call_count, 1)

    def test_reference_verified_path_reuses_transitions(self):
        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        planner = vision_v4_planner.V4ReferenceAgent()
        action = planner.step(state)
        with patch.object(
            vision_v4_planner, "grid_step", side_effect=AssertionError("Repeated replay")
        ):
            successor = planner.predict(state, action)
            self.assertTrue(planner.predict(successor, planner.step(successor)).win)

    def test_invalid_replay_never_installs_edges(self):
        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        planner = vision_v4_planner.V4ReferenceAgent()
        result = planner._verify_cached(
            state,
            (backend.Action.RIGHT,),
            (reference_key(state), ("wrong",)),
            reference_key,
            float("inf"),
        )
        self.assertEqual(result.status, "invalid")
        self.assertFalse(planner._verified_edges)


def reference_key(state):
    return backend.reference_state_key(state)


class ReferenceTransitionTests(unittest.TestCase):
    def test_generic_public_transitions_match_specialized_public_states(self):
        import random

        from agents.vision_v4_search import v4_reference_transition
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        rng = random.Random(20261113)
        for label, build in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
            expected = actual = build()
            for _ in range(40):
                action = rng.choice(tuple(backend.Action))
                expected = backend.grid_step(expected, action)
                actual = v4_reference_transition(actual, action)
                self.assertEqual(reference_key(actual), reference_key(expected), label)
                if expected.win or expected.lose:
                    break

    def test_generic_search_preserves_pickups_effects_and_door_semantics(self):
        from agents.vision_v4_search import v4_reference_transition
        from tests.planner_p1_fixtures import mechanic

        for kind in (
            "key_door",
            "coin",
            "gem",
            "box_coin",
            "damage",
            "boots",
            "ghost",
            "shield",
        ):
            state, actions = mechanic(kind)
            expected = actual = state
            for action in actions:
                expected = backend.grid_step(expected, action)
                actual = v4_reference_transition(actual, action)
                self.assertEqual(
                    reference_key(actual), reference_key(expected), (kind, action)
                )


class LateExecutionTests(unittest.TestCase):
    def test_verified_suffix_can_execute_after_search_cutoff_without_new_search(self):
        from time import perf_counter

        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        state = corridor()
        now = perf_counter()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        planner._execution_cutoff = now + 4.65
        first = planner.step(state)
        state = planner.predict(state, first)
        with (
            patch.object(vision_v4_planner, "perf_counter", return_value=now + 3.8),
            patch.object(
                vision_v4_planner, "vision_v4_search", side_effect=AssertionError("Late search")
            ),
        ):
            self.assertEqual(planner.step(state), backend.Action.RIGHT)
        with patch.object(vision_v4_planner, "perf_counter", return_value=now + 4.7):
            self.assertEqual(planner.step(state), backend.Action.WAIT)

    def test_late_discrepancy_does_not_replan(self):
        from time import perf_counter

        from agents import vision_v4_planner
        from tests.planner_p1_fixtures import corridor

        now = perf_counter()
        state = corridor()
        planner = PlannerAdapter(EpisodeBudget(now, now + 3, now + 3.5))
        planner._execution_cutoff = now + 4.65
        planner.step(state)
        with (
            patch.object(vision_v4_planner, "perf_counter", return_value=now + 3.8),
            patch.object(
                vision_v4_planner, "vision_v4_search", side_effect=AssertionError("Late search")
            ),
        ):
            self.assertEqual(planner.step(state), backend.Action.WAIT)


class StaticConversionTests(unittest.TestCase):
    def test_reference_search_does_not_reconstruct_unchanged_walls(self):
        from grid_universe.grid.convert import _entity_object_from_state

        from agents import vision_v4_search
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
        entities = sum(len(cell) for col in state.grid for cell in col)
        with patch.object(
            vision_v4_search, "v4_entity_from_state", wraps=_entity_object_from_state
        ) as convert:
            result = vision_v4_search.v4_reference_transition(state, backend.Action.DOWN)
        self.assertLess(convert.call_count, entities // 2)
        self.assertEqual(
            reference_key(result),
            reference_key(backend.grid_step(state, backend.Action.DOWN)),
        )
