"""Reference incumbents must leave time for replay and a long public episode."""

import unittest
from unittest.mock import patch

from grid_adventure.step import Action

from agents import hierarchical_p3
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS


def slow_episode(cost, *, entry="Agent"):
    clock = [0.0]
    public_step = hierarchical_p3.grid_step
    original_search = hierarchical_p3.reference_search
    searches = []

    def transition(state, action):
        clock[0] += cost
        return public_step(state, action)

    def search(state, **kwargs):
        kwargs["clock"] = lambda: clock[0]
        result = original_search(state, **kwargs)
        searches.append((clock[0], result.winning, len(result.actions)))
        return result

    state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
    state.seed = 42
    state.turn_limit = 150
    trace = []
    with (
        patch.object(
            hierarchical_p3, "perf_counter", side_effect=lambda: clock[0]
        ),
        patch.object(hierarchical_p3, "grid_step", side_effect=transition),
        patch.object(hierarchical_p3, "reference_search", side_effect=search),
    ):
        agent = (
            hierarchical_p3.Agent()
            if entry == "Agent"
            else hierarchical_p3.PlannerAdapter(
                hierarchical_p3.EpisodeBudget(0, 3, 3.5)
            )
        )
        while not (state.win or state.lose):
            action = agent.step(state)
            trace.append(action)
            state = transition(state, action)
    return state, clock[0], searches, trace


class DispatchTests(unittest.TestCase):
    def test_long_verified_route_finishes_before_original_cutoff_on_slow_cpu(self):
        state, elapsed, searches, trace = slow_episode(0.0055)
        self.assertTrue(searches[0][1], "search must already have an incumbent")
        self.assertTrue(state.win, (state.score, elapsed, trace.count(Action.WAIT)))
        self.assertEqual(state.score, -387)
        self.assertLess(elapsed, 3.5)
        self.assertNotIn(Action.WAIT, trace)

    def test_plain_agent_late_reference_keeps_original_clock_origin(self):
        from tests.planner_p1_fixtures import corridor

        clock = [0.0]
        empty = hierarchical_p3.reference_SearchResult(
            (), (), False, None, "exhausted", 0
        )
        with (
            patch.object(
                hierarchical_p3, "perf_counter", side_effect=lambda: clock[0]
            ),
            patch.object(
                hierarchical_p3, "reference_search", return_value=empty
            ) as search,
        ):
            agent = hierarchical_p3.Agent()
            clock[0] = 3.1
            agent._fallback(corridor())
            self.assertEqual(agent._reference._constructed, 0)
            self.assertEqual(agent._reference._cutoff, 3.5)
            self.assertLessEqual(search.call_args.kwargs["deadline"], 3.5)
            self.assertEqual(search.call_args.kwargs["dispatch_cutoff"], 3.5)

    def test_structural_adapter_uses_the_same_dispatch_reserve(self):
        state, elapsed, _, _trace = slow_episode(0.0055, entry="adapter")
        self.assertTrue(state.win)
        self.assertEqual(state.score, -387)
        self.assertLess(elapsed, 3.5)

    def test_reference_replay_rejection_never_installs_partial_cache(self):
        from time import perf_counter

        from tests.planner_p1_fixtures import corridor

        state = corridor()
        plan = hierarchical_p3.reference_search(
            state, deadline=perf_counter() + 1, mode="bfs"
        )
        self.assertTrue(plan.winning)
        for failure in ("invalid", "deadline"):
            with self.subTest(failure=failure):
                clock = [0.0]
                keys = list(plan.expected_keys)
                if failure == "invalid":
                    key = list(keys[-1])
                    key[8] += 1
                    keys[-1] = tuple(key)
                proposed = plan._replace(expected_keys=tuple(keys))
                original = hierarchical_p3.grid_step

                def transition(s, a, clock=clock, failure=failure, original=original):
                    if failure == "deadline":
                        clock[0] += 2
                    return original(s, a)

                with (
                    patch.object(
                        hierarchical_p3,
                        "perf_counter",
                        side_effect=lambda clock=clock: clock[0],
                    ),
                    patch.object(
                        hierarchical_p3,
                        "reference_search",
                        return_value=proposed,
                    ),
                    patch.object(
                        hierarchical_p3, "grid_step", side_effect=transition
                    ),
                ):
                    backend = hierarchical_p3.ReferenceAgent()
                    backend._cutoff = backend._initial_deadline = 1.0
                    action = backend.step(state)
                self.assertEqual(action, Action.WAIT)
                self.assertEqual(backend._actions, ())
                self.assertEqual(backend._keys, ())
                self.assertEqual(backend._next, 0)
                self.assertEqual(backend.info()["verification_status"], failure)


if __name__ == "__main__":
    unittest.main()
