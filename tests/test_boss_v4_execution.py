"""Boss dispatch keeps verified work usable after the planning deadline."""

import unittest
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import boss_v4, vision_v4
from agents import vision_v4_backend as backend
from tests.planner_p1_fixtures import corridor
from tools.package_vision_v3 import inline
from tools.package_vision_v4 import MODULES


class BossExecutionTests(unittest.TestCase):
    def setUp(self):
        self.started = perf_counter()
        self.now = self.started
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (boss_v4, vision_v4, backend):
            self.stack.enter_context(
                patch.object(module, "perf_counter", lambda: self.now, create=True)
            )

    def planned(self, reference=False):
        self.now = self.started
        agent = boss_v4.BossAgent()
        state = corridor()
        # Starting at six seconds also catches accidental Task 2 deadlines and
        # reference constructors that renew the episode's original time origin.
        self.now = self.started + 6
        with ExitStack() as planning:
            if reference:
                planning.enter_context(
                    patch.object(
                        backend, "encode", side_effect=backend.UnsupportedState
                    )
                )
            self.assertEqual(agent.step(state), Action.RIGHT)
        active = agent._structured._reference if reference else agent._structured
        self.assertIsNotNone(active)
        self.assertLess(active._next, len(active._actions))
        return agent, active, state, step(state, Action.RIGHT)

    def forbid_planning(self, agent):
        guards = ExitStack()
        for name in (
            "search",
            "reference_search",
            "verify_plan",
            "reference_verify_plan",
        ):
            guards.enter_context(
                patch.object(backend, name, side_effect=AssertionError("Late " + name))
            )
        for name in ("_rebind_suffix", "_rebind_reference_suffix"):
            guards.enter_context(
                patch.object(
                    agent._structured, name, side_effect=AssertionError("Late " + name)
                )
            )
        return guards

    def test_verified_direct_and_reference_suffixes_finish_after_planning_cutoff(self):
        for reference in (False, True):
            for elapsed in (12, 14.64):
                with self.subTest(reference=reference, elapsed=elapsed):
                    agent, active, _, current = self.planned(reference)
                    budgets = (
                        agent._budget,
                        agent._planning_budget,
                        active._constructed,
                        active._initial_deadline,
                        active._cutoff,
                    )
                    self.now = self.started + elapsed
                    with self.forbid_planning(agent):
                        action = agent.step(current)
                    self.assertEqual(action, Action.RIGHT)
                    self.assertTrue(step(current, action).win)
                    self.assertEqual(active._next, len(active._actions))
                    self.assertEqual(
                        (
                            agent._budget,
                            agent._planning_budget,
                            active._constructed,
                            active._initial_deadline,
                            active._cutoff,
                        ),
                        budgets,
                    )

    def test_late_mismatch_and_expiration_preserve_unconsumed_suffix(self):
        for reference in (False, True):
            with self.subTest(reference=reference):
                agent, active, initial, current = self.planned(reference)
                self.now = self.started + 13
                with self.forbid_planning(agent):
                    self.assertEqual(agent.step(initial), Action.WAIT)
                self.assertEqual(active._next, 1)
                self.now = self.started + 14.65
                self.assertEqual(agent.step(current), Action.WAIT)
                self.assertEqual(active._next, 1)

    def test_dispatch_rechecks_expiration_after_observation_key(self):
        for reference in (False, True):
            with self.subTest(reference=reference):
                agent, active, _, current = self.planned(reference)
                with patch.object(
                    boss_v4,
                    "perf_counter",
                    side_effect=(self.started + 13, self.started + 14.65),
                    create=True,
                ):
                    self.assertEqual(agent.step(current), Action.WAIT)
                self.assertEqual(active._next, 1)

    def test_delegate_crossing_planning_cutoff_dispatches_same_unconsumed_state(self):
        for reference in (False, True):
            with self.subTest(reference=reference):
                agent, active, _, current = self.planned(reference)
                first_clock_read = True

                def crossing_clock():
                    nonlocal first_clock_read
                    if first_clock_read:
                        first_clock_read = False
                        self.now = self.started + 12.01
                        return self.started + 11.99
                    return self.now

                with self.forbid_planning(agent), patch.object(
                    boss_v4, "perf_counter", side_effect=crossing_clock
                ):
                    action = agent.step(current)
                    self.assertEqual(action, Action.RIGHT)
                    self.assertTrue(step(current, action).win)
                    self.assertEqual(active._next, 2)
                    self.assertEqual(agent.step(current), Action.WAIT)
                    self.assertEqual(active._next, 2)

    def test_delegate_verified_wait_consumes_only_one_action_at_cutoff(self):
        for reference in (False, True):
            with self.subTest(reference=reference):
                agent, active, initial, _ = self.planned(reference)
                actions = (Action.WAIT, Action.RIGHT, Action.RIGHT)
                states = [initial]
                for action in actions:
                    states.append(step(states[-1], action))
                if reference:
                    keys = tuple(backend.reference_state_key(state) for state in states)
                    verified = backend.reference_verify_plan(
                        initial, actions, keys, deadline=self.started + 12
                    )
                else:
                    keys = tuple(backend.observe_key(active._board, state) for state in states)
                    verified = backend.verify_plan(
                        active._board, initial, actions, keys, deadline=self.started + 12
                    )
                self.assertEqual(verified.status, "verified")
                active._actions, active._keys, active._next = actions, verified.actual_keys, 0
                self.now = self.started + 11.99
                with self.forbid_planning(agent), patch.object(
                    boss_v4, "perf_counter", side_effect=(
                        self.started + 11.99, self.started + 12.01
                    )
                ):
                    self.assertEqual(agent.step(initial), Action.WAIT)
                self.assertEqual(active._next, 1)

    def test_late_unplanned_agent_does_not_search(self):
        agent = boss_v4.BossAgent()
        self.now = self.started + 12
        with self.forbid_planning(agent):
            self.assertEqual(agent.step(corridor()), Action.WAIT)

    def test_terminal_and_non_grid_observations_do_not_consume_suffix(self):
        agent, active, _, current = self.planned()
        terminal = step(current, Action.RIGHT)
        self.now = self.started + 13
        self.assertEqual(agent._structured.step(terminal), Action.WAIT)
        self.assertEqual(agent._structured.step(None), Action.WAIT)
        self.assertEqual(active._next, 1)

    def test_notebook_scope_inlined_boss_dispatches_after_twelve_seconds(self):
        root = Path(__file__).resolve().parents[1]
        # Assemble the package's runtime in memory; do not freeze a submission.
        chunks = [
            (root / "agents/vision_v4_backend.py").read_text(),
            ("\nStructuredAgent = Agent\nBackendPlannerAdapter = PlannerAdapter\n"
            "P3_REFERENCE_STATE_KEY = reference_state_key\n"),
        ]
        for module in MODULES + ("boss_v4",):
            if module == "vision_v4_tracking":
                chunks.append("\nperception_policy = current_policy\n")
            chunks.append(inline((root / "agents" / (module + ".py")).read_bytes()))
        source = "\n".join(chunks)
        cell = "def cell(force_reference):\n" + "\n".join(
            "    " + line for line in source.splitlines()
        )
        cell += (
            "\n    if force_reference:\n"
            "        def encode(*args):\n"
            "            raise UnsupportedState('test reference fallback')\n"
            "    return BossAgent\n"
        )
        namespace = {}
        exec(cell, namespace)  # noqa: S102 -- exercise pasted notebook scope
        with patch("time.perf_counter", side_effect=lambda: self.now):
            for reference in (False, True):
                with self.subTest(reference=reference):
                    self.now = self.started
                    agent = namespace["cell"](reference)()
                    state = corridor()
                    self.assertEqual(agent.step(state), Action.RIGHT)
                    if reference:
                        self.assertIsNotNone(agent._structured._reference)
                    self.now = self.started + 13
                    self.assertEqual(
                        agent.step(step(state, Action.RIGHT)), Action.RIGHT
                    )


if __name__ == "__main__":
    unittest.main()
