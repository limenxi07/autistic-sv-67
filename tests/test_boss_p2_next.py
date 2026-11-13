import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from grid_adventure.entities import CoinEntity, GemEntity, KeyEntity, LockedDoorEntity
from grid_adventure.grid import step
from grid_adventure.step import Action

from agents import boss_p2_next as candidate
from tests.boss_p2_next_fixtures import boss_like, semantic_projection
from tests.planner_p1_fixtures import corridor, layout
from tests.test_planner_p1_state import player
from tools.differential_compact_p2 import project, project_fast


def many_items():
    s = layout(width=15)
    for n, cls in enumerate(
        (
            CoinEntity,
            GemEntity,
            CoinEntity,
            KeyEntity,
            GemEntity,
            CoinEntity,
            CoinEntity,
        ),
        0,
    ):
        e = cls()
        e.entity_id = 10**12 + n * 32 + 5
        s.add((n, 1), e)
    s.add((8, 1), LockedDoorEntity())
    actions = [Action.PICK_UP]
    for _ in range(6):
        actions.extend((Action.RIGHT, Action.PICK_UP))
    actions.extend((Action.RIGHT, Action.USE_KEY))
    return s, tuple(actions)


class OneKeyTests(unittest.TestCase):
    def compare(self, state, actions):
        board, current, score = candidate.encode(state)
        for action in (None,) + actions:
            if action is not None:
                current, delta = candidate.fast_step(board, current, action)
                score += delta
                state = step(state, action)
            self.assertEqual(
                semantic_projection(project(state), board.keys.bit_count()),
                semantic_projection(
                    project_fast(board, current, score), board.keys.bit_count()
                ),
            )
            self.assertEqual(candidate.observe_key(board, state), (current, score))
            for alternative in Action:
                nxt, delta = candidate.fast_step(board, current, alternative)
                self.assertEqual(
                    semantic_projection(
                        project(step(state, alternative)), board.keys.bit_count()
                    ),
                    semantic_projection(
                        project_fast(board, nxt, score + delta), board.keys.bit_count()
                    ),
                    str(alternative),
                )

    def test_many_items_one_key_all_actions(self):
        self.compare(*many_items())

    def test_boxes_cannot_be_pushed_onto_a_collidable_hazard(self):
        from grid_adventure.entities import BoxEntity, LavaEntity

        state = layout(width=8)
        state.add((1, 1), BoxEntity())
        state.add((2, 1), LavaEntity())
        self.compare(state, (Action.RIGHT, Action.WAIT))

    def test_phasing_still_pushes_boxes_and_can_overlap_blocked_boxes(self):
        from grid_adventure.entities import (
            BoxEntity,
            LavaEntity,
            PhasingPowerUpEntity,
            WallEntity,
        )

        for cls in (
            None,
            WallEntity,
            LockedDoorEntity,
            LavaEntity,
            BoxEntity,
            CoinEntity,
        ):
            state = layout(width=8)
            state.add((0, 1), PhasingPowerUpEntity())
            state = step(state, Action.PICK_UP)
            state.add((1, 1), BoxEntity())
            if cls:
                state.add((2, 1), cls())
            self.compare(
                state, (Action.RIGHT, Action.WAIT, Action.RIGHT) + (Action.WAIT,) * 5
            )

    def test_permuting_nonkey_and_single_key_inventory_preserves_successors(self):
        state, actions = many_items()
        for action in actions[:-1]:
            state = step(state, action)
        other = deepcopy(state)
        player(other).inventory_list.reverse()
        b, _, _ = candidate.encode(state)
        self.assertEqual(
            candidate.observe_key(b, state), candidate.observe_key(b, other)
        )
        for action in Action:
            self.assertEqual(
                semantic_projection(project(step(state, action)), 1),
                semantic_projection(project(step(other, action)), 1),
            )
        consumed = step(state, Action.USE_KEY)
        self.assertFalse(
            any(getattr(e, "key", None) for e in player(consumed).inventory_list)
        )

    def test_many_keys_still_reject(self):
        state, _ = many_items()
        state.add((10, 1), KeyEntity())
        with self.assertRaisesRegex(candidate.UnsupportedState, "Large keyed"):
            candidate.encode(state)

    def test_boss_like_witnesses_and_all_action_effect_frames(self):
        for i in range(6):
            s, actions, _ = boss_like(i)
            self.compare(s, actions)
            for action in actions:
                s = step(s, action)
            self.assertTrue(s.win)
            self.assertFalse(s.lose)


class DeadlineTests(unittest.TestCase):
    def adapter(self, **kwargs):
        return candidate.PlannerAdapter(
            SimpleNamespace(started_at=10.0, initial_deadline=13.0, cutoff=13.5),
            **kwargs,
        )

    def test_absolute_origin_and_expired_cached_dispatch(self):
        agent = self.adapter()
        self.assertEqual(
            (agent._constructed, agent._initial_deadline, agent._cutoff), (10, 13, 13.5)
        )
        state = corridor()
        agent._board, current, score = candidate.encode(state)
        after, delta = candidate.fast_step(agent._board, current, Action.RIGHT)
        agent._actions = (Action.RIGHT, Action.RIGHT)
        agent._keys = ((current, score), (after, score + delta))
        agent._next = 1
        with patch.object(candidate, "perf_counter", return_value=14):
            self.assertEqual(agent.step(step(state, Action.RIGHT)), Action.WAIT)
        self.assertEqual(agent._stats["search_calls"], 0)

    def test_late_reference_constructor_inherits_absolute_budget(self):
        agent = self.adapter(reference_initial_deadline=12.0)
        state = corridor()
        state.step_cost = 4
        with (
            patch.object(candidate, "perf_counter", return_value=11),
            patch.object(candidate.ReferenceAgent, "step", return_value=Action.WAIT),
        ):
            agent.step(state)
        self.assertEqual(agent._reference._initial_deadline, 12)
        self.assertEqual(agent._reference._cutoff, 13.5)

    def test_task3_is_explicit_and_default_stays_short(self):
        with patch.object(candidate, "perf_counter", return_value=100):
            normal, boss = candidate.Agent(), candidate.Task3Agent()
        self.assertEqual((normal._initial_deadline, normal._cutoff), (103, 103.5))
        self.assertEqual((boss._initial_deadline, boss._cutoff), (111, 112))

    def test_budget_only_tightens(self):
        agent = self.adapter()
        agent.budget.initial_deadline, agent.budget.cutoff = 50, 60
        with patch.object(candidate, "perf_counter", return_value=14):
            agent.step(corridor())
        self.assertEqual((agent._initial_deadline, agent._cutoff), (13, 13.5))

    def test_failed_or_expired_rebinding_does_not_install_partial_keys(self):
        from time import perf_counter

        now = perf_counter()
        agent = candidate.PlannerAdapter(candidate.EpisodeBudget(now, now + 1, now + 2))
        state = corridor()
        board, current, score = candidate.encode(state)
        agent._board, agent._actions, agent._keys = (
            board,
            (Action.RIGHT, Action.RIGHT),
            ("old",),
        )
        before = (agent._board, agent._actions, agent._keys, agent._next)
        for status in ("invalid", "deadline"):
            with patch.object(
                candidate,
                "verify_plan",
                return_value=candidate.Verification(status, None, ()),
            ):
                agent._rebind_suffix(state)
            self.assertEqual(
                (agent._board, agent._actions, agent._keys, agent._next), before
            )

    def test_fast_cached_dispatch_checks_time_after_observation(self):
        agent = self.adapter()
        state = corridor()
        agent._board, current, score = candidate.encode(state)
        agent._actions, agent._keys = (Action.RIGHT,), ((current, score),)
        clock = [13.0]
        calls = [0]
        original = candidate.observe_key

        def observe(*args):
            calls[0] += 1
            value = original(*args)
            if calls[0] == 2:
                clock[0] = 13.6
            return value

        with (
            patch.object(candidate, "perf_counter", side_effect=lambda: clock[0]),
            patch.object(candidate, "observe_key", side_effect=observe),
        ):
            action = agent.step(state)
        self.assertEqual(action, Action.WAIT)
        self.assertEqual(agent._next, 0)

    def test_reference_cached_dispatch_checks_time_after_observation(self):
        agent = self.adapter()
        state = corridor()
        backend = agent._reference = candidate.ReferenceAgent()
        backend._cutoff = 13.5
        backend._actions, backend._keys = (
            (Action.RIGHT,),
            (candidate.reference_state_key(state),),
        )
        clock, calls = [13.0], [0]
        original = candidate.reference_state_key

        def observe(*args):
            calls[0] += 1
            value = original(*args)
            if calls[0] == 2:
                clock[0] = 13.6
            return value

        with (
            patch.object(candidate, "perf_counter", side_effect=lambda: clock[0]),
            patch.object(candidate, "reference_state_key", side_effect=observe),
        ):
            action = agent.step(state)
        self.assertEqual(action, Action.WAIT)
        self.assertEqual(backend._next, 0)

    def test_search_completion_after_cutoff_does_not_dispatch(self):
        from time import perf_counter

        state = corridor()
        result = candidate.reference_search(
            state, deadline=perf_counter() + 1, mode="bfs"
        )
        for fast in (True, False):
            agent = self.adapter()
            clock = [12.9]

            def search(*args, **kwargs):
                clock[0] = 13.6
                if fast:
                    return candidate.FastResult(
                        result.actions,
                        result.expected_keys,
                        True,
                        result.final_score,
                        "deadline",
                        1,
                        2,
                        0.1,
                        2,
                        3,
                    )
                return result

            if not fast:
                agent._reference = candidate.ReferenceAgent()
                agent._reference._initial_deadline = 13
            name = "search" if fast else "reference_search"
            with (
                patch.object(candidate, "perf_counter", side_effect=lambda: clock[0]),
                patch.object(candidate, name, side_effect=search),
            ):
                action = agent.step(state)
            self.assertEqual(action, Action.WAIT, "fast=" + str(fast))

    def test_late_fallback_uses_remaining_original_cutoff_immediately(self):
        agent = self.adapter()
        agent._tried = True
        state = corridor()
        state.step_cost = 4
        result = candidate.reference_SearchResult(
            (), (), False, None, "deadline", 0, 0, 1, 1, None, None, None
        )
        with (
            patch.object(candidate, "perf_counter", return_value=13.1),
            patch.object(candidate, "reference_search", return_value=result) as search,
        ):
            agent.step(state)
        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["deadline"], 13.5)


class CandidateBridgeTests(unittest.TestCase):
    def exercise(self, name):
        from tests import test_vision_v3_harness

        def planner(budget):
            return candidate.PlannerAdapter(
                budget,
                reference_initial_deadline=min(
                    budget.initial_deadline, budget.started_at + 2.0
                ),
            )

        with patch.object(
            test_vision_v3_harness, "PlannerAdapter", side_effect=planner
        ):
            getattr(test_vision_v3_harness.HarnessTests(name), name)()

    def test_fast_suffix_survives_public_id_changes(self):
        self.exercise("test_public_pickup_reidentification_retains_verified_suffix")

    def test_reference_suffix_survives_public_id_changes(self):
        self.exercise("test_reference_suffix_survives_public_resource_reidentification")

    def test_original_deadline_is_shared(self):
        self.exercise("test_planner_never_resets_original_deadline")


class BossOrderingTests(unittest.TestCase):
    def test_key_branch_is_not_starved_by_an_uncollected_ghost(self):
        from public_levels.catalog import PUBLIC_BOSS_LEVEL_BUILDERS

        state = list(PUBLIC_BOSS_LEVEL_BUILDERS.values())[1]()
        board, start, score = candidate.encode(state)
        calls = [0]
        original = candidate.fast_step

        def transition(*args):
            calls[0] += 1
            return original(*args)

        clock = lambda: calls[0]
        with patch.object(candidate, "fast_step", side_effect=transition):
            result = candidate.search(
                board,
                start,
                score,
                deadline=50000,
                mode="p2a",
                clock=clock,
                verify_candidate=lambda a, k: candidate.verify_plan(
                    board, state, a, k, deadline=50000, clock=clock
                ),
            )
        self.assertTrue(result.winning, result.stop_reason)
        for action in result.actions:
            state = step(state, action)
        self.assertTrue(state.win)
        self.assertFalse(state.lose)
        self.assertEqual(state.score, result.final_score)


if __name__ == "__main__":
    unittest.main()
