import unittest
from copy import deepcopy
from dataclasses import replace

from grid_adventure.grid import step
from grid_adventure.step import Action

from agents.planner_p1 import state_key
from tests.planner_p1_fixtures import corridor, fixture_cases, mechanic


def traces():
    for builder, witness in fixture_cases("mechanics").values():
        state = builder()
        yield state
        for action in witness:
            state = step(state, action)
            yield state


def player(state):
    return next(
        e
        for column in state.grid
        for cell in column
        for e in cell
        if getattr(e, "agent", None) is not None
    )


class StateTests(unittest.TestCase):
    def test_copy_has_same_key(self):
        for state in traces():
            self.assertEqual(state_key(state), state_key(deepcopy(state)))
            hash(state_key(state))

    def test_each_dynamic_component_changes_key(self):
        for field, value in [
            ("turn", 4),
            ("turn_limit", 4),
            ("score", -1),
            ("win", True),
            ("lose", True),
            ("step_cost", 4),
            ("seed", 2),
        ]:
            a = corridor()
            b = deepcopy(a)
            setattr(b, field, value)
            self.assertNotEqual(state_key(a), state_key(b), field)
        a = corridor()
        b = deepcopy(a)
        player(b).health = replace(player(b).health, current_health=4)
        self.assertNotEqual(state_key(a), state_key(b))
        b = deepcopy(a)
        player(b).health = replace(player(b).health, max_health=6)
        self.assertNotEqual(state_key(a), state_key(b))
        for kind, list_name in [
            ("key_door", "inventory_list"),
            ("boots", "status_list"),
            ("ghost", "status_list"),
            ("shield", "status_list"),
        ]:
            a, _ = mechanic(kind)
            a = step(step(a, Action.RIGHT), Action.PICK_UP)
            b = deepcopy(a)
            getattr(player(b), list_name).clear()
            self.assertNotEqual(state_key(a), state_key(b), kind)
            if list_name == "status_list":
                b = deepcopy(a)
                effect = player(b).status_list[0]
                counter = getattr(effect, "time_limit", None) or effect.usage_limit
                name = (
                    "time_limit"
                    if getattr(effect, "time_limit", None)
                    else "usage_limit"
                )
                setattr(effect, name, replace(counter, amount=counter.amount - 1))
                self.assertNotEqual(state_key(a), state_key(b), kind + " timer")
        for kind in ("coin", "gem", "key_door", "box_coin", "damage"):
            a, _ = mechanic(kind)
            b = step(a, Action.RIGHT)
            # Remove timing/score differences: board/resources alone still differ.
            b.turn = a.turn
            b.score = a.score
            self.assertNotEqual(state_key(a), state_key(b), kind)

    def test_overlapping_entities_are_retained(self):
        from grid_adventure.entities import BoxEntity, CoinEntity

        a = corridor()
        a.add((1, 0), BoxEntity())
        b = deepcopy(a)
        b.add((1, 0), CoinEntity())
        self.assertNotEqual(state_key(a), state_key(b))
        b.grid[1][0].reverse()
        c = deepcopy(b)
        c.grid[1][0].reverse()
        self.assertEqual(state_key(b), state_key(c))

    def test_equal_keys_have_equal_successors(self):
        for state in traces():
            copy = deepcopy(state)
            self.assertEqual(state_key(state), state_key(copy))
            for action in Action:
                a, b = step(state, action), step(copy, action)
                self.assertEqual(state_key(a), state_key(b))
                self.assertEqual(a.score - state.score, b.score - copy.score)
                self.assertEqual((a.win, a.lose), (b.win, b.lose))

    def test_mechanic_witnesses_and_active_schema(self):
        for builder, witness in fixture_cases("mechanics").values():
            state = builder()
            for action in witness:
                state = step(state, action)
            self.assertTrue(state.win)
        for kind in ("boots", "ghost"):
            s, _ = mechanic(kind)
            s = step(step(s, Action.RIGHT), Action.PICK_UP)
            self.assertEqual(player(s).status_list[0].time_limit.amount, 5)
            s = step(s, Action.WAIT)
            self.assertEqual(player(s).status_list[0].time_limit.amount, 4)
            for _ in range(4):
                s = step(s, Action.WAIT)
            self.assertEqual(player(s).status_list, [])
        s, actions = mechanic("shield")
        for a in actions[:7]:
            s = step(s, a)
        self.assertEqual(player(s).status_list, [])
        self.assertEqual(player(s).health.current_health, 5)
        s, actions = mechanic("key_door")
        for a in actions[:4]:
            s = step(s, a)
        self.assertEqual(player(s).inventory_list, [])
        self.assertFalse(any(getattr(e, "locked", None) for e in s.objects_at((3, 1))))

    def test_stacked_shield_reference_identity_is_preserved(self):
        from grid_adventure.entities import ShieldPowerUpEntity

        a, _ = mechanic("shield")
        a.add((2, 1), ShieldPowerUpEntity())
        for action in (Action.RIGHT, Action.PICK_UP, Action.RIGHT, Action.PICK_UP):
            a = step(a, action)
        effects = player(a).status_list
        effects[0].usage_limit = replace(effects[0].usage_limit, amount=2)
        effects[1].usage_limit = replace(effects[1].usage_limit, amount=3)
        b = deepcopy(a)
        other = player(b).status_list
        other[0].usage_limit, other[1].usage_limit = (
            other[1].usage_limit,
            other[0].usage_limit,
        )
        self.assertNotEqual(state_key(a), state_key(b))
        self.assertNotEqual(
            state_key(step(a, Action.RIGHT)), state_key(step(b, Action.RIGHT))
        )

    def test_public_roundtrip_keeps_reference_keys(self):
        from grid_adventure.grid import from_state, to_state

        for state in traces():
            self.assertEqual(state_key(state), state_key(from_state(to_state(state))))
