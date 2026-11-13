"""Reproduce a consistent observation being labelled unresolved at expiry."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmark_planner_p1 import load_agent
from tests.vision_v3_fixtures import image_env
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

cls = load_agent(str(ROOT / "submissions/task2_agent.py"))
state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS["B4 Measure Twice and Cut Once"]()
state.turn_limit = 150
env = image_env(state, seed=42)
obs, _ = env.reset()
agent = cls()
for _ in range(95):
    action = agent.step(obs)
    obs, _, win, lose, _ = env.step(action)
    assert not win and not lose
remaining_before = len(agent._planner._actions) - agent._planner._next
ns = cls.step.__globals__
original_check = ns["check_observation"]
now = [agent._budget.cutoff - .01]
ns["perf_counter"] = lambda: now[0]
checks = []


def crossing_deadline(*args, **kwargs):
    result = original_check(*args, **kwargs)
    checks.append(dict(consistent=result.consistent, mismatches=result.mismatches))
    # Model crossing the deadline during the existing observation check.
    now[0] = agent._budget.cutoff + .03
    return result


ns["check_observation"] = crossing_deadline
action = agent.step(obs)
assert str(action) == "wait"
assert checks == [dict(consistent=True, mismatches=())], checks
assert agent.info()["reason"] == "unresolved_perception"
assert not agent._hypotheses and agent._planner is None
assert agent._stats["visual_discrepancies"] == agent._stats["metadata_discrepancies"] == 0
report = dict(diagnostic_only=True, controlled_clock=True, observed_turn=obs["info"]["status"]["turn"],
    remaining_actions_before=remaining_before, elapsed=now[0] - agent._budget.started_at,
    action=str(action), check=checks[0], stats=agent._stats,
    hypothesis_count=len(agent._hypotheses), planner_cleared=agent._planner is None)
(Path(__file__).parent / "expiry-label-reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
