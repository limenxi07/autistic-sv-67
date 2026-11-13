"""Validate the end probe and independently execute its reported final action."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmark_planner_p1 import load_agent
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tests.vision_v3_fixtures import image_env
from grid_adventure.step import Action

folder = Path(__file__).resolve().parent
source = folder / "task2_end_debug_only.py"
base = (ROOT / "submissions/task2_agent.py").read_bytes()
assert source.read_bytes() == base + (folder / "end_probe_suffix.py").read_bytes()
rows = []
for notebook in (False, True):
    if notebook:
        ns = {}
        exec("def cell():\n" + "".join("    " + line + "\n" for line in source.read_text().splitlines()) + "    return Agent\n", ns)
        cls = ns["cell"]()
    else:
        cls = load_agent(str(source))
    for label, builder in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
        state = builder()
        state.turn_limit = 150
        env = image_env(state, seed=42)
        obs, _ = env.reset()
        agent = cls()
        try:
            for _ in range(150):
                action = agent.step(obs)
                obs, _, win, lose, _ = env.step(action)
                assert not (win or lose), "Missing report before terminal action"
        except Exception as exc:
            assert str(exc).startswith("TASK2_END_0b61 "), repr(exc)
            report = json.loads(str(exc).split(" ", 1)[1])
            assert report["source"] == hashlib.sha256(base).hexdigest()
            assert report["trigger"] == "predicted_terminal_action", report
            assert report["plan_remaining"] == 0, report
            action = next(a for a in Action if str(a) == report["tail"][-1]["action"])
            _, _, win, lose, _ = env.step(action)
            assert win and not lose, report
            assert env.state.score == report["predicted_status"][0]["score"]
            rows.append(dict(notebook=notebook, label=label, report=report,
                actual_terminal_win=bool(win), actual_score=env.state.score))
        else:
            raise AssertionError("Missing end report")

guards = []
for offset in (4.70, 4.80):
    cls = load_agent(str(source))
    ns = cls.step.__globals__
    real = ns["perf_counter"]
    extra = [0.0]
    ns["perf_counter"] = lambda: real() + extra[0]
    obs, _ = image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS["A0 Basic Movement"](), seed=42).reset()
    agent = cls()
    extra[0] = offset
    try:
        agent.step(obs)
    except ns["Task2EndDiagnostic"] as exc:
        report = json.loads(str(exc).split(" ", 1)[1])
        assert report["trigger"] == ("first_fallback" if offset < 4.75 else "near_hard_limit_before_step")
        if offset < 4.75:
            assert report["stats"]["reason"] == "cutoff"
        guards.append(report)
    else:
        raise AssertionError("Missing guard report")
(folder / "end-probe-verification.json").write_text(json.dumps(dict(cases=rows, controlled_clock_guards=guards), indent=2) + "\n")
print(f"{len(rows)}/44 reports verified with actual terminal wins; {len(guards)}/2 deadline guards passed")
