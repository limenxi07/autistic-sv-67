"""Use a controlled clock to check diagnostic exits before hard timeout."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmark_planner_p1 import load_agent
from tests.vision_v3_fixtures import image_env
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

folder = Path(__file__).resolve().parent
rows = []
for scenario in ("before_step", "inside_stage"):
    cls = load_agent(str(folder / "task2_agent_debug_only.py"))
    ns = cls.step.__globals__
    real = ns["perf_counter"]
    offset = [0.0]
    ns["perf_counter"] = lambda: real() + offset[0]
    env = image_env(PUBLIC_GAMEPLAY_LEVEL_BUILDERS["A0 Basic Movement"](), seed=42)
    obs, _ = env.reset()
    agent = cls()
    try:
        if scenario == "before_step":
            offset[0] = 2.6
            agent.step(obs)
        else:
            ns["_episode_probe"]["current"] = agent._episode_data
            agent._episode_data["step_started"] = ns["perf_counter"]()
            def slow_stage():
                offset[0] = 2.6
            ns["_episode_wrap"](slow_stage, "controlled_slow_stage")()
    except ns["V4EpisodeDiagnostic"] as exc:
        report = json.loads(str(exc).split(" ", 1)[1])
        expected = "before_step_deadline" if scenario == "before_step" else "checkpoint:controlled_slow_stage"
        assert report["trigger"] == expected, report
        assert 2.5 <= report["elapsed"] < 3, report
        rows.append(report)
    else:
        raise AssertionError("Guard did not report")
(folder / "guard-verification.json").write_text(json.dumps(rows, indent=2) + "\n")
print("Both controlled-clock early-report guards passed")
