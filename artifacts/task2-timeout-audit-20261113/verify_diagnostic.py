"""Check intentional reports in module and notebook-function contexts."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.benchmark_planner_p1 import load_agent
from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
from tests.vision_v3_fixtures import image_env

folder = Path(__file__).resolve().parent
source = folder / "task2_agent_debug_only.py"
base = (ROOT / "submissions/task2_agent.py").read_bytes()
binding = json.loads((folder / "binding.json").read_text())
assert hashlib.sha256(base).hexdigest() == binding["base_sha256"]
assert source.read_bytes() == base + (folder / "diagnostic_suffix.py").read_bytes()
rows = []
for notebook in (False, True):
    if notebook:
        namespace = {}
        exec("def cell():\n" + "".join("    " + line + "\n" for line in source.read_text().splitlines()) + "    return Agent\n", namespace)
        cls = namespace["cell"]()
    else:
        cls = load_agent(str(source))
    for label, build in PUBLIC_GAMEPLAY_LEVEL_BUILDERS.items():
        state = build()
        state.turn_limit = 150
        env = image_env(state, seed=42)
        obs, _ = env.reset()
        agent = cls()
        try:
            for turn in range(150):
                action = agent.step(obs)
                obs, _, win, lose, _ = env.step(action)
                assert not (win or lose), "Diagnostic must report before episode ends"
        except Exception as exc:
            marker = binding["marker"] + " "
            assert str(exc).startswith(marker), repr(exc)
            report = json.loads(str(exc)[len(marker):])
            assert report["source"] == binding["base_sha256"]
            assert 1 <= report["sampled_actions"] <= 40
            assert report["steps_first_last"][0]["turn"] == 0
            assert report["config"]["width"] == state.width
            assert report["config"]["height"] == state.height
            assert report["image_shape"]
            assert report["board_signature"]
            assert report["stages_inclusive"]["cnn_total"]["calls"] >= 1
            assert report["stages_inclusive"]["winning_replay"]["calls"] >= 1
            assert report["transition_cache_hits"] > 0
            assert report["errors"] == []
            rows.append(dict(notebook=notebook, label=label, report=report, error_length=len(str(exc))))
        else:
            raise AssertionError("Missing diagnostic error")
(folder / "diagnostic-verification.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(dict(verified=len(rows), contexts=["module", "notebook function"],
    max_error_chars=max(row["error_length"] for row in rows),
    triggers=sorted({row["report"]["trigger"] for row in rows})), indent=2))
