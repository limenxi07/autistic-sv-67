"""Evaluate image agents in fresh workers; keep oracle inputs diagnostic-only."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def power_state():
    if not hasattr(os, "sched_getaffinity"):
        return None
    cpu = min(os.sched_getaffinity(0))
    root = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    result = {"cpu": cpu}
    for key, filename in (
        ("epp", "energy_performance_preference"),
        ("governor", "scaling_governor"),
        ("max_freq", "scaling_max_freq"),
    ):
        path = root / filename
        result[key] = path.read_text().strip() if path.is_file() else None
    return result


def oracle_labels(truth):
    """Evaluator-only complete semantic layers, with no IDs/components attached."""
    labels = []
    for y in range(truth.height):
        row = []
        for x in range(truth.width):
            names = []
            for entity in truth.grid[x][y]:
                name = entity.appearance.name
                if name == "door":
                    name = (
                        "locked"
                        if getattr(entity, "locked", None) is not None
                        else "opened"
                    )
                names.append(name)
            row.append(tuple(names) if names else ("floor",))
        labels.append(row)
    return labels


def semantic_labels(state):
    """Erase only duplicate inert walls; preserve every other multiplicity."""
    return [
        [
            tuple(sorted(set(c))) if c and set(c) == {"wall"} else tuple(sorted(c))
            for c in row
        ]
        for row in oracle_labels(state)
    ]


def recorded_appearance(env, options):
    """Evaluator-only renderer reproducing the frozen corpus's texture choices."""
    import inspect
    import io
    import random

    import numpy as np
    from grid_adventure.env import GridAdventureEnv
    from grid_universe.renderer.image import ImageRenderer
    from PIL import Image

    from agents.vision_v3_bridge import CLASSES

    manifest = json.loads(Path(options["appearance_manifest"]).read_text())
    if options.get("appearance_manifest_sha256"):
        actual = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if actual != options["appearance_manifest_sha256"]:
            raise ValueError("Appearance manifest differs from frozen job")
    rng = random.Random(options["appearance_seed"])
    textures = {}
    for kind in CLASSES:
        entries = [
            a
            for a in manifest["assets"]
            if a["class"] == kind and a["split"] == options["appearance_split"]
        ]
        entry = rng.choice(entries)
        path = Path(options["asset_root"]) / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Appearance asset hash mismatch")
        with Image.open(path) as image:
            textures[kind] = image.convert("RGBA")

    def lookup(obj, size):
        kind = obj.appearance.name
        if kind == "door":
            kind = "locked" if "locked" in obj.properties else "opened"
        return textures[kind].resize((size, size), Image.Resampling.BILINEAR)

    original = (
        inspect.signature(GridAdventureEnv).parameters["render_image_map"].default
    )
    renderer = ImageRenderer(
        resolution=options["resolution"],
        asset_root=options["asset_root"],
        image_map=type(original)({k: v for k, v in original.items() if v in CLASSES}),
        tex_lookup_fn=lookup,
    )

    class RenderedEnv:
        @property
        def state(self):
            return env.state

        def render_observation(self, observation, *, initial=False):
            rendered = renderer.render(env.state)
            if initial and options.get("expected_pixels_sha256"):
                if (
                    hashlib.sha256(np.asarray(rendered).tobytes()).hexdigest()
                    != options["expected_pixels_sha256"]
                ):
                    raise ValueError(
                        "Initial decoded pixels differ from the frozen corpus"
                    )
            elif initial and options.get("expected_image_sha256"):
                buffer = io.BytesIO()
                rendered.save(buffer, format="PNG")
                if (
                    hashlib.sha256(buffer.getvalue()).hexdigest()
                    != options["expected_image_sha256"]
                ):
                    raise ValueError("Initial image differs from the frozen corpus")
            observation["image"] = np.asarray(rendered)
            return observation

        def reset(self):
            observation, info = env.reset()
            return self.render_observation(observation, initial=True), info

        def step(self, action):
            observation, reward, win, lose, info = env.step(action)
            # Dead artwork is outside the 14 initial classes. A terminal loss
            # already has an authoritative result and needs no next image.
            if not lose:
                observation = self.render_observation(observation)
            return observation, reward, win, lose, info

    return RenderedEnv()


def evaluate_episode(env, agent, *, observation, evaluation_limit=5.0, optimal=None):
    from grid_adventure.grid import from_state, step

    if observation not in ("image", "oracle", "gridstate"):
        raise ValueError("Unknown observation mode")
    began = perf_counter()
    obs, _ = env.reset()
    total = 0
    win = lose = False
    predicted = None
    unresolved = []
    turns = 0
    initial_board = None
    initial_confusion = []

    def budget_of_agent():
        return getattr(agent, "_budget", getattr(agent, "budget", None))

    initial_budget = budget_of_agent()
    original_budget = tuple(initial_budget) if initial_budget is not None else None
    budget_reset = False
    while not (win or lose):
        if perf_counter() - began >= evaluation_limit:
            break
        if observation == "oracle":
            from agents.vision_v3_bridge import reconstruct

            # Ground truth never escapes this diagnostic branch as an agent API.
            truth = from_state(env.state)
            hypothesis = reconstruct(oracle_labels(truth), obs["info"], predicted)
            if hypothesis.unresolved:
                unresolved.extend(hypothesis.unresolved)
                break
            action = agent.step(hypothesis.state)
            predicted = step(hypothesis.state, action)
        else:
            action = agent.step(obs)
            if turns == 0 and observation == "image":
                from collections import Counter

                history = getattr(agent, "_history", None)
                truth = semantic_labels(from_state(env.state))
                parsed = (
                    semantic_labels(history.hypotheses[0].state)
                    if history and history.hypotheses
                    else None
                )
                initial_board = parsed == truth
                counts = Counter(
                    (
                        "+".join(cell),
                        "+".join(parsed[y][x]) if parsed is not None else "unresolved",
                    )
                    for y, row in enumerate(truth)
                    for x, cell in enumerate(row)
                )
                initial_confusion = [
                    [actual, predicted, count]
                    for (actual, predicted), count in sorted(counts.items())
                ]
        if original_budget is not None and tuple(budget_of_agent()) != original_budget:
            budget_reset = True
        obs, reward, win, lose, _ = env.step(action)
        total += reward
        turns += 1
    elapsed = perf_counter() - began
    status = "timeout" if elapsed >= evaluation_limit else "win" if win else "loss"
    if status != "win":
        effective = 0.0
    elif optimal is None:
        effective = None
    else:
        effective = max(0, total - optimal * 1.5) / (optimal - optimal * 1.5)
    return {
        "status": status,
        "win": bool(win),
        "lose": bool(lose),
        "reward": total,
        "evaluation_seconds": elapsed,
        "turns": turns,
        "effective_performance": effective,
        "unresolved": unresolved,
        "observation": observation,
        "initial_board_exact": initial_board,
        "initial_confusion": initial_confusion,
        "agent_info": agent.info() if hasattr(agent, "info") else {},
        "budget_reset": budget_reset,
        "budget_seconds": [
            original_budget[1] - original_budget[0],
            original_budget[2] - original_budget[0],
        ]
        if original_budget is not None
        else None,
    }


def _worker(connection, label, seed, observation, source, options):
    began = perf_counter()
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        from grid_adventure.env import GridAdventureEnv
        from grid_adventure.grid import to_state

        from public_levels.catalog import (
            PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
            PUBLIC_LEVEL_MAX_REWARDS,
        )
        from tools.benchmark_planner_p1 import load_agent, provenance

        def build():
            if options.get("p2_case") is not None:
                from tests.compact_p2_fixtures import heldout_case

                return heldout_case(options["p2_case"])[0]
            if options.get("generated") is not None:
                from tools.vision_v3_data import new_layout

                return new_layout(
                    options["generated"],
                    split=options.get("layout_split", "validation"),
                    multiple_gems=options.get("multiple_gems", False),
                )[0]
            return PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]()

        def initial(*args, **kwargs):
            state = build()
            state.seed, state.turn_limit = seed, 150
            return to_state(state)

        sample = build()
        if options.get("expected_layout_sha256"):
            from tools.vision_v3_data import canonical, digest

            if (
                digest(
                    canonical(
                        {
                            "width": sample.width,
                            "height": sample.height,
                            "labels": oracle_labels(sample),
                        }
                    )
                )
                != options["expected_layout_sha256"]
            ):
                raise ValueError("Generated layout differs from its frozen manifest")
        render_options = {}
        if options.get("asset_root"):
            import inspect

            from agents.vision_v3_bridge import CLASSES

            original = (
                inspect.signature(GridAdventureEnv)
                .parameters["render_image_map"]
                .default
            )
            render_options = {
                "render_asset_root": options["asset_root"],
                "render_image_map": type(original)(
                    {k: v for k, v in original.items() if v in CLASSES}
                ),
            }
        env = GridAdventureEnv(
            initial_state_fn=initial,
            width=sample.width,
            height=sample.height,
            observation_type="gridstate" if observation == "gridstate" else "image",
            **render_options,
        )
        if options.get("appearance_manifest"):
            env = recorded_appearance(env, options)
        cls = load_agent(source) if source else None
        if cls is None and observation != "oracle":
            raise ValueError("Image/gridstate mode requires an explicit agent source")
        setup = perf_counter() - began
        metadata = provenance()
        power_before = power_state()
        connection.send(("started", setup))
        if cls is None:
            from agents.vision_v3_planner import PlannerAdapter
            from agents.vision_v3_types import EpisodeBudget

            started = perf_counter()
            agent: Any = PlannerAdapter(
                EpisodeBudget(started, started + 3, started + 3.5)
            )
        else:
            if observation == "oracle" and hasattr(cls, "_step_image"):
                entry: Any = cls._step_image
                namespace = entry.__globals__
                started = perf_counter()
                budget = namespace["EpisodeBudget"](started, started + 3, started + 3.5)
                agent = namespace["PlannerAdapter"](budget)
            else:
                agent = cls()
        if options.get("no_tracker"):
            from agents.vision_v3_types import ObservationCheck

            agent._step_image.__globals__["check_observation"] = (
                lambda *args, **kwargs: ObservationCheck(True, (), ())
            )
        if options.get("planner_mode", "p2a") != "p2a":
            original_adapter = agent._step_image.__globals__["PlannerAdapter"]
            agent._step_image.__globals__["PlannerAdapter"] = lambda budget: (
                original_adapter(budget, mode=options["planner_mode"])
            )
        result = evaluate_episode(
            env,
            agent,
            observation=observation,
            optimal=PUBLIC_LEVEL_MAX_REWARDS.get(label),
        )
        power_after = power_state()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result.update(
            label=label,
            seed=seed,
            setup_seconds=setup,
            cold_import_to_terminal_seconds=perf_counter() - began,
            peak_rss_bytes=rss if sys.platform == "darwin" else rss * 1024,
            provenance=metadata,
            runtime_overrides={
                k: v
                for k, v in options.items()
                if k in ("no_tracker", "planner_mode") and v not in (False, "p2a")
            },
            entrypoint="PlannerAdapter" if observation == "oracle" else "Agent",
            power_state_before=power_before,
            power_state_after=power_after,
            source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest()
            if source
            else None,
        )
        if "torch" in sys.modules:
            torch = sys.modules["torch"]
            result["torch_threads"] = [
                torch.get_num_threads(),
                torch.get_num_interop_threads(),
            ]
        connection.send(("result", result))
    except BaseException:  # noqa: BLE001 -- report isolated worker failures to parent
        connection.send(
            (
                "result",
                {
                    "label": label,
                    "seed": seed,
                    "status": "error",
                    "win": False,
                    "effective_performance": 0.0,
                    "error": traceback.format_exc(),
                },
            )
        )
    finally:
        connection.close()


def run_case(
    label, *, seed=2, observation="oracle", source=None, cold_limit=5.0, **options
):
    from tools.benchmark_planner_p1 import THREAD_VARS

    for key in THREAD_VARS:
        os.environ[key] = "1"
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker, args=(sender, label, seed, observation, source, options)
    )
    cold_started = perf_counter()
    process.start()
    sender.close()
    result = {
        "label": label,
        "seed": seed,
        "status": "setup_timeout",
        "win": False,
        "effective_performance": 0.0,
    }
    try:
        if receiver.poll(min(30, cold_limit)):
            kind, value = receiver.recv()
            if kind == "started":
                remaining = max(0, cold_limit - (perf_counter() - cold_started))
                if receiver.poll(min(5, remaining)):
                    _, result = receiver.recv()
                else:
                    result.update(status="cold_timeout", setup_seconds=value)
            else:
                result = value
        else:
            result.update(status="cold_setup_timeout")
        result["strict_cold_seconds"] = perf_counter() - cold_started
        result["cold_watchdog_seconds"] = cold_limit
    except EOFError:
        result.update(status="worker_error")
    finally:
        if process.is_alive():
            process.terminate()
        process.join(2)
        if process.is_alive():
            process.kill()
            process.join()
        receiver.close()
    return result


def run_jobs(jobs, output, *, source, observation="image"):
    """Fresh sequential workers; jobs contain evaluator-only fixture selectors."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    report = {
        "source_sha256": sha,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "observation": observation,
        "results": rows,
        "evidence_boundary": "Local evaluation only; generated optima unknown.",
    }
    for job in jobs:
        spec = dict(job)
        label = spec.pop("label")
        row = run_case(label, source=source, observation=observation, **spec)
        row.update(source_sha256=sha, job=job)
        rows.append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(len(rows), label, row["status"], row.get("reward"), flush=True)
    return report


def benchmark(
    output,
    *,
    observation="oracle",
    source=None,
    seed=2,
    labels=None,
    paired=True,
    **options,
):
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    baseline = "agents/boss_p2_next.py"
    for label in labels or PUBLIC_GAMEPLAY_LEVEL_BUILDERS:
        result = run_case(
            label, seed=seed, observation=observation, source=source, **options
        )
        pair = (
            run_case(
                label,
                seed=seed,
                observation="gridstate",
                source=baseline,
                **{
                    k: v
                    for k, v in options.items()
                    if k in ("generated", "layout_split")
                },
            )
            if paired
            else {}
        )
        result["paired_backend"] = pair
        result["paired_backend_source"] = baseline
        result["paired_reward_match"] = result["status"] == pair.get(
            "status"
        ) == "win" and result.get("reward") == pair.get("reward")
        results.append(result)
        paths = list(Path("agents").glob("vision_v3*.py")) + [
            Path(__file__),
            Path(baseline),
        ]
        if source:
            paths.append(Path(source))
        report = {
            "observation": observation,
            "seed": seed,
            "options": options,
            "results": results,
            "bridge_gate_passed": len(results) == 22
            and all(r["paired_reward_match"] for r in results),
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
            "evidence_boundary": "Local "
            + observation
            + " evaluation; not a graded result. Generated optima unknown.",
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            label,
            result["status"],
            result.get("reward"),
            "paired",
            pair.get("reward"),
            result.get("agent_info", {}).get("reason"),
            flush=True,
        )
    return report


def probe_metadata(output):
    """Archive black-box public observations from deterministic mechanic traces."""
    from grid_adventure.entities import ShieldPowerUpEntity

    from tests.vision_v3_fixtures import image_env
    from tests.planner_p1_fixtures import mechanic
    from tools.benchmark_planner_p1 import provenance

    rows = []
    for kind in (
        "key_door",
        "coin",
        "gem",
        "boots",
        "ghost",
        "shield",
        "box_coin",
        "damage",
    ):
        state, actions = mechanic(kind)
        if kind == "shield":
            state.add((1, 1), ShieldPowerUpEntity())
        env = image_env(state)
        obs, _ = env.reset()
        rows.append(
            {
                "fixture": kind,
                "action": None,
                "info": obs["info"],
                "image_shape": list(obs["image"].shape),
            }
        )
        for action in actions:
            obs, reward, win, lose, _ = env.step(action)
            rows.append(
                {
                    "fixture": kind,
                    "action": action.name,
                    "reward": reward,
                    "win": bool(win),
                    "lose": bool(lose),
                    "info": obs["info"],
                    "image_shape": list(obs["image"].shape),
                }
            )
            if win or lose:
                break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"observations": rows, "provenance": provenance()}, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-only", action="store_true")
    parser.add_argument("--probe-metadata", action="store_true")
    parser.add_argument("--suite", choices=("public",), default="public")
    parser.add_argument(
        "--observation", choices=("image", "oracle", "gridstate"), default="oracle"
    )
    parser.add_argument("--source")
    parser.add_argument("--jobs-json", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--asset-root")
    parser.add_argument("--generated", type=int)
    parser.add_argument("--layout-split", default="validation")
    parser.add_argument("--no-paired", action="store_true")
    parser.add_argument("--no-tracker", action="store_true")
    parser.add_argument("--planner-mode", choices=("p2a", "p2b"), default="p2a")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.jobs_json:
        run_jobs(
            json.loads(args.jobs_json.read_text()),
            args.output,
            source=args.source,
            observation=args.observation,
        )
        return
    if args.probe_metadata:
        probe_metadata(args.output)
        return
    report = benchmark(
        args.output,
        observation="oracle" if args.bridge_only else args.observation,
        source=args.source,
        seed=args.seed,
        labels=args.case,
        paired=not args.no_paired,
        asset_root=args.asset_root,
        generated=args.generated,
        layout_split=args.layout_split,
        no_tracker=args.no_tracker,
        planner_mode=args.planner_mode,
    )
    if args.bridge_only and not report["bridge_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
