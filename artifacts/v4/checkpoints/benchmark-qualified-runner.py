"""V4 evaluator (forked from V3): oracle diagnostics and image evaluation have separate inputs.

Only the oracle branch extracts semantic labels. Real image agents receive the
unmodified image observation; rewards/termination always come from the true env.
"""

from __future__ import annotations

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

    from agents.v3_bridge import CLASSES

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
        if options.get("appearance_sampling") == "class_group_variant":
            from tools.v4_data import choose_asset

            entry = choose_asset(entries, rng)
        else:
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
            # Dead artwork is outside the14 initial classes. A terminal loss
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
            from agents.v3_bridge import reconstruct

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
    vision_library_preloaded = "torch" in sys.modules
    try:
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        from grid_adventure.env import GridAdventureEnv
        from grid_adventure.grid import to_state

        from public_levels.catalog import (
            PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
            PUBLIC_LEVEL_MAX_REWARDS,
        )
        from tools.benchmark_p1 import load_agent, provenance

        def build():
            if options.get("p2_case") is not None:
                from tests.p2_fixtures import heldout_case

                return heldout_case(options["p2_case"])[0]
            if options.get("generated") is not None:
                if options.get("generator") == "v4_optional_sides":
                    from tools.v4_data import new_layout
                else:
                    from tools.v3_data import new_layout

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
            from tools.v3_data import canonical, digest

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

            from agents.v3_bridge import CLASSES

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
        vision_preloaded_at_construction = "torch" in sys.modules
        if cls is None:
            from agents.v3_planner import PlannerAdapter
            from agents.v3_types import EpisodeBudget

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
            from agents.v3_types import ObservationCheck

            agent._step_image.__globals__["check_observation"] = (
                lambda *args, **kwargs: ObservationCheck(True, (), ())
            )
        if options.get("planner_mode", "p2a") != "p2a":
            original_adapter = agent._step_image.__globals__["PlannerAdapter"]
            agent._step_image.__globals__["PlannerAdapter"] = lambda budget: (
                original_adapter(budget, mode=options["planner_mode"])
            )
        if options.get("trace"):
            agent = trace_agent(agent, env, options)
        result = evaluate_episode(
            env,
            agent,
            observation=observation,
            optimal=PUBLIC_LEVEL_MAX_REWARDS.get(label),
        )
        if options.get("trace"):
            result.update(diagnostic_only=True, trace=agent._v4_trace)
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
            diagnostic_only=bool(options.get("trace")),
            vision_library_preloaded=vision_library_preloaded,
            vision_preloaded_at_construction=vision_preloaded_at_construction,
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
    from tools.benchmark_p1 import THREAD_VARS

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
        result["worker_pid"] = process.pid
        result["fresh_worker"] = True
        result["watchdog_origin"] = "before_process_start"
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
        row.update(job=job)
        rows.append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(len(rows), label, row["status"], row.get("reward"), flush=True)
        if row.get("source_sha256") != sha:
            raise ValueError("Worker source differs from the frozen report source")
    return report


def validate_timing_report(report):
    if report.get("diagnostic_only") or any(
        r.get("diagnostic_only") for r in report.get("results", ())
    ):
        raise ValueError("diagnostic trace cannot qualify as timing")
    seen = set()
    for row in report.get("results", ()):
        if row.get("source_sha256") != report.get("source_sha256"):
            raise ValueError("mixed-source report")
        if row.get("runtime_overrides"):
            raise ValueError("runtime overrides cannot qualify")
        pid = row.get("worker_pid")
        if not row.get("fresh_worker") or pid is None or pid in seen:
            raise ValueError("qualification requires a fresh worker for every row")
        seen.add(pid)
        if (
            row.get("vision_library_preloaded") is not False
            or row.get("vision_preloaded_at_construction") is not False
        ):
            raise ValueError("preloaded vision library invalidates cold qualification")
        if (
            row.get("watchdog_origin") != "before_process_start"
            or row.get("cold_watchdog_seconds") != 5
        ):
            raise ValueError("changed cold watchdog origin or limit")
    return True


def replay_job(job: dict, source: Path, output: Path, *, trace: bool) -> dict:
    spec = dict(job)
    label = spec.pop("label")
    result = run_case(
        label,
        source=str(source),
        observation="image",
        cold_limit=30 if trace else 5,
        trace=trace,
        trace_output=str(output.with_suffix("")),
        **spec,
    )
    result.update(job=dict(job), diagnostic_only=trace)
    report = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "diagnostic_only": trace,
        "results": [result],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def trace_agent(agent, env, options):
    """Evaluator-only wrappers. Never included in any submitted agent."""
    from grid_adventure.grid import from_state

    events = []
    first_exception = None
    original_step = agent.step
    original_image = getattr(agent, "_step_image", None)
    stages = {}
    observations = {}
    if original_image:
        namespace = original_image.__func__.__globals__
        for name in (
            "classify_initial",
            "check_observation",
            "reconcile",
            "common_safe_prefix",
            "grid_step",
        ):
            original = namespace.get(name)
            if original is None:
                continue

            def measured(*args, _name=name, _fn=original, **kwargs):
                start = perf_counter()
                try:
                    value = _fn(*args, **kwargs)
                    if _name == "check_observation":
                        observations["check"] = {
                            "consistent": value.consistent,
                            "mismatches": value.mismatches,
                            "changed_cells": value.changed_cells,
                        }
                    return value
                finally:
                    stages[_name] = stages.get(_name, 0) + perf_counter() - start

            namespace[name] = measured
        original_adapter = namespace["PlannerAdapter"]

        class MeasuredAdapter(original_adapter):
            def step(self, state):
                start = perf_counter()
                try:
                    return super().step(state)
                finally:
                    stages["planner"] = (
                        stages.get("planner", 0) + perf_counter() - start
                    )

        namespace["PlannerAdapter"] = MeasuredAdapter

        def image_step(obs):
            nonlocal first_exception
            try:
                return original_image(obs)
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                if first_exception is None:
                    first_exception = {
                        "type": type(error).__name__,
                        "message": str(error),
                        "stack": traceback.format_exc(),
                        "turn": obs["info"]["status"]["turn"],
                    }
                    output = Path(options["trace_output"])
                    output.mkdir(parents=True, exist_ok=True)
                    from PIL import Image

                    Image.fromarray(obs["image"]).save(output / "first-exception.png")
                    (output / "first-exception-info.json").write_text(
                        json.dumps(obs["info"], indent=2)
                    )
                # The caller is the original production step handler: it catches
                # this same exception and emits its real WAIT exactly once.
                raise

        agent._step_image = image_step

    def traced(obs):
        stages.clear()
        observations.clear()
        turn = obs["info"]["status"]["turn"]
        before = tuple(h.state.turn for h in getattr(agent, "_hypotheses", ()))
        start = perf_counter()
        action = original_step(obs)
        history = getattr(agent, "_history", None)
        truth = semantic_labels(from_state(env.state))
        parsed = (
            semantic_labels(history.hypotheses[0].state)
            if history and history.hypotheses
            else None
        )
        event = {
            "turn": turn,
            "action": action.name,
            "prediction_turns_before": before,
            "prediction_turns_after": [
                h.state.turn for h in getattr(agent, "_hypotheses", ())
            ],
            "hypothesis_count": len(getattr(agent, "_hypotheses", ())),
            "current_board_matches_truth": parsed == truth,
            "agent_info": agent.info(),
            "elapsed": perf_counter() - start,
            "stages": dict(stages),
            "first_exception": first_exception,
            "observations": dict(observations),
        }
        events.append(event)
        return action

    agent.step = traced
    agent._v4_trace = events
    return agent


def corpus_jobs(corpus: Path, appearance_manifest: Path) -> list[dict]:
    """Bind every evaluator selector to a previously frozen decoded image."""
    from tools.v4_data import canonical, digest

    corpus, appearance_manifest = Path(corpus), Path(appearance_manifest)
    manifest = json.loads(appearance_manifest.read_text())
    report = json.loads((corpus / "report.json").read_text())
    if (
        hashlib.sha256((corpus / "boards.jsonl").read_bytes()).hexdigest()
        != report["records_sha256"]
    ):
        raise ValueError("Corpus records differ from frozen report")
    jobs = []
    for line in (corpus / "boards.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["manifest_sha256"] != digest(canonical(manifest)):
            raise ValueError("Corpus appearance manifest mismatch")
        label = f"{record['family']}/{record['layout_sha256'][:12]}/{record['appearance_seed']}"
        jobs.append(
            {
                "label": label,
                "seed": record["layout_seed"],
                "generated": record["layout_index"],
                "generator": "v4_optional_sides",
                "multiple_gems": True,
                "layout_split": record["geometry_split"],
                "expected_layout_sha256": record["layout_sha256"],
                "appearance_manifest": str(appearance_manifest),
                "appearance_manifest_sha256": digest(canonical(manifest)),
                "appearance_split": record["appearance_split"],
                "appearance_seed": record["appearance_seed"],
                "appearance_sampling": "class_group_variant",
                "resolution": record["resolution"],
                "asset_root": str(corpus / "asset-root"),
                "expected_pixels_sha256": record["pixels_sha256"],
                "expected_image_sha256": record["image_sha256"],
            }
        )
    return jobs


def run_matrix(jobs: list[dict], candidates: dict[str, Path], output: Path) -> dict:
    if set(candidates) != {
        "old_perception_old_tracking",
        "old_perception_new_tracking",
        "new_perception_old_tracking",
        "new_perception_new_tracking",
        "historical_v3",
    }:
        raise ValueError(
            "Matrix requires four P3 corners and the historical V3 control"
        )
    for job in jobs:
        if any(
            k in job for k in ("oracle_labels", "truth", "no_tracker", "planner_mode")
        ):
            raise ValueError(
                "Oracle truth/runtime overrides cannot enter real-image matrix"
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name, source in candidates.items():
        report = run_jobs(
            jobs, output / (name + ".json"), source=str(source), observation="image"
        )
        validate_timing_report(report)
        reports[name] = report
    summary = {
        "candidate_sources": {
            k: hashlib.sha256(Path(v).read_bytes()).hexdigest()
            for k, v in candidates.items()
        },
        "cases": len(jobs),
        "wins": {
            k: sum(r["status"] == "win" for r in v["results"])
            for k, v in reports.items()
        },
        "exact_initial_boards": {
            k: sum(r["initial_board_exact"] is True for r in v["results"])
            for k, v in reports.items()
        },
    }
    (output / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
