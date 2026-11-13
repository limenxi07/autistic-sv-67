"""Sequential Task 1 benchmarks with isolated workers and episode watchdogs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def starter_ast():
    """Load the baseline without running notebook or evaluation code."""
    tree = ast.parse((ROOT / "agents/starter.py").read_text())
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # Module documentation is not part of the baseline hash.
    if (
        sum(isinstance(node, ast.ClassDef) and node.name == "Agent" for node in body)
        != 1
    ):
        raise ValueError("Expected exactly one starter Agent class")
    if any(
        not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
        for node in body
    ):
        raise ValueError("Starter contains unexpected executable statements")
    return ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))


def load_agent(spec: str) -> type:
    if spec == "starter":
        namespace: dict[str, Any] = {"__name__": "p1_starter"}
        exec(  # noqa: S102
            compile(starter_ast(), str(ROOT / "agents/starter.py"), "exec"), namespace
        )
        return namespace["Agent"]
    if spec.endswith(".py"):
        module_spec = importlib.util.spec_from_file_location(
            "p1_artifact", Path(spec).resolve()
        )
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"Cannot load {spec}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        return module.Agent
    module_name, class_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def source_bytes(spec):
    if spec == "starter":
        return ast.unparse(starter_ast()).encode()
    if spec.endswith(".py"):
        return Path(spec).read_bytes()
    module = importlib.util.find_spec(spec.split(":", 1)[0])
    if module is None or module.origin is None:
        raise ValueError(f"No source for {spec}")
    return Path(module.origin).read_bytes()


def effective_performance(reward, optimal, minimum, win, status):
    if status != "win" or not win:
        return 0.0
    if optimal is None or minimum is None:
        return None  # No invented optimum for generated layouts.
    from utils import get_performance

    return get_performance(reward, optimal, minimum, win)


def provenance():
    from importlib.metadata import distribution

    packages = {}
    for name in ("grid-adventure", "grid-universe", "numpy", "pillow", "torch"):
        try:
            dist = distribution(name)
            packages[name] = {
                "version": dist.version,
                "direct_url": json.loads(dist.read_text("direct_url.json") or "null"),
            }
        except Exception:  # noqa: BLE001
            packages[name] = None
    cpu = platform.processor() or None
    if sys.platform == "linux":
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu": cpu,
        "python": sys.version,
        "packages": packages,
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_limits": {key: os.environ.get(key) for key in THREAD_VARS},
    }


THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _worker(connection, spec, label, seed, split, evaluation_limit, profile):
    began = time.perf_counter()
    started = None
    try:
        if hasattr(os, "sched_setaffinity"):
            # Respect taskset; pin each worker to one allowed CPU.
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        agent: Any = load_agent(spec)
        if profile:
            module: Any = sys.modules.get(agent.__module__)
            if module is None or not hasattr(module, "PROFILE"):
                raise ValueError("Agent does not expose the P1 profiling switch")
            module.PROFILE = True
        instances = []

        class CapturedAgent(agent):
            def __init__(self):
                super().__init__()
                instances.append(self)

        from public_levels.catalog import (
            PUBLIC_GAMEPLAY_LEVEL_BUILDERS,
            PUBLIC_LEVEL_MAX_REWARDS,
        )
        from utils import create_env, evaluate

        if split is None:
            builder = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[label]
            optimal = PUBLIC_LEVEL_MAX_REWARDS[label]
            minimum = optimal * 1.5
        else:
            from tests.planner_p1_fixtures import fixture_cases

            builder, _ = fixture_cases(split)[label]
            optimal = minimum = None
        env = create_env(
            builder, seed=seed, turn_limit=150, observation_type="gridstate"
        )
        metadata = provenance()
        metadata["source_sha256"] = hashlib.sha256(source_bytes(spec)).hexdigest()
        started = time.perf_counter()
        connection.send(
            ("started", started, {**metadata, "setup_seconds": started - began})
        )
        reward, win, lose, history = evaluate(CapturedAgent, env)
        elapsed = time.perf_counter() - started
        status = (
            "timeout" if elapsed >= evaluation_limit else ("win" if win else "loss")
        )
        result = {
            "status": status,
            "reward": reward,
            "win": win,
            "lose": lose,
            "evaluation_seconds": elapsed,
            "turns": len(history) - 1,
            "optimal_reward": optimal,
            "minimum_reward": minimum,
            "effective_performance": effective_performance(
                reward, optimal, minimum, win, status
            ),
            "termination": "evaluator-detected" if status == "timeout" else "completed",
            "trace_actions": [a.value for _, a in history[:-1]],
            "agent_statistics": instances[0].info()
            if hasattr(instances[0], "info")
            else None,
        }
    except BaseException:  # noqa: BLE001
        result = {
            "status": "setup_error" if started is None else "error",
            "error": traceback.format_exc(),
            "effective_performance": 0.0,
            "evaluation_seconds": None
            if started is None
            else time.perf_counter() - started,
            "termination": "exception",
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    try:
        connection.send(("result", result))
    finally:
        connection.close()


def run_case(
    agent_spec: str,
    level_label: str,
    seed: int,
    *,
    split=None,
    evaluation_limit=5.0,
    setup_limit=30.0,
    profile=False,
) -> dict[str, object]:
    # Set before spawn/import so numerical libraries cannot create a thread pool.
    previous = {name: os.environ.get(name) for name in THREAD_VARS}
    for name in THREAD_VARS:
        os.environ[name] = "1"
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(child, agent_spec, level_label, seed, split, evaluation_limit, profile),
    )
    result = {
        "agent": agent_spec,
        "level": level_label,
        "seed": seed,
        "split": split,
        "turn_limit": 150,
        "evaluation_limit_seconds": evaluation_limit,
        "setup_limit_seconds": setup_limit,
        "profiled": profile,
        "status": "setup_error",
        "effective_performance": 0.0,
        "reward": None,
        "turns": None,
        "win": None,
        "lose": None,
        "evaluation_seconds": None,
        "setup_seconds": None,
        "peak_rss_bytes": None,
        "source_sha256": None,
    }
    started = None
    wall_start = time.perf_counter()
    try:
        process.start()
        child.close()
        deadline = wall_start + setup_limit
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0 or not parent.poll(max(0, remaining)):
                result.update(
                    status="setup_error" if started is None else "timeout",
                    termination="local-watchdog",
                )
                if started is not None:
                    result["evaluation_seconds"] = time.perf_counter() - started
                else:
                    result["setup_seconds"] = time.perf_counter() - wall_start
                break
            try:
                message = parent.recv()
            except EOFError:
                result.update(
                    status="setup_error" if started is None else "error",
                    termination="worker-exit",
                    error="Worker exited without result",
                )
                break
            if message[0] == "started":
                _, started, metadata = message
                result.update(metadata)
                deadline = started + evaluation_limit
            else:
                result.update(message[1])
                break
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            result["worker_exitcode"] = process.exitcode
        parent.close()
        child.close()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    result["worker_wall_seconds"] = time.perf_counter() - wall_start
    return result


def verified_fixture_cases(split):
    """Exclude unverified witnesses before any generated win-rate denominator."""
    from grid_adventure.grid import step

    from tests.planner_p1_fixtures import fixture_cases

    verified, excluded = [], []
    for label, (builder, witness) in fixture_cases(split).items():
        try:
            state = builder()
            for action in witness:
                state = step(state, action)
            if not state.win:
                raise ValueError("Witness did not win")
            verified.append(label)
        except Exception as error:  # noqa: BLE001
            excluded.append({"level": label, "error": str(error)})
    return verified, excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="starter")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Separate diagnostic timers; not acceptance evidence",
    )
    parser.add_argument(
        "--suite", choices=("intro", "task1", "generated"), default="task1"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--split", choices=("tuning", "heldout"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    excluded = []
    if args.suite == "generated":
        if args.split is None:
            parser.error("--split is required for generated")
        labels, excluded = verified_fixture_cases(args.split)
        if not labels:
            parser.error("Generated fixtures are not implemented yet (Task 5)")
    else:
        from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS

        labels = [
            label
            for label in PUBLIC_GAMEPLAY_LEVEL_BUILDERS
            if args.suite == "task1" or label.startswith("A")
        ]
    report = {
        "schema_version": 1,
        "suite": args.suite,
        "agent": args.agent,
        "source_hash_definition": "SHA256 of AST-extracted imports/class for starter; file bytes otherwise",
        "unavailable_measurements": "null; watchdog kills cannot return final reward, turns or peak RSS",
        "verified_generated_count": len(labels) if args.suite == "generated" else None,
        "excluded_generated": excluded,
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(args.repeat):
        for label in labels:
            result = run_case(
                args.agent,
                label,
                42 + repetition,
                split=args.split if args.suite == "generated" else None,
                profile=args.profile,
            )
            result["repetition"] = repetition
            report["results"].append(result)
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(args.output)
            print(
                f"{label}: {result['status']} p={result['effective_performance']} "
                f"seconds={result['evaluation_seconds']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
