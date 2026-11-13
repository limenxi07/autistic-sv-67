"""Diagnostic only: fresh processes, full public-helper loop and timing stress."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def worker(args):
    import cProfile
    import pstats
    import io
    from tools.benchmark_planner_p1 import load_agent, provenance
    from grid_adventure.env import GridAdventureEnv
    from grid_adventure.grid import to_state, from_state
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS, PUBLIC_LEVEL_MAX_REWARDS

    real = time.perf_counter
    origin = real()
    extra = 0.0

    def clock():
        return origin + args.scale * (real() - origin) + extra

    time.perf_counter = clock
    load_start = real()
    cls = load_agent(args.source)
    load_s = real() - load_start
    state = PUBLIC_GAMEPLAY_LEVEL_BUILDERS[args.label]()
    state.seed, state.turn_limit = args.seed, 150
    env = GridAdventureEnv(initial_state_fn=lambda *a, **k: to_state(state),
        width=state.width, height=state.height, observation_type="image")
    profile = cProfile.Profile() if args.profile else None
    if profile:
        profile.enable()
    started, real_started = clock(), real()
    t = real()
    obs, _ = env.reset()
    reset_s = real() - t
    t = real()
    agent = cls()
    init_s = real() - t
    history, rows = [], []
    win = lose = False
    reward = 0
    first_cutoff = None
    while not (win or lose) and len(rows) < 150 and clock() - started < 5:
        t = real()
        action = agent.step(obs)
        agent_s = real() - t
        reason = agent.info().get("reason")
        if reason == "cutoff" and first_cutoff is None:
            first_cutoff = {"turn": len(rows), "elapsed": clock() - started}
        t = real()
        if args.history:
            history.append((from_state(env.state), action))
        history_s = real() - t
        t = real()
        obs, r, win, lose, _ = env.step(action)
        env_s = real() - t
        extra += max(0, args.env_floor - args.scale * env_s)
        reward += r
        rows.append(dict(turn=len(rows), action=action.name, reason=reason,
            agent_s=agent_s, history_s=history_s, env_s=env_s,
            elapsed=clock() - started))
    if args.history:
        t = real()
        history.append((from_state(env.state), None))
        final_history_s = real() - t
    else:
        final_history_s = 0
    elapsed, real_elapsed = clock() - started, real() - real_started
    if profile:
        profile.disable()
        buf = io.StringIO()
        pstats.Stats(profile, stream=buf).sort_stats("cumulative").print_stats(45)
        Path(args.profile).write_text(buf.getvalue())
    target = PUBLIC_LEVEL_MAX_REWARDS[args.label]
    perf = max(0, (reward - target * 1.5) / (-target * .5)) if win and elapsed < 5 else 0
    result = dict(label=args.label, seed=args.seed, scale=args.scale, env_floor=args.env_floor,
        history=args.history, win=bool(win), lose=bool(lose), timeout=elapsed >= 5,
        performance=perf, score=min(perf / .5, 1), reward=reward, target=target,
        seconds=elapsed, real_seconds=real_elapsed, source_load_s=load_s,
        reset_s=reset_s, init_s=init_s, turns=len(rows), first_cutoff=first_cutoff,
        agent_s=sum(r["agent_s"] for r in rows),
        history_s=sum(r["history_s"] for r in rows) + final_history_s,
        env_s=sum(r["env_s"] for r in rows), steps=rows, info=agent.info(),
        provenance=provenance())
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="submissions/task2_agent.py")
    p.add_argument("--output")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--label")
    p.add_argument("--seeds", type=int, nargs="+", default=[2, 43, 20261109])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=float, default=1)
    p.add_argument("--env-floor", type=float, default=0)
    p.add_argument("--history", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threads", choices=["default", "one"], default="default")
    p.add_argument("--profile")
    args = p.parse_args()
    if args.worker:
        worker(args)
        return
    from public_levels.catalog import PUBLIC_GAMEPLAY_LEVEL_BUILDERS
    rows = []
    report = dict(source=args.source, sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        synthetic=args.scale != 1 or args.env_floor != 0, threads=args.threads,
        evidence_boundary="Local macOS diagnosis; no grader board mapping or platform certification.", results=rows)
    env = dict(os.environ)
    thread_vars = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
    if args.threads == "one":
        env.update({key: "1" for key in thread_vars})
    report["thread_environment"] = {key: env.get(key) for key in thread_vars}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    for label in [args.label] if args.label else PUBLIC_GAMEPLAY_LEVEL_BUILDERS:
        for seed in args.seeds:
            cmd = [sys.executable, __file__, "--worker", "--source", args.source,
                "--label", label, "--seed", str(seed), "--scale", str(args.scale),
                "--env-floor", str(args.env_floor), "--history" if args.history else "--no-history"]
            if args.profile:
                cmd += ["--profile", args.profile]
            try:
                result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30, check=True)
                row = json.loads(result.stdout)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(e.stderr) from e
            except subprocess.TimeoutExpired:
                row = dict(label=label, seed=seed, process_timeout=True, win=False, performance=0)
            rows.append(row)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in row.items() if k not in ("steps", "provenance", "info")}), flush=True)


if __name__ == "__main__":
    main()
