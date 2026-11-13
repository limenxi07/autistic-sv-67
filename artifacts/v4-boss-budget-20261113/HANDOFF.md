# Boss-mode deadline repair

Paste the complete `submission/task3_agent.py` into the **Task 3 / boss-mode Agent cell**. Continue using the previous Task 2 submission for Task 2.

Source SHA256: `8b8f72d79335ba31cf504ae8d1cf33954946905bcd45c6dfecd05f2308323104`. Size: 720,407 bytes.

## Diagnosis

The user reported 2/4 Debugging passes for V4 `0b61f5c7c594`, with The Heist losing in both observation modes at reward -450, without an error or platform timeout. That source retains Task 2's 3.0-second initial search, 3.5-second planning cutoff and 4.65-second image execution cutoff. The platform's 15-second boss limit does not extend these internal deadlines.

The exact old source reproduced the 2/4 pattern under a fourfold elapsed-clock slowdown: C0 won in both modes; Heist used 150 WAIT actions and lost at -450 in both. The initial image reconstruction was exact. Search stopped at its deadline with no first win. See `baseline-scale4.json`; this is synthetic timing evidence, not a trace from the platform.

## Change

An explicit `BossAgent` wrapper uses 11 seconds for initial planning, 12 seconds for the planning cutoff and 14.65 seconds for image execution, all measured from the original construction time. Structured inputs use the existing P3 `Task3Agent` with the same original planning budget, including reference fallback. Image inputs retain V4 perception, verification, tracking and fallback. The exported public class remains `Agent` and requires no constructor arguments.

The candidate preserves the complete prior Task 2 source as its prefix and adds 617 bytes. Task 2 reproduces byte-for-byte at SHA256 `0b61f5c7c594541b12656995aa2389642ce37a48ad9f5a2bc3544360c45d92d5`; the frozen P3 backend remains `d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`. No model, search algorithm, level-specific route or geometry-based task selection changed.

## Qualification

- Exact standalone on Omarchy, one CPU core and one library thread, fresh workers: **12/12 wins**, both bosses and both modes, seeds 42, 123 and 211. Maximum episode duration 0.929 seconds.
- Function-scoped notebook source on Omarchy: **4/4 wins**.
- Eightfold elapsed-clock stress on Omarchy: **4/4 wins**, maximum simulated episode duration 7.391 seconds. Synthetic stress is not a platform runtime measurement.
- The original fourfold local reproduction now wins **4/4** with the repaired source.
- Every image run reconstructs the initial board exactly; no deadline resets or timeouts occurred.
- Heist rewards: **-163 structured / -165 image**, corresponding to performance **0.9367 / 0.9114** against the public optimum of -158. These are wins above the boss threshold, not optimal solutions.
- Regression tests observed WAIT failures before the wrapper, then passed for planning after five seconds, reference fallback, notebook scoping and deadline expiry. `red-behavior.log` retains the failures.
- Final repository suite: **381/381 tests passed** in 166.538 seconds; Ruff formatting/lint and type checks passed. The first suite run had three P1 harness setup failures (`test_error_and_loss`, `test_hanging_evaluation_is_terminated`, `test_optional_agent_statistics`) while Matplotlib repeatedly rebuilt an unwritable cache. All three passed with writable `MPLCONFIGDIR`/`XDG_CACHE_HOME`, followed by the clean full-suite run in `all-tests-final.log`. No production change was made for those environment failures.

These are development/public checks, not graded results. The new boss source still needs the user's platform retest.

## Reproduction

From the repository root, use `tools.package_v4.package(root, export, policy, output_dir, task3=True)` with export `artifacts/v4-search-recovery-20261113/export` and this directory's `policy.json`.

Run public bosses with:

```sh
.venv/bin/python -m tools.benchmark_v4_boss \
  --source artifacts/v4-boss-budget-20261113/submission/task3_agent.py \
  --output /tmp/boss-retest.json --seeds 42 123 211
```

Add `--scale 4` for a synthetic fourfold elapsed-clock slowdown. The benchmark applies a 15-second episode limit and an external 30-second process watchdog.
