# P2 Fast Simulator and Reward-Aware Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans, or superpowers:subagent-driven-development if that execution method is selected, to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the stable P1 Task 1 checkpoint through a validated compact simulator and bounded reward-aware search, with measurable CPU headroom and stronger generalization.

**Architecture:** P2a ports the frozen P1 search policy to compact states and fast transitions. P2b adds proved reward bounds while preserving a verified winning incumbent. A reference path validates transitions and executable plans; all runtime work shares one episode budget.

**Tech Stack:** Existing Python/uv environment, standard library, public Grid Adventure API, unittest, CPU profiling. No new runtime dependency or GPU training is required.

**Spec:** [Proposed P2 design](../specs/2026-11-08-p2-fast-planner-design.md), [P2 approach](../../approaches.md), [assignment](../../../capstone-project.py), [tutorial](../../../tutorial.py).

## Global constraints

- Structured Task 1 input; P2 is the approach label from `approaches.md`. Task 2 images and learned guidance are later work.
- 150 turns, 5 seconds for the entire episode, 1.5 GB memory, source below 2,000,000 bytes, one CPU core, no GPU.
- Initial planning ends at 3.0 seconds and all planning/repair/reference replay ends by 3.5 seconds from construction. Target measured episodes below 4.25 seconds and RSS below 1.0 GB.
- Public documentation/tutorial and black-box API probes only; no installed engine-source inspection. No map names, layouts, catalog rewards or memorized action sequences in the submitted agent.
- Preserve P1 source, artifacts and existing work. The baseline associated with 20/22 is archived at `artifacts/p1/pre-reference/candidate/task1_agent.py`, hash `ce5a70347773f3eb9609db7f6e9c04f253c1eb0fdc44754c6bb277ce9165929b`; the submitted hash is unconfirmed. The current export and working source changed during planning; select by hash and do not overwrite them.
- Retain compatible standalone `Agent().step(...) -> Action`, Python 3.10 grammar, grader-prelude and unregistered-module loading behavior.
- Every reported win is replayable through the public engine. Bounds and dominance may prune only under documented assumptions. All capped results are best found unless exhaustive completion establishes optimality.
- This document schedules implementation; writing it does not authorize a platform submission or start a compute job.

## Review focus

1. Stacked effects, overlapping objects and action/event order may invalidate a seemingly sufficient state (Tasks 2–3).
2. Boots/phasing can invalidate movement heuristics; coins invalidate ordinary nonnegative-cost shortest-path assumptions (Task 5).
3. A faster transition can expose heuristic/heap/memory overhead that older P1 profiling did not measure (Tasks 1, 4–5).
4. Verification or fallback can spend the execution reserve or discard a previously usable incumbent (Tasks 4–5).
5. Source drift, a reused held-out set, or local-only success can make qualification misleading (Tasks 1, 6).

## File map and interfaces

Keep the runtime self-contained in `agents/p2.py`, matching P1's packaging convention. Organize board/state encoding, transitions, ordering/search, and Agent as distinct sections. Test/benchmark tools remain external.

- Create `agents/p2.py`: `Board`, `FastState`, `FastResult`, `Verification`, `UnsupportedState`, `encode`, `observe_key`, `predicted_key`, `fast_step`, `search`, `verify_plan`, `Agent` and `ReferenceAgent`. The reference fallback copies/namespaces the frozen P1 helpers into this file with `reference_` prefixes; it never imports `agents.p1` at runtime. Adapt only its budget plumbing initially.
- Create `tests/p2_fixtures.py`: mechanic combinations, deterministic layout families, action witnesses and split manifests.
- Create `tests/test_p2_state.py`, `tests/test_p2_step.py`, `tests/test_p2_search.py`, `tests/test_p2_agent.py`, `tests/test_p2_qualification.py` and `tests/test_p2_package.py`.
- Create `tools/differential_p2.py`: independent public-state projection, traces, mismatch minimization and coverage reports.
- Create `tools/benchmark_p2.py`: paired baseline/P2 measurements, profiles, constrained budgets and new layout suites, reusing `tools/benchmark_p1.py` worker/watchdog/provenance helpers.
- Create `tools/qualify_p2.py`: P2-specific complete-coverage, paired-regression, deadline/memory, differential and hash gates; do not reuse P1's lower win threshold.
- Reuse `tools/package_p1.py --source agents/p2.py --output-dir artifacts/p2/candidate` for unchanged-byte copying; its output name `task1_agent.py` is appropriate. Extend recognized configuration names only if P2 adds them, with existing package tests preserved.
- Write execution artifacts under `artifacts/p2/`: `baseline/`, `differential/`, `p2a/`, `p2b/`, `candidate/`, split manifests and reports.

Shared runtime contracts, all in `agents/p2.py`:

```python
encode(observation: GridState) -> tuple[Board, FastState, int]
observe_key(board: Board, observation: GridState) -> tuple
predicted_key(board: Board, state: FastState, score: int) -> tuple
fast_step(board: Board, state: FastState, action: Action) -> tuple[FastState, int]
search(board: Board, start: FastState, score: int, *,
       deadline: float, mode: str,
       verify_candidate: Callable[[tuple[Action, ...], tuple[tuple, ...]], Verification],
       clock=perf_counter) -> FastResult
verify_plan(board: Board, observation: GridState, actions: tuple[Action, ...],
            predicted_keys: tuple[tuple, ...], *,
            deadline: float, clock=perf_counter) -> Verification
```

`encode` returns board, dynamic state and accumulated score, or raises `UnsupportedState`. `fast_step` returns successor and score delta, retaining terminal flags in the state. `observe_key` canonicalizes a live state relative to the episode's stable slots and includes score; incompatible observations raise `UnsupportedState`. `predicted_key` produces the same schema from compact values. `Verification` contains status (`verified`, `invalid` or `deadline`), actual_score (`int | None`), and actual_keys (a tuple of canonical keys). Only `verified` has a complete N+1 sequence, a matching predicted sequence and a winning final state. The Agent supplies a callback closing over Board, starting observation and the shared deadline. `FastResult` contains actions, expected_keys, winning, final_score, stop_reason, expansions, transitions, time_to_first_win, peak_states and peak_records; winning and final_score refer only to reference-verified incumbents, and time_to_first_win includes verification. On `invalid`, abort fast search; on `deadline`, finish with the previous verified incumbent. Use NamedTuple/primitive tuples and non-PEP-695 aliases.

### Task 1: Freeze evidence and establish current bottlenecks

- [ ] Verify the baseline SHA256 and copy its exact bytes and manifest to `artifacts/p2/baseline/`; record the user-reported 20/22 and unconfirmed submitted hash distinctly. Retain the working P1 hash separately.
- [ ] Add benchmark tests for paired case/repetition matching, watchdog status, full provenance and requested transition budgets. Run `uv run python -m unittest tests.test_p2_qualification -v` before and after implementing those helpers.
- [ ] Run the unchanged exported P1 through all 22 cases with three repetitions on the chosen single-core CPU. Capture per-case reward/performance, whole-episode time, cache hits and resource use. Keep profiler timings in a separate run.
- [ ] Profile B5, B6 and representative easy/coin/effect levels. Account separately for transitions, state keys, route estimates/action ordering, frontier maintenance and execution. Save first-win time and simulations, not only the full search timeout.
- [ ] Establish deterministic operation-budget probes at 300, 600, 1,500 and 5,000 transitions, plus 0.75-second and normal 3.0-second planning limits. Budget injection is benchmark-only and preserves baseline bytes in the acceptance run. These probes diagnose sensitivity; they do not emulate the remote CPU.

**Gate:** Exact baseline and reproducible current cost breakdown exist. If ordering now dominates, prioritize its compact port and caching alongside transitions; do not repeat the historical 86% figure as a new finding.

### Task 2: Encode complete compact states

- [ ] Add state tests for copies, all entity classes, keys before/after use, opened doors, active and stacked effects, box/object overlaps, score-only differences, exact remaining turns and unsupported fields. Add an independent public-field semantic projection in the differential tool.
- [ ] Run `uv run python -m unittest tests.test_p2_state -v` to see the missing behavior, then implement `Board`, `FastState`, `UnsupportedState`, `encode` and `observe_key`.
- [ ] Cache static board geometry and item slots per episode. Use masks for remaining items/doors, primitive positions for boxes, and explicit counters/references for resources. Preserve semantically relevant object order/identity until probes establish interchangeability. Assert initial score is not assumed zero when encoding a repair state.
- [ ] Exercise every reachable frame from P1 mechanic witnesses, including dynamic overlap. Confirm canonical state differences independently; do not validate only by calling the same encoder twice.
- [ ] Run state and existing P1 tests. Document supported schema and rejection behavior beside `encode`.

**Gate:** No known gameplay field is dropped; unsupported configurations route to the reference path rather than guessed dynamics.

### Task 3: Implement and differentially validate transitions

- [ ] Add focused public-API characterization tests before each mechanic implementation: movement/bounds, pickup and reward, key use with multiple adjacent doors, pushing/blocked pushes, hazards/health/shields, boots substeps, phasing, stacked effects/expiry, WAIT/invalid actions, terminal ordering and the final free winning turn. Probe interactions, not only isolated mechanics.
- [ ] Run `uv run python -m unittest tests.test_p2_step -v`, implement the corresponding behavior in pure `fast_step`, then rerun. Assert input states are unchanged and no mutable data leaks between successors.
- [ ] Implement `tools/differential_p2.py --layouts N --comparisons N --seed N --output PATH`. At sampled reachable states compare every Action to public `grid_step`: independently projected state, score delta, win and lose. Mix witness-biased rollouts and random actions so later effects and puzzle stages are reached.
- [ ] Check a minimum of 100,000 transitions from at least 1,000 valid layouts. Include all mandatory interaction categories; report counts per category and fail if a category has no coverage. Persist seed, source hashes, minimal builder/action trace and first differing field for every mismatch.
- [ ] Add each minimized mismatch as a regression test, then rerun its affected categories and the full differential gate. Benchmark transition-plus-key throughput on the same saved input/action corpus, separately from reference-oracle overhead.

**Gate:** Zero known mismatches and complete mandatory coverage. Any unresolved mismatch blocks promoting the fast simulator, regardless of win count or speed.

### Task 4: P2a — integrate fast simulation without changing search policy

- [ ] Add tests for inherited action ordering, deferred alternatives, finite horizon, shared absolute deadlines, memory caps and preservation of an incumbent's parent chain. Use fixed clocks for control-flow tests.
- [ ] Implement `search(..., mode="p2a")` with the frozen P1 priority/deferred expansion policy operating directly on compact states. Initially retain score in deduplication to isolate the backend change; optional score separation belongs to Task 5. Port route estimates without treating them as admissible bounds.
- [ ] Implement `ReferenceAgent` from namespaced baseline code with `step_until(observation, deadline)` so fallback shares the outer cutoff. Test its actions and rewards against the frozen baseline under equal deterministic budgets; do not initialize a fresh time allowance after a mismatch.
- [ ] Implement `verify_plan` and the P2 Agent callback/cache. Search calls `verify_candidate(actions, predicted_keys)` for potential improvements, before accepting them. The callback replays through the public API, compares all N+1 canonical states including scores, and returns explicit status and actual reward. Only `verified` updates the executable incumbent; `invalid` stops fast search and triggers bounded reference fallback if needed. Check deadlines before and after each replay step.
- [ ] Test a deliberately wrong fast transition, unexpected live observation, failed/expired upgrade replay and unsupported schema. Assert no unverified action sequence is dispatched, a valid incumbent survives failed upgrades, and reference repair obeys the original deadline. If no verified solution exists, return an explicit bounded fallback Action and record failure; do not call a partial plan a win.
- [ ] Run `uv run python -m unittest tests.test_p2_search tests.test_p2_agent -v`. Compare P1/P2a on public22, existing P1 regressions and tuning layouts, both equal operations and equal wall time. Record end-to-end speed and first verified win; verify trace equivalence for equivalent operation schedules.

**Gate:** Preserve baseline wins/rewards under paired equal-operation checks and gain measurable wall-clock headroom. A 10x transition-plus-key speedup is an ambition, not a claim or sufficient acceptance criterion. Preserve P2a as its own candidate if qualified.

### Task 5: P2b — safe reward branch-and-bound

- [ ] Add an exhaustive reference search for tiny maps with short finite horizons. Test longer coin routes beating shorter wins, distinct turns/effects staying distinct, WAIT needed for useful timer expiry, and the standard score identity from coins/turns/terminal flags. If retaining a generic best-score table, test reopening with explicitly synthetic scores; such collisions do not arise naturally under the exact supported key.
- [ ] Implement `search(..., mode="p2b")`: continued reward improvement after the first verified win, optional score separation for bookkeeping, stale-record handling, and incumbent-preserving resource caps. Do not claim score dominance reduces search with exact turns and remaining coins retained. Keep `mode="p2a"` for ablation.
- [ ] Add `reward_upper_bound(board, state, score) -> int`. Start with S+5*C, then separately enable the proved MST/maximum-speed relaxation from the design. On every exhaustively solved tiny state assert the bound is at least the best reachable final reward; bounds must never prune a higher-reward winning continuation.
- [ ] Maintain independent ordering estimates and admissible bounds. Prune against the verified incumbent's actual score only; test that a deliberately inflated simulated reward cannot establish a pruning threshold. Keep all Action alternatives and prohibit unproved deadlock pruning, timer dominance, forced pickups and irreversible box macros.
- [ ] Add tests for deadline/cap exits before and after a win, interrupted candidate replay, stale heap growth and record retention. Account for every persistent table/queue/path, not only live states.
- [ ] Run `uv run python -m unittest tests.test_p2_search tests.test_p2_agent -v`. Compare P2b to P2a at the same end-to-end budgets; tune only on development layouts. Change caps from 4,096/32,768 only after memory measurements, and do not exceed the proposed 65,536/262,144 ceilings without a new review of RSS.

**Gate:** Tiny-map exhaustive reward agreement, sound tested bounds, no paired reliability regression, and a measured reward or first-win improvement. Otherwise ship the independently qualified P2a.

### Task 6: Qualify one exact artifact

- [ ] Freeze a new P2 held-out manifest before tuning: 200 layouts across ten families, 20 each (maze, multi-gem, coins, keys/doors, boxes, lava/shields, boots, phasing, mixed effects, combined mechanics). Include structurally different 15x15 and low-health cases, not only rotations of public maps. Replay solvability witnesses through the reference API; keep unsolved/unsolvable diagnostics separate. Keep the old 100 P1 generated layouts as regressions.
- [ ] Add qualification tests that reject missing/duplicate cases, mismatched hashes, profiled acceptance runs, timeouts/errors, new lost wins, per-case reward regressions on paired wins, and resource violations. Test complete candidate import in a temporary directory, Python 3.10 parsing, grader prelude and module-registration edge cases.
- [ ] Run `uv run python -m unittest discover -s tests -p 'test_p*.py' -v`, the repository's Ruff lint/format checks and ty with warnings treated as errors. Package with `uv run python tools/package_p1.py --source agents/p2.py --output-dir artifacts/p2/candidate`.
- [ ] Benchmark the exported artifact: public22 three repeats; 200 fresh held-out cases; P1 regression suite; B5/B6 budget curves; and cold import/setup separately. Linux qualification pins one CPU and one library thread. Use an external 5-second evaluation watchdog. Do not replace unsupported affinity checks on macOS with a claim of pinned-core qualification.
- [ ] Require all public episodes below 4.25 seconds, peak RSS below 1.0 GB, no errors/timeouts, 22/22 wins each repeat, and effective performance at least 0.85 on each public case. On fresh held-out cases require at least 95% wins and no paired lost baseline wins or reward regressions; report raw reward when optima are unknown. Targets are proposed local gates, not estimated marks.
- [ ] Save exact bytes/hash, manifest, source/tests/tools, replay traces, differential coverage, raw reports and a supplementary archive. Keep separate statuses for local qualification and observed Coursemology Debugging. Hand the candidate to the user for Debugging; record the actual result/hash before considering it a platform improvement. Graded submissions remain a separate handoff.

**Gate:** One reproducible candidate with explicit passed/failed checks and an identifiable rollback to P1. If a gate fails, retain the prior checkpoint and report the unmet target.

## Execution order and compute

Tasks 1–3 establish the bottleneck and correctness; Task 4 yields the first usable P2 checkpoint; Task 5 is an independently judged upgrade; Task 6 qualifies the selected implementation. Freeze Task 6's held-out manifest at the start of execution, then leave its evaluation results unseen until selection. Use deterministic layout signatures to verify disjointness from tuning.

Planning allowance: about half a day for characterization/encoding, one day for transition interactions/differential testing, and half to one day for integration and qualification. Reassess at the end of each stage rather than promising 22/22. Resolve simulator correctness before increasing search sophistication.

Use the available CPU first. If offline sweeps are slow, inspect the M5080 host's identity and workload, then use idle CPU workers for data generation; qualification remains sequential single core. No GPU allocation or training is on the critical path. An optional learned ranker and the later V3 image parser require separate measured pilots.

## Plan review and status

All implementation tasks are pending. A read-only builder review identified two material issues, now resolved: search has an explicit verification callback with actual rewards/status, and score separation is correctly described as bookkeeping under the conservative key. The plan covers simulator fidelity, search correctness, runtime verification, bounded resources, held-out evaluation and exact-source handoff. P1 reports and code were inspected; no P2 code, training, remote job, submission or new benchmark was executed while drafting it.
