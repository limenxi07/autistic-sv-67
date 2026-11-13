# Runtime Repair Implementation Plan

> **For agentic workers:** Use executing-plans and test-driven-development; independently scoped boss repair may run in parallel under dispatching-parallel-agents.

**Goal:** Reduce image-agent runtime and prevent premature boss route abandonment.
**Architecture:** Retain existing planners/perception and replace redundant work with equivalent validation. Boss-only wrapper separates planning from verified execution.
**Tech Stack:** Python 3.13, NumPy, public Grid Adventure engine, unittest.
**Spec:** docs/superpowers/specs/2026-11-13-runtime-repair-design.md

## Global Constraints

Frozen P3/model/policy bytes unchanged. Task2 budgets 3.0/3.5/4.65; boss budgets 11/12/14.65. No geometry-based task selection, route lookup, deadline resets, external access or scored submission. Performance comparisons run sequentially.

## Review Focus

Mutation of cached source/successor or scenery must invalidate reuse. Differing keys and unknown state must retain fallback. Reference boss routes must obey original planning deadline. Image bookkeeping must preserve exact pixels and uncertainty checks. Packaged function-scoped source must behave like modules.

### Task 1: Runtime reduction

Files: agents/vision_v4_planner.py, agents/vision_v4_search.py, optionally vision_v4_tracking.py/perception.py/vision_v4.py; new tests/test_vision_v4_runtime_repair.py.
- [x] Add failing cost/behavior tests for repeated compact-schema scans during verified dispatch, pixel-equivalent background handling and deadline reason.
- [x] Implement the smallest measured reductions; keep complete state mutation guards and independent replay.
- [x] Run targeted V4 tests, package exact source, measure B3/B4 native and synthetic full-helper episodes. Expected: correct winning scores with improved runtime.

### Task 2: Boss execution

Files: agents/boss_v4.py and tests/test_boss_v4_execution.py.
- [x] Red tests: verified direct and reference suffixes execute after12 but before14.65; mismatches/expired/unplanned states WAIT without search.
- [x] Implement boss-specific structured adapter preserving frozen backend and planning deadlines.
- [x] Verify module/notebook packaging and repeat synthetic Heist reproduction. Expected: correct wins without budget resets.

### Task 3: Qualification and delivery

- [x] Run full unittest suite, syntax/package checks and independent code review; fix material findings.
- [x] Run candidate public66, notebook22 and boss12 with full helper overhead; retain source hashes and reports.
- [x] Compare sixfold task2 and eightfold boss stress with captured baselines; report any remaining limitations.
- [x] Reproduce packages, preserve older release bytes, and provide Debugging-first handoff.
- [x] Promote Task2 after user-reported platform Debugging: B3 3.69s, B4 4.58s, both performance1.0. Archive old source and preserve historical links.
- [x] Promote Task3 after user-reported Debugging passes in both GridState and ImageObservation modes; preserve old release and historical links. Detailed boss runtimes and scores were not supplied.

## Final evidence

Candidate4: Task2 `291207ab9396`, Task3 `ff7b47d1f2e6`. Full suite415/415; Omarchy public66/66, notebook22/22, boss12/12 plus notebook4/4. Final eightfold boss stress12/12; sixfold image stress21/22. B4 agent-only reduction30% local /25% Omarchy. See `artifacts/v4-runtime-repair-20261113/README.md` and `summary.json`. No scored submission; platform retest pending.

Platform update: exact Task2 candidate4 promoted after the two user-reported Debugging passes. No scored attempt made. Local synthetic B4 limitation remains recorded; no all22 platform claim.

Boss update: exact Task3 candidate4 promoted after the user reported both input modes passed Debugging. No graded result claimed.
