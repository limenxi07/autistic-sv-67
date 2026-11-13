# Boss-search planner (P2-next): initial work plan

Session: `01a1bdfd-d20d-7b93-bbed-ec4ad01b556e`  
Started: 2026-11-09T01:48:10.678Z  
Export: 1 visible message. See [conversation index](README.md).

## enxi — 2026-11-09T01:49:56.089Z
<!-- source-line: 13 -->

Work on Compact-state planner (P2) latency and structured Task 3 boss support alongside the ongoing Image-tracking agent (V3) vision work.

Compute: ssh enxi@omarchy — Ultra 9 275HX; benchmark sequentially on one P-core.

Start with measurement and targeted Compact-state planner (P2) fixes. Introduce Hierarchical planner (P3) only if measured failures justify it.

Verified starting facts:
- Compact-state planner (P2)’s default search deadlines are 3.0/3.5 seconds.
- Task 1/2 allow 5 seconds per complete episode; Task 3 allows 15 seconds.
- All tasks allow 150 turns, 1.5 GB memory, <2,000,000-byte source, one CPU core and no runtime GPU.
- Both public bosses currently fail P2.encode with “Large keyed inventory iteration history.”
- Each boss has ONE key. The guard counts keys + coins + gems, rather than keys alone.
- Image-tracking agent (V3)’s oracle bridge currently wins all 22 gameplay layouts with paired P2a rewards. Preserve that integration boundary.

Ownership:
- Create new candidate files, tests and artifacts, for example agents/boss_p2_next.py, tools/benchmark_boss.py and artifacts/p2-next/.
- Do not modify agents/compact_p2.py, frozen P1/P2 artifacts, existing evidence, or active v3_* files.
- Keep boss work separate from Image-tracking agent (V3)’s initial release.

Milestone 1 — structured boss baseline and coverage audit:
1. Benchmark C0 and C1 using the exact frozen P2a source.
2. Compare existing deadlines against explicit Task 3 budgets.
3. Record first REFERENCE-VERIFIED win time, final reward, complete episode time, peak RSS, fallback use, stop reasons and source hashes.
4. Audit every fast-path rejection, including resource order, overlaps, effect counters and state/record caps.
5. Use fresh workers, hard watchdogs, one CPU/core and one library thread. Separate setup, evaluation and conservative cold-start timing.
Milestone 2 — validated Compact-state planner (P2) improvements:
- Investigate supporting one-key/many-item states before attempting general large keyed inventories.
- Do not simply remove the inventory guard. Establish which ordering distinctions affect future transitions.
- Differential-test reachable states and every action against public grid_step, including sparse/permuted IDs, mixed inventories, key consumption, stacked effects, expiry, boxes and overlaps.
- Retain reference fallback for unsupported cases.
- Accept caller-supplied absolute deadlines. Preserve a verified winning incumbent and reserve time for execution, verification and recovery.
- Keep Task 1/2’s five-second behavior separate from an explicit Task 3 entrypoint; do not infer the time allowance from board size.
