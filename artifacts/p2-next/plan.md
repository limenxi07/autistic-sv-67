# P2 measured boss support implementation plan

Goal: reproduce the frozen structured boss baseline, characterize the one-key
resource boundary, and deliver the smallest validated P2 candidate.

Architecture: fork frozen P2a into `agents/p2_next.py`; keep primitive search,
reference fallback and caps. Expose a structural absolute-budget adapter without
vision imports and an explicit 15-second structured entrypoint. No P3 unless
this change leaves a diagnosed deficit. User's milestone instructions authorize
inline execution; files are isolated by the requested ownership names because
the active V3 work shares this untracked checkout. No commits or V3 edits.

Constraints: no engine implementation inspection; 150 turns, one core/thread,
1.5 GB, source <2,000,000 bytes. Tasks 1/2 retain 3.0/3.5 planning under 5 s;
Task 3 explicitly uses 11/12 planning under 15 s. Never extend a caller deadline.

Review focus: resource order with sparse IDs; key consumption with many items;
stacked effects and expiry; stale cached actions after cutoff; fallback using a
late constructor. Tests exercise each before acceptance.

- [x] Measurement: create `tools/benchmark_boss.py`; fresh workers, watchdogs,
  setup/evaluation/cold wall, nested fallback statistics, independent trace
  replay. Frozen exact-source acceptance separate from instrumented first
  reference-verified-win diagnostics. Audit all rejection sites and traces.
- [x] Schema: failing tests in `tests/test_p2_next.py`, then allow one-key/many
  item states with order canonicalization; retain ambiguous guards. New
  differential corpus in `tests/p2_next_fixtures.py` and runner under `tools/`.
- [x] Deadlines: failing tests, then `PlannerAdapter(budget, mode='p2a',
  reference_initial_deadline=None)` and `Task3Agent`. Preserve verified
  incumbent; all planning, verification, recovery and dispatch share cutoff.
- [x] Validation: >=100,000 all-action comparisons across >=1,000 layouts,
  public22 x3, generated regressions, unfamiliar boss-like cases and C0/C1 x3
  sequentially on Omarchy CPU 0. Record raw generated rewards; compare paired
  legacy traces; run unchanged V3 oracle regression. Freeze hashes before
  acceptance. Benefit gate: fix C1 loss and preserve paired public rewards and
  C0 best budget-only -68; otherwise label experimental and diagnose.
- [x] Handoff: report exact hashes, coverage failures, reproduction commands,
  latency distributions and conditional V3 adapter interface. No integration or
  platform submission.
