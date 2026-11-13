# What debugging taught us

These notes were checked against project Codex chat excerpts and saved handoffs. Chat conclusions explain decisions; the original reports and source hashes identify the tested candidates. Imported snapshots are not a fabricated chronological edit history.

| Version | Lesson | Evidence |
|---|---|---|
| Reference-state planner (P1) | State identity must retain inventory and effect details that change future actions. Preserve the accepted baseline. | `artifacts/p1/`, original P1 tests |
| Compact-state planner (P2) | A fallback win can hide an incorrect fast simulation. Compare transitions against the reference simulator. | `artifacts/p2/HANDOFF.md` |
| Boss-search planner (P2-next) | Larger search limits alone did not solve difficult layouts. Ordering and repeated work mattered. | `artifacts/p2-next/handoff.md` |
| Hierarchical planner (P3) | Finding a route is not enough: leave time to execute it. Preserve the original deadline during fallback. | `artifacts/p3/b4-repair/HANDOFF.md` |
| Image-tracking agent (V3) | Board reconstruction and episode wins are separate checks. Stale predictions can make failures persist across turns. | `artifacts/v3/`, V4 diagnostic evidence |
| Image-recovery agent (V4) | Cold imports and repeated simulation consume the same episode budget as planning. NumPy export reduced runtime setup overhead. | `artifacts/v4-numpy-runtime-20261112/HANDOFF.md`, dated V4 repair bundles |
| Boss-mode agent (V4) | A 15-second platform allowance does not extend an internal Task 2 deadline. Select the longer budget explicitly. | `artifacts/v4-boss-budget-20261113/HANDOFF.md` |

The P3 chat records the user saying “all passed” and “freeze it”; this is a reported Debugging outcome, not a graded result. Boss development reports record 12/12 native public wins, 4/4 notebook wins and 4/4 synthetic slowdown wins. Its recorded platform result remains pending. Synthetic slowdown is not measured platform runtime.

## Chat provenance

Relevant messages were reviewed from project sessions on 8–13 November 2026. The [Markdown conversation exports](chats/README.md) retain session IDs, timestamps and source-line references. Original transcripts remain outside this repository. The index records the selected conversations and source digests.

Reviewed session references include:

- `01a1be74-a176-7a32-a315-58a7ef14b91e`: P3 dispatch repair, reported pass and freeze.
- `01a1bed2-0b24-73c3-9c6d-7721707d94bf`: notebook scope, cold-start recovery and NumPy export.
- Boss-budget session excerpts were cross-checked against the current boss handoff and status record.

## Existing evidence discrepancy

The P2-next historical verifier's protected-before receipt expects earlier bytes for `agents/v3_planner.py`, `agents/v3_bridge.py`, `agents/v3_perception.py`, and `agents/v3_types.py`. The pre-cleanup snapshot already differs from those four hashes. The cleanup preserves both sides unchanged and does not certify that receipt. Frozen release hashes remain intact.

Version commits are imports for navigation, not independently qualified historical environments. In particular, the V4 import includes boss packaging tests whose wrapper arrives in the following boss commit. Use the complete final tree for the repository suite.
