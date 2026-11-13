# Review and retained failures

A fresh read-only reviewer compared against frozen a8efb9c0. They identified a late-fallback direct-call corner: step_until was clamping away the original dispatch reserve. The fix now inherits the original constructor origin/initial/cutoff and calls backend.step directly, consistent with PlannerAdapter. The reviewer reran adapter/search tests (27 pass), checked reserve propagation to ordering and original deadline monotonicity, and found no remaining blocking correctness defect.

Reviewer requested invalid and expired independent reference-replay tests. Both are included in test_reference_replay_rejection_never_installs_partial_cache, verifying WAIT and empty actions/keys/cursor after either mismatch or timeout. All four dispatch tests pass. Full current repository discovery passes247 tests.

dispatch-red.log and late-fallback-red.log intentionally remain red evidence. full-suite.log belongs to the intermediate imported version and contains the late-fallback failure; full-suite-final.log is the subsequent final-source run,247/247. The frozen a8efb9c0 parent and old artifacts were never edited.
