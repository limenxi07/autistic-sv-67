# V4 planning/execution deadline repair

Paste the complete `final/task2_agent.py`. SHA256 `32f6fc2832f7b28dec5211665fe45b07fcd8d638920bf2b2665f053c83041a51`; 712,313 bytes. This is normal Agent source, not a diagnostic.

## Platform diagnosis

The4023 episode traces established two different failures. B3 had40 verified actions/cache hits and66 actions remaining at2.624s. Vision and metadata agreed, no repairs occurred, and the3.5s action cutoff was insufficient. B4 returned WAIT on turn0 at2.075s: the2-second reference planning window expired after185 simulations without a verified plan.

## Changes

Search remains limited to the original3.0s initial/3.5s absolute planning budget. A separate image execution cutoff of4.65s permits only a still-matching, already verified suffix after search closes. No new search or suffix rebinding occurs in that late path; discrepancies return WAIT. Structured P3 dispatch retains its original3.5s cutoff. Both budgets share the original construction timestamp and are never renewed.

Reference search still executes the public native engine and round-trips the input through the public converter, preserving inventory ordering. Output conversion reuses only unchanged static blocking/exit/appearance entities while reconstructing dynamic entities with public helpers. Portal/pathfinding worlds use the full public path. Search treats shared static entities as read-only. The complete winning route is still independently replayed using original specialized grid_step before execution. Consecutive replay iterations reuse a previously computed successor key, eliminating duplicate key construction without skipping verification.

## Validation

Exact final source:66/66 public and22/22 pasted-notebook wins, exact initial boards, and effective performance1.0 on Omarchy. Maximum episode0.744s; strict cold completion0.856s; peak RSS102961152bytes (98.19MiB). No budget resets. See summary.json and full reports.

Fourfold elapsed-clock diagnostic:22/22 wins;20/22 match target rewards; longest accelerated episode4.155s. This used the identical executable code immediately before a class-level type annotation; annotation-equivalence.json proves AST equality with that annotation removed. Synthetic slowdown is not platform certification.

Repository suite369/369 passes in137.626s before the annotation-only change; focused final-source tests and Ruff/type checks also pass. Independent review covered late reference dispatch, unchanged-source branching, and154 differential action branches, in addition to repository random/public mechanics tests. Exact-source reproduction and old-source/P3 preservation verified.

## Status

Ready for platform retest. No claim that platform22/22 is confirmed. No new CNN training or final-holdout rerun. The4.65s execution cutoff leaves0.35s below the platform's5s limit; scheduling/setup variation can affect that margin, so actual platform results remain authoritative. Use final/task2_agent.py; earlier folders are preserved experiments.
