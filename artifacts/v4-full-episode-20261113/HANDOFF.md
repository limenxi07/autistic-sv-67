# V4 full-episode cost repair

Paste the complete `submission/task2_agent.py` into the Agent cell. SHA256: `eb417a551f9c448808d06889e283a571f332b42bed220c9e423f09c0b39daa21`. Size:715,470 bytes. This is normal Agent source, not a diagnostic.

## Evidence and changes

The user reported18/22 passes for preceding source32f6fc2832f7, with B3,B4,B5,B10 all reaching the platform's5-second timeout. Prior platform traces established correct B3 perception/dispatch with a long remaining route, and a costly B4 reference search. Full local episode profiling of32f6fc showed repeated public GridState conversion during winning replay, duplicate cached-dispatch key reads, and repeated tile resize/normalization.

This candidate reduces those costs without extending deadlines:

- Winning routes are replayed separately using the public native engine and the existing optimized converter. Every dynamic entity is rebuilt; only unchanged plain blocking/exit scenery is shared read-only. Every expected route key and final win is still checked. The guarded transition cache still rejects changed source/successor states.
- Full-cell probabilities are cached by exact raw pixels, shape and dtype before resizing/normalization. Cache resets on configure; returned arrays do not alias cache entries. Limits:1,024 entries and8MiB of raw pixels. Changed pixels still trigger classification; no frame or metadata check is skipped.
- Cached action dispatch reads the current state key once, retaining deadline and mismatch checks.
- V4 state keys avoid recursive serialization for exact plain-wall/exit records and avoid sorting singleton cells. Output is identical to the full reference key; modified markers, extra fields, nested resources and dynamic entities use the original serializer. Frozen P3 functions and bytes remain unchanged.

Search budgets remain3.0s initial/3.5s absolute; image execution cutoff remains4.65s, all measured from the original construction timestamp. CNN weights, calibration, thresholds and structured P3 dispatch are unchanged.

Verification boundary: reference search and route replay now share the optimized converter. Independent differential tests against original specialized public grid_step validate that converter; runtime route replay cannot catch a defect common to both conversions. No environment truth or public-level solution lookup enters the submitted source.

## Validation

Exact submission on Omarchy, fresh single-core workers, one library thread, Performance EPP:

| Batch | Wins | Exact boards | Performance1.0 | Max episode | Max strict cold |
|---|---:|---:|---:|---:|---:|
| Public module |66/66|66/66|66/66|0.600s|0.711s|
| Function-scoped pasted notebook |22/22|22/22|22/22|0.593s|0.706s|

Peak RAM100,102,144bytes (95.46MiB). No budget resets. Reports:submission-public-66.json, submission-notebook-22.json, submission-summary.json. Recorded remote exit0; all evidence retrieved.

Synthetic fivefold elapsed clock plus a minimum14ms virtual environment cost per action: final candidate22/22wins,20/22target rewards, maximum4.498s. Previous source21/22wins,19/22target rewards; B4 exceeded5s. The four reported problem levels took4.136→3.208s, >5→4.498s,3.136→2.372s,2.625→2.106s. These are diagnostic comparisons, not Coursemology timing or guaranteed platform headroom. Earlier sixfold experiments are retained and were not all successful.

Full repository suite377/377 passed in142.724s; focused37 tests and corrected cost-counter tests passed; Ruff and ty checks passed. Independent review reported22,198 key/transition comparisons across22 public levels and8 random trajectories each. See final-all-tests.log for the complete repository-suite result and review.md for review scope. Cost regressions were observed failing before implementation. Final packaging reproduced the candidate bytes; protected-hashes.json verifies the previous submission and frozen P3 remain intact.

## Status

Ready for platform retest, not a claim of platform22/22. No retraining, final-holdout rerun, baseline overwrite, or promotion under the original qualification gate. Use only submission/task2_agent.py from this folder; candidate/reviewed/final/key-cost are preserved intermediate packaging stages.
