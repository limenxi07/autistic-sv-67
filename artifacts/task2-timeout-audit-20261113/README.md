# Task 2 timeout audit

Production source is unchanged: `submissions/task2_agent.py`, SHA256 `0b61f5c7c594541b12656995aa2389642ce37a48ad9f5a2bc3544360c45d92d5`.

## Next platform check: Debugging only

Paste the complete `task2_agent_debug_only.py` into the Task 2 Agent cell and use **Debugging**, not a scored submission. It contains the exact production source followed by instrumentation. Do not append it to an existing Agent or submit it for grading. Restore the production file before any eventual submission.

An intentional error beginning `TASK2_TIMING_0b61` is the expected result. Copy that error back, preferably all reported cases. It reports source identity, observation dimensions/configuration, an observed-board signature, planning/perception/replay timing, time between agent calls, remaining route length, and fallback reason. A custom exception bypasses the agent's normal recovery handler. Short routes report before their predicted terminal action; longer routes report after 40 actions or at the first instrumented checkpoint after 2.5 seconds. A blocking call that itself exceeds the hard timeout can still prevent a report.

Diagnostic error/zero-score outcomes are intentional and are not performance results. Stage times are inclusive and overlap. Signature matching identifies a local observed layout if available; it does not prove a mapping of grader test-set numbers.

## Findings

- Fresh-process local macOS runs, default thread settings, 22 public boards × 3 seeds: **66/66 wins and full scores**. The complete loop includes reset, construction, per-turn specialized history conversion, environment stepping and image rendering. Imports/setup are outside episode timing; source loading is reported separately.
- Slowest native case: B4, 0.936 seconds for 130 actions. Agent 0.511 seconds; history conversion 0.150 seconds; environment 0.272 seconds. The previous qualification loop omitted full history recording.
- Synthetic sixfold elapsed clock plus at least 14ms per environment step, full history: **21/22 wins**, B4 times out at 5.010 seconds. First cutoff WAIT occurs at turn 105, about 4.689 seconds.
- Same synthetic settings without history: **21/22 wins**, B4 reaches its cutoff at turn 129, about 4.697 seconds, then WAITs until the turn limit. This is a loss before five seconds, not a hard-timeout reproduction.
- These results establish a local timing vulnerability. They do **not** establish which board the grader used or the cause of its timeout. No production change is justified solely by this artificial stress.
- B4's separate cProfile run locates substantial cost in winning replay, state conversion and rendering. Its instrumented timings are diagnostic, not acceptance measurements.

## Diagnostic verification

All 22 boards emitted the expected diagnostic in both module and notebook-function scope (**44 checks**), with cache hits, replay and CNN timing, and no captured agent exceptions. Both controlled-clock early-report guard checks passed. Longest report was 2,021 characters. See `diagnostic-verification.json`, `guard-verification.json`, and `binding.json`.

## Reproduce from repository root

```sh
.venv/bin/python artifacts/task2-timeout-audit-20261113/profile_public.py --output /tmp/task2-native.json
.venv/bin/python artifacts/task2-timeout-audit-20261113/profile_public.py --output /tmp/task2-stress.json --scale 6 --env-floor .014 --seeds 42
.venv/bin/python artifacts/task2-timeout-audit-20261113/verify_diagnostic.py
.venv/bin/python artifacts/task2-timeout-audit-20261113/verify_guards.py
```

No scored submission was made. The user's submission attempts are limited; Debugging runs are unlimited. Obtain platform diagnostic evidence before selecting a production fix or recommending another submission.

## Platform Debugging evidence received

The user supplied all 22 `TASK2_TIMING_0b61` reports. Selected fields are preserved in `platform-debugging-summary.json` with their provenance stated explicitly. All visible-kind signatures match local diagnostic signatures; this is not a comparison of full semantics or image pixels. All reports have zero recorded discrepancies, repairs, recoveries, transition-cache misses and caught errors. Seventeen stop at a predicted terminal action; five stop early by design. None is a scored completion result.

B4 has completed 35 dispatched actions at 2.504827 seconds with 95 planned actions left. First-action time is 1.400808 seconds. Measured later-agent cost averages about 14.82ms per call, and outside-agent gaps average about 17.14ms. Holding those costs constant estimates completion near 5.54 seconds, past the 4.65-second execution cutoff. B3 projects near 4.37 seconds. B5/B6/B10 project near 3.66/3.02/3.37 seconds. These linear projections are estimates; later frames and platform scheduling can change costs. Inclusive stage times overlap and were not added together.

For a confirming **Debugging-only** run, use the complete `task2_end_debug_only.py`. It keeps the production prefix unchanged, removes per-stage timing wrappers, and continues until the predicted final action, first actual fallback, or a near-hard-limit checkpoint. Expected intentional errors begin `TASK2_END_0b61`. Send B4 and B3 results first; no need to resend every traceback. A single long call or external gap can still prevent output before the platform hard timeout. Do not use this copy for a scored submission or for bosses.

The end probe passed 44 checks across all 22 boards in module and notebook-function contexts. Each reported final action was independently applied to the real environment and produced the reported winning score. Two controlled-clock checks confirm it reports both an actual execution-cutoff WAIT and a near-hard-limit entry. See `end-probe-verification.json`, `end-probe-verification.log` and `end-probe-binding.json`. Production Task 2 and Task 3 hashes remain unchanged.

## Platform end-probe results received

The user supplied B3 and B4 results; selected fields are in `platform-end-summary.json`.

- B3 prepared its 106th and predicted winning action at 4.347926 seconds, score -315, no remaining plan. This leaves about 0.302 seconds before the internal execution cutoff. The diagnostic intercepts the final action, so this is not a completed graded win.
- B4 reached observed turn 95 (95 actions already applied), then returned WAIT at 4.682607 seconds with reason `unresolved_perception`. It recorded zero visual/metadata discrepancies, repairs and recoveries. The planner and hypotheses were cleared. Its original verified route has 130 actions, so approximately 35 useful actions remain; the 96th counted action is the fallback WAIT.
- Source inspection shows `reconcile()` returns an empty hypothesis tuple if the deadline expires. `_step_image()` checks for empty hypotheses before its subsequent cutoff check and therefore labels this `unresolved_perception`. `_emit_action()` then clears the planner. This label alone is not evidence of a misclassified image.
- `reproduce_expiry_label.py` reproduces that exact path on the current unchanged source at B4 turn 95 using a controlled clock. The actual observation check is consistent with no mismatches; crossing the deadline makes the agent WAIT, report unresolved perception and clear the planner. See `expiry-label-reproduction.json`.

The evidence now establishes the platform execution-budget failure on B4 in Debugging. It still does not identify the grader's numbered set 4. The remaining work is a measured runtime reduction; merely renaming the fallback reason cannot fix it, and extending 4.65 to just below 5 seconds would provide only a fraction of the time required for 35 moves. No further diagnostic run is needed to identify this failure mechanism. Production optimization and platform acceptance remain outstanding.
