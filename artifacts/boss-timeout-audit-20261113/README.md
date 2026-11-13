# Boss timeout audit

Exact boss source: `submissions/task3_agent.py`, SHA256 `8b8f72d79335ba31cf504ae8d1cf33954946905bcd45c6dfecd05f2308323104`. Production files are unchanged; no submission was made.

## Result

The same class of deadline failure can affect the current boss agent. Its GridState path stops all actions at the 12-second planning cutoff, even when it has a verified winning route remaining. The image path can dispatch a verified route until 14.65 seconds. Both use an 11-second initial planning window, measured from original construction.

Fresh-process local macOS checks include reset, agent construction, full per-turn history conversion, environment stepping and rendering/conversion. Imports and source loading are outside episode timing. Both public bosses, both input modes, and three seeds (42, 123, 211) were evaluated. Each run's full-score threshold is 0.85 and maximum score is 2; all three native performances in each board/mode group exceed that threshold, so their group averages do too.

| Batch | Wins within 15 seconds | Outcome |
| --- | --- | --- |
| Native, 3 seeds | 12/12 | Full credit in every run; maximum 1.625 seconds |
| Sixfold clock, minimum 14ms environment step | 4/4 | Full credit in every run; maximum 10.227 simulated seconds |
| Eightfold clock, minimum 14ms environment step, 3 seeds | 9/12 | The Heist GridState failed all three; all image runs passed |
| Focused repeat of eightfold Heist GridState | 0/3 | Confirmed cutoff WAIT with 27–32 planned actions remaining |

Focused repeat first WAITs occurred at simulated 12.014, 12.022 and 12.018 seconds. Each reports `structured_cutoff` and a 12.0-second cutoff. These runs ended at the turn limit with reward -450. The earlier eightfold batch also includes one loss shortly after WAIT began, consistent with waiting in an unsafe location. These are game losses, not exceptions or evaluator hard timeouts. The first verified win was found around 7.3–7.5 simulated seconds, while first-action dispatch occurred around 11.2–11.3 seconds. Search refinement therefore consumed substantial time before route execution.

Native Heist performance was 0.9367 (GridState) and 0.9114 (image), both above 0.85. Native C0 performance was 1.0 and 0.9412 respectively.

## Task 2 source is not the boss source

As a control, the Task 2 agent was also run on both bosses and both modes under fourfold synthetic slowdown. It won 2/4: Heist failed in both modes because the shorter Task 2 search budget expired before any win was found. Merely selecting a 15-second evaluator does not extend an agent's internal budget. Use the separate Task 3 source for boss evaluation; the Task 2 diagnostic copy is not a boss diagnostic.

## Boundary and next step

The slowdown is artificial and runs on macOS, not the grader. It proves a reproducible local failure mechanism but does not prove that the platform has encountered it. An appropriate candidate repair would preserve the planning deadline while permitting an already verified, matching GridState suffix to execute during the remaining boss time, and/or reduce refinement after a qualifying route is found. Such a repair needs its own tests and qualification; this audit makes no production change. Use unlimited Debugging before consuming a scored attempt.

## Reproduce

```sh
.venv/bin/python artifacts/boss-timeout-audit-20261113/profile_boss.py --output /tmp/boss-native.json
.venv/bin/python artifacts/boss-timeout-audit-20261113/profile_boss.py --output /tmp/boss-stress.json --scale 8 --env-floor .014
.venv/bin/python artifacts/boss-timeout-audit-20261113/profile_boss.py --label 'C1 The Heist' --mode gridstate --output /tmp/heist-cutoff.json --scale 8 --env-floor .014
```

Full per-turn evidence, first-WAIT route counts, source hashes and dependency provenance are in the JSON reports. `summary.json` provides batch totals. Timing varies between runs; the focused repeat records its own evidence rather than reusing earlier timings.
