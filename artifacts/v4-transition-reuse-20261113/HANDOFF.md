# V4 long-episode repair — submission

Paste the complete `submission/task2_agent.py`. SHA256 `4023b9ae0d50c9820733031a2d7453ecd687ee72720b9876d798d64211542d08`; 709,320 bytes. This is normal runnable Agent source.

Previous release `5f260f30163bd` was user-reported 17/22, failing B3/B4/B5/B6/B10. Their -450 losses near the time limit were consistent with the action cutoff. A new platform per-turn trace was not obtained; platform retest remains pending.

## Changes

B3 profiling showed 212 public simulator transitions for a 106-action route: full winning-plan verification followed by repeated simulation during dispatch. A bounded cache now retains at most 256 edges from a completely verified winning replay. Reuse requires the identical source object and matching full source and successor state keys. Changed state/action falls back to public grid_step. The existing image/metadata and all-hypothesis safety checks remain active. Failed/expired replay never installs edges.

B4 retains its multiple-key/inventory fallback guard. Reference SEARCH uses the same public transition engine without rebuilding specialized static entities after every move. Actor and carried/effect entities retain public specialization, including collectible-marker restoration for carried keys. Generic empty nested-list attributes are normalized. Winning-plan verification still independently uses the original specialized public simulator before dispatch/cache installation.

Fast refinement remains bounded: 25ms after first verified win normally, 150ms when coins exist so Optional Coin Path and Vault retain target rewards. The ordering deadline is reduced with the outer search deadline. Reference refinement remains25ms. Original absolute3.0/3.5s budgets remain unchanged; structured P3 unchanged.

## Exact submission evidence

Omarchy public66: 66/66 wins, exact initial boards and effective performance1.0. Pasted notebook22:22/22 wins, exact initial boards and effective performance1.0. Maximum strict cold completion1.843s. Full reports:submission-public-66.json,submission-notebook-22.json andsubmission-summary.json.

119 V4 tests pass after the final coin-window adjustment (35.219s). Full366-test repository suite passed immediately before that one-line policy adjustment (136.842s); both logs retained. Ruff/type checks pass. Exact-source reproduction checked.

Independent review found no safety blocker. Differential tests cover random actions on all22 public levels and focused pickups, effects, doors, boxes and damage. Reviewer separately checked575 transitions on48 mechanic/generated witness paths. Source/successor mutation invalidation, reference-owner dispatch and rejected-replay behavior are tested.

## Limits

Fourfold elapsed-clock diagnostic: 21/22 wins; failures: B4 Measure Twice and Cut Once. This is synthetic local stress, not the platform. Do not claim all22 platform tests pass until user retests.

No model training or final-holdout rerun. Prior frozen sources unchanged. No original final-gate promotion. Earlier `candidate`, `release` and `final` folders are experiments; use only `submission/task2_agent.py` above.
