# V4 vision and recovery design

Status: proposed, 9 November 2026. This request authorizes planning. No V4 implementation, training, remote jobs, or submissions have started. Numerical V4 targets below are proposed acceptance criteria, not measured results.

## Intent and scope

Repair the weaknesses exposed by V3's failed image qualification while integrating the working P3 supplied by the user at `artifacts/p3/frozen/task1_agent.py`. V4 will use that exact frozen backend through its existing planner adapter. Establish the backend integration separately before tracking/perception changes; P3 is not assumed to fix image parsing.

Deliver a reproducible successor with better artwork discrimination, recovery that stays synchronized after every returned action, and evidence that distinguishes repaired known failures from new generalization. Keep the independently usable Task 1 submission intact.

## Current evidence

The planning audit checked the current runtime files against `artifacts/v3/qualification-reviewed/acceptance-manifest.json`: all eight runtime source hashes match. The frozen executable is 668,698 bytes, SHA256 `71c0fd0e84d55dd76dc785ce8aba298766f6e1ca709f8f5c1d9a87e131f9ef1d`. These are current file checks; episode measurements below are saved V3 results, not reruns during planning.

| V3 measurement | Saved result | Implication for V4 |
|---|---|---|
| Public images | 66/66 wins and exact initial boards | Preserve this coverage and reward |
| Final-v5 images | 295/300 wins; 288/300 exact initial boards | Failed the 297/300 exact-board gate |
| Initial errors | 15 gem→coin, 7 gem→ghost, 1 box→wall | Train/evaluate by artwork group and class, not aggregate accuracy alone |
| Gem recall | 98/120, 81.67% | A high win rate hides an objective-critical weakness |
| Correct-board loss | 44 tracked actions, 106 repairs, zero recoveries, final `unsupported_observation:ValueError` | Trace the first divergence; final exception name is insufficient |
| Other four losses | Initially wrong boards, final `cutoff` | Attribute perception, repair, and planning time separately |
| Structured/oracle | 266/266 paired structured rewards; 22/22 paired oracle rewards | Preserve parent behavior and bridge equivalence |
| Maximum resource use | 3.8028 s episode, 3.9287 s strict cold, 332,054,528 bytes RSS | Extra inference must earn its CPU cost |

Primary evidence: `artifacts/v3/HANDOFF.md`, `qualification-reviewed/final-300.json`, `summary.json`, `qualification.json`, and `acceptance-manifest.json`.

The exact-board loss is `combined/7297a35c9f88/6216229300`, layout seed `208801192575`. Reproduce from the row's complete `job`, not from this display label alone. All five lost jobs and all twelve initially incorrect jobs must be derived from the saved report, deduplicated, and bound to the original decoded pixels.

### Supplied P3 baseline

The frozen source is 96,112 bytes, SHA256 `d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`; `agents/p3.py` currently matches it byte-for-byte. The freeze record reports 247 tests, 326/326 local episodes, 257,467 differential comparisons, and user-reported platform Debugging “all passed” with no test count or graded claim. These saved results were inspected, not rerun during planning. `frozen/platform-validation.json` supersedes the historical pending-platform language in `b4-repair/HANDOFF.md` and `qualification-local.json`.

The 326 local episodes use two entrypoints: public 66 + generated 200 use default `Agent`; stress 36 + retired holdout 18 + boss 6 use `Task3Agent`. Only the 266 default-entrypoint rows establish the corresponding Task 1/2 baseline. The other 60 rows are longer-budget structured evidence, not five-second image qualification.

P3 exposes `PlannerAdapter(budget, mode="p2a", reference_initial_deadline=None)` and preserves public-ID suffix rebinding. Its B4 repair reserves time for dispatch after finding a reference incumbent and independently verifies the route; it does not increase deadlines. Existing adapter/dispatch tests cover the shared origin, late fallback, failed replay, and re-identification. V4 must retain these properties with perception included in the original clock. Its image wrapper passes `reference_initial_deadline=min(budget.initial_deadline, budget.started_at + 2.0)`. Never select P3's explicit 11/12-second `Task3Agent` for an image or default structured episode.

## What the source establishes, and what remains a hypothesis

1. `agents/v3_perception.py:classify_initial` only applies its extra 5% inset view when the full-cell winner is gem or shield. It does not address a full-cell winner of coin or ghost. Extending that pair blindly is not evidence of improved discrimination; V3 already recorded damage to boxes/exits from indiscriminate inset crops.
2. V3's selected temperature is 0.25, selected on an almost perfect validation set. Positive scalar temperature changes confidence, not the winning class. Recalibration alone cannot fix the 23 wrong labels. Whether confidence sharpening worsens recovery decisions must be measured on grouped development data.
3. Several early returns and the exception handler in `agents/v3.py` emit WAIT without recording it or advancing/invalidation of `_hypotheses`. `reconstruct` requires same-turn history. This is a concrete synchronization hazard. It is a plausible explanation for the repeated ValueError, not yet a traced cause of that saved episode.
4. The harness records final reason/counters, but lacks the first exception context and per-stage timeline needed to attribute all five losses. Diagnostic trace and uninstrumented timing runs must be separate.
5. The finite public artwork inventory is already partitioned into 198 training, 65 validation, and 70 final groups; V3 evaluated the final partition. New seeds do not make those groups unseen again. Historical exposure must survive every re-split.

## Chosen approach and alternatives

**Recommended:** integrate and verify the frozen P3 adapter with the existing V3 model/tracker first, repair tracking second, then train the same small CNN with group-balanced data and explicit difficult-class coverage; evaluate bounded extra views as an ablation. Keep P3 fixed throughout the perception/tracking comparison. This isolates the backend change and the two image failures while retaining a compact CPU deployment.

An inference-only extension of the gem/shield rule is a cheap diagnostic candidate, but insufficient as the default release strategy: it misses the box error and may retain confident artwork mistakes. A multi-model ensemble, larger model, learned policy, or further planner redesign adds runtime and attribution risk without addressing the demonstrated tracking contract. Defer those until the bounded comparison shows a specific need.

The user has supplied the completed P3 artifact for this plan. Other architecture and training choices remain recommendations. Providing the artifact updates the plan's dependency; it does not start implementation or compute jobs.

## Global constraints

- Tasks 1/2: 150 turns, 5 seconds for the whole episode, 1.5 GB memory, one CPU core, no GPU at submission time.
- Submission source must be below 2,000,000 bytes, including embedded model and loader. Target at most 1,600,000 bytes to leave room for fixes.
- Retain Python 3.10 grammar and prelude/no-module-registration loader checks. Deployment uses the repository's pinned CPU Torch 2.7.0 and NumPy/Pillow environment.
- Use public assets, assignment/tutorial, public constructors and black-box API observations. Do not inspect engine implementation source. No network, external asset reads, or model downloads in the submitted agent.
- Preserve all existing P1/P2/P3/V3 source and evidence bytes. Add V4 runtime files and write new results only beneath `artifacts/v4/`.
- Backend is `artifacts/p3/frozen/task1_agent.py`, SHA256 `d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`. Vendor these exact bytes into V4; preserve the frozen artifact. P2-next is retained only as the historical V3 baseline.
- One clock: initial planning through t0+3.00 s, verification/repair through t0+3.50 s; initial reference refinement is capped at t0+2.00 s. Model loading/parsing consumes this budget. Target every complete qualification episode <4.25 s and <1,000,000,000 bytes RSS; strict cold import-to-terminal must be <5 s.
- Preserve the current CPU dependency lock. Any GPU training environment is separate, hardware-verified, and recorded. Planning does not authorize starting compute jobs.
- Archive generation/training/export code, configurations, checkpoints, manifests, datasets, qualification reports, and exact source hashes. No platform submission is included.
- The checkout contains untracked required inputs and unrelated changes. Do not broad-stage, reset, clean, or use a HEAD-only worktree that omits those inputs. Scope snapshots/checkpoints to V4 and record parent hashes.

## Runtime contracts

### A. Observe, reconcile, plan, emit

Use one action emission path for every image-mode return, including WAIT and supported exception handling. Keep a last-emission record with observation turn (if valid), action, reason, and whether a next-turn prediction exists. Distinguish actions emitted from plans found and recoveries confirmed.

Compute reconciliation and the next prediction transactionally. Commit tracker history only when the action actually returned is known. Never leave a half-updated planner/tracker pair after an exception. Before every reconstruction assert that the prediction turn equals the observation turn. Invalidate the cached suffix on any contradiction.

For an ordinary fallback WAIT with a usable current conditional state and remaining budget, predict the actual WAIT through public `grid_step`, retaining all existing uncertainty. Do not mark that prediction observed. If current turn, geometry, history, or time is insufficient, explicitly break continuity, clear executable cached actions, and retain no stale state as a same-turn prediction. After cutoff, emission must be cheap and must not launch parsing/replay.

A later valid observation can establish a new current hypothesis only from visible pixels, metadata, and valid historical layers. Unknown covered layers stay unresolved. Do not solve desynchronization by overwriting `state.turn`, inventing hidden objects, dropping timers, or copying expected labels into observations. Terminal input produces a terminal WAIT without advancing the world.

A recovery counts only when a later independent observation corroborates the repaired prediction. Known engine/resource ID binding and key/effect ordering remain as in the characterized V3 bridge. Duplicate inert-wall normalization remains unchanged.

Use the same history-aware parsing semantics for `parse` and execution. `parse` is read-only: an actor over a previously known exit must preserve that exit; an unknown covered exit must stay unresolved. Fix V3's deferred covered-exit visualization limitation alongside this contract.

### B. Perception and uncertainty

Retain the 104,526-parameter, RGB 48×48 CellCNN as the first training architecture. Balance samples by class and then by artwork group within class; include backgrounds in provenance. Add renderer-realistic examples of gem/coin/ghost/shield distinctions and box/wall borders, plus reachable actor/item, box/item, lava, wall, and exit overlaps. Keep initial-board classification and temporal actor localization as separately reported tasks.

For training composites, save complete semantic layers separately from the single visible target. Use V3's characterized visible priority: human, box, wall, locked, opened, gem, coin, key, boots, ghost, shield, lava, exit, then floor. Thus actor-over-gem trains `human`, and box-over-item trains `box`; it does not teach the hidden class as visible. Add black-box rendered fixtures confirming this target for each included composite family. Exclude/report a composite whose dominant visible target cannot be established; do not silently discard all overlaps as the V3 board loader does.

Evaluate two preprocessing policies: full-cell only, and full-cell plus a batched 5% inset view for cells whose full-cell top two include gem/coin/ghost/shield. The latter trigger applies even to high-confidence coin/ghost winners. Explicitly skip this extra view when the full-cell winner is floor, wall, box, exit, locked, opened, or lava, even if a rival class is second. These protected winners keep their full-cell probabilities; the box→wall error must be fixed through training. This protection applies to the initial-classification inset policy, not the independent temporal actor-localization path. The old gem/shield policy is a baseline ablation, not silently inherited as the new default.

For the bounded extra-view candidate, average full/inset calibrated probabilities equally for selected cells; retain view disagreement as explicit evidence. Confidence thresholds must be selected jointly with the fusion policy on development boards and episodes. Do not multiply view probabilities as if views were independent. A temperature adjustment or a solvable candidate map cannot establish the true class.

Keep at most four board hypotheses spanning at most two ambiguous cells. A view disagreement introduces ambiguity even when the fused winner is confident. Include the two view winners when they disagree, rather than discarding one behind a fused top-two ranking. More disagreements remain unresolved; do not increase the cap to hide weak perception. Common actions must pass public-simulator checks across every retained hypothesis; predicted wins and reward pruning remain conditional on the parse.

### C. Bounded learning and evidence

Retire all of V3 final-v5 to development/regression now that its outcomes inform V4. Preserve the original files. Before training, record group lineage (`used_for_fit`, `used_for_selection`, `evaluated_before_v4`, source manifests) and audit every public group for prior exposure. Never erase historical exposure by changing a seed or split name.

Use three deterministic group folds stratified by class, with floor groups separated as well. Train each fold model from random initialization on the other two folds; reserve the held fold for out-of-fold logits. V3 checkpoints cannot initialize fold models, because they have already seen some held-fold groups. Generate separate layout sets for training, out-of-fold calibration/selection, and the final fresh-layout test. These fold metrics are development evidence on historically exposed public artwork, not a blind private-style claim.

Select one epoch `e*` (1–20) by aggregate held-fold exact-board accuracy, then worst-class recall, then lower NLL, then earlier epoch. Generate the pooled calibration logits from all three fold checkpoints at that same epoch. Fit production from scratch on all three development folds, excluding every untouched reserve, for exactly `e*` epochs and use its final checkpoint. Do not choose its checkpoint from final outcomes. If separate development-layout calibration transfer fails, stop this four-run batch with its evidence rather than silently selecting a different production epoch.

Initial training budget: three fold runs and one final production fit, each at most 20 epochs, using the same architecture. Start each run with 200,000 balanced augmented crops plus 600 real initial boards and up to 600 reachable frames (derived from at least 100 training layouts, at most six selected frames per layout). Select reachable frames to cover the overlaps above; report actual counts and class support. Stop to diagnose if these four runs do not suffice; do not launch an open-ended architecture sweep. Compare both view policies from the same logits/checkpoints rather than multiplying training runs.

On development data, choose view policy and calibration by exact-board accuracy, worst-class recall, high-confidence error count, downstream wins, then runtime. Fit temperature on out-of-fold logits using the existing 0.25–8 grid including 1.0; evaluate confidence thresholds {0.85, 0.90, 0.95} and margins {0.10, 0.20, 0.30}. Record the selected scalar, thresholds, disagreement rule, and corpus hashes in the exported model report. Validate transfer of calibration to the production model on separate development layouts; do not recalibrate on final outcomes.

An untouched artwork gate is available only if the audit finds sufficient groups that have never entered fitting, selection, or evaluated model outcomes. Do not assume they exist. If absent, use fresh layouts with explicitly exposed public artwork, report `unseen_art_status=unproven`, and allow a **regression-qualified V4** designation only. This cannot be relabeled as passing V3's original unseen-art promotion gate. A later truly independent artwork corpus is an additional qualification, not permission to consume graded submissions.

## Qualification and stop rules

Evaluate the four combinations of old/new perception and old/new tracking on development data before the final freeze, with the exact P3 backend and identical clocks in all four. The old/old corner is a separately frozen V3-on-P3 control built before image fixes. Keep the original frozen V3/P2-next as a fifth historical control to isolate the planner change. This reveals whether recovery improvements conceal initial errors or additional inference consumes repair time. Only trace replay is instrumented; performance gates use fresh uninstrumented standalone workers.

| Gate | Proposed requirement |
|---|---|
| V3-v5 regression | 300/300 exact initial boards and wins; all five old losses repaired; no lost per-case reward on V3's 295 winning rows |
| Public | 66/66 exact initial boards and wins at seeds 2, 43, 20261109; no per-case reward regression from V3 |
| Structured parent | 266/266 wins and equal per-case rewards to frozen P3's default Agent; preserve V3/P2-next winning rewards on the same cases |
| Oracle bridge | 22/22 wins with rewards paired to P3 PlannerAdapter on true structured states under the same image-adapter budget, including the 2 s reference cap; all-action successor fixtures still agree |
| Fresh-layout final | 100 disjoint layouts × 3 appearance assignments, ten families × 30; >=297/300 exact initial boards and >=285/300 wins |
| Critical classes | >=99% recall separately for gem, coin, ghost, box, wall; all 14 classes reported with denominator and per-artwork-group results |
| Coverage | At least 100 instances each for gem, coin, ghost, box, wall across the 300 final boards; choose legal case contents before candidate outcomes, retain replayable solution witnesses |
| Temporal reliability | No unhandled exceptions, no stale-turn WAIT loop, no budget reset, no known failure replay left unattributed |
| Resources/package | Every row meets global limits, exact embedded candidate hashes, no runtime overrides, clean loader/file/network-denial checks |

The fresh-layout final may use exposed public art and then proves only new-layout performance. If untouched art exists, record a separate strict unseen-art result with group-disjoint lineage; require the same board/win/resource targets for that result. A class with missing/insufficient support fails coverage rather than passing by omission.

The new coverage minimum means the unchanged V3 ten-family generator is not automatically sufficient. Add V4-only legal variants for objective-neutral collectibles/powerups/boxes where needed, record all layout hashes, and replay witnesses through the public API before freezing. Never change corpus contents after seeing final predictions.

Any code/model/policy change after unlocking final results retires that corpus to development; a new final test requires a new freeze and an honest exposure label. Failures remain failures even if all unit tests pass. If the P3 bridge fails, tracking cannot be repaired from observable evidence, or the four-run comparison misses its gates, stop with the failing artifact and diagnosis. Do not switch backends, use Task3Agent's longer budget, or extend deadlines to declare success.

## Delivery and sequencing

1. Freeze baseline references and extract diagnostic replay jobs; add first-divergence tracing.
2. Bind exact P3 bytes and verify the adapter/oracle/structured contracts; freeze the V3-on-P3 image control with unchanged model and tracker.
3. Fix action emission/reconciliation using the existing model; establish synthetic and recorded recovery regressions.
4. Audit artwork exposure and generate group-balanced development data plus a predeclared final corpus.
5. Run the bounded perception comparison; select one deployment model/policy.
6. Package, run paired ablations and exact-source qualification, then archive the measured result and limitations.

Prefer inline execution for the later implementation because these tasks share backend/tracking/perception interfaces. That is a recommendation, not an execution instruction for this planning turn. Use a byte-identical V4-owned backend module for development imports, while packaging reads and checks the authoritative frozen P3 source directly. Do not bind production to a mutable `agents/p3.py` or leave V4 tracking coupled to `agents/p2_next.py` helpers.

The initial builder review resolved the visible composite label rule, protected inset classes, and production checkpoint selection as proposed implementation choices. This revision additionally pins the supplied P3 source, distinguishes its two evaluation budgets, and isolates backend effects in the ablation matrix. The follow-up builder check found no important unresolved planning decisions in this P3 revision.
