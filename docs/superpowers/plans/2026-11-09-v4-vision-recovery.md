# V4 Vision and Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for the recommended inline implementation, or superpowers:subagent-driven-development if the user selects delegation. Steps use checkbox (`- [ ]`) syntax for tracking. This document is a proposed plan; implementation has not been authorized by the planning request.

**Goal:** Fix V3's observed image and temporal-recovery failures, preserve the working structured planner, and produce an honestly qualified standalone V4 candidate.

**Architecture:** A versioned V4 runtime uses the exact user-supplied frozen P3 planner behind its existing adapter boundary. A transactional tracker records every emitted action; the existing small CNN is retrained on group-balanced development data with bounded view-policy ablations. Exact-source qualification separates known regressions, fresh layouts, and genuinely untouched artwork.

**Tech Stack:** Repository CPU lock, Python with 3.10-compatible submission grammar, CPU Torch 2.7.0, NumPy/Pillow, public Grid Adventure constructors/rendering/step APIs, unittest, existing fresh-worker evaluation patterns. GPU training uses a separate verified environment later.

**Spec:** `docs/superpowers/specs/2026-11-09-v4-vision-recovery-design.md` — read it with this plan.

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

## Review focus

- A fallback WAIT or recoverable exception advances the real turn: record the actual action and never re-use stale-turn history (Task 3).
- A confident wrong coin/ghost prediction can hide a required gem: extra-view triggering and ambiguity must work even above confidence thresholds (Tasks 5–6).
- A covered exit/item is invisible: preserve corroborated layers without making `parse` mutate execution or inventing unknown layers (Task 3).
- Former holdout art can be silently relabeled as new: enforce historical group exposure, including floor sprites and inherited checkpoints (Tasks 4 and 9).
- Diagnostics, repair bursts, and additional views can consume the same five-second episode: separate trace runs from fresh exact-source timing and prohibit nested clock resets (Tasks 1, 7–9).

## Files and interfaces

All listed V4 files are new; no existing source files need editing. Fork only modules whose V4 behavior/packaging needs a versioned copy. Reuse public constructors and unchanged offline helpers explicitly, rather than refactoring old versions.

| Files | Responsibility |
|---|---|
| `agents/v4_types.py`, `agents/v4_bridge.py` | Existing runtime records/metadata/identity semantics; V4 emission and view-evidence records |
| `agents/v4.py`, `agents/v4_tracking.py` | Dispatch, transactional emission, reconciliation, read-only parsing |
| `agents/v4_perception.py`, `agents/v4_model.py` | Batched visual evidence, selected thresholds/policy, lazy embedded model |
| `agents/v4_backend.py`, `agents/v4_planner.py` | Byte-identical frozen P3 copy and thin image-budget adapter |
| `tools/diagnose_v4.py`, `tools/benchmark_v4.py` | Baseline extraction, diagnostic traces, exact-source fresh workers |
| `tools/v4_data.py`, `tools/train_v4.py` | Exposure ledger, folds/corpora, balanced training/calibration |
| `tools/package_v4.py`, `tools/qualify_v4.py` | Deterministic standalone assembly and evidence-specific acceptance |
| `tests/v4_fixtures.py`, `tests/test_v4_*.py` | Independent behavioral expectations, dataset/package/harness tests |
| `artifacts/v4/` | Baseline references, diagnostics, data, model selection, candidates and qualification |

Keep V3's `EpisodeBudget`, `BoardHypothesis`, `CellScores`, and `ObservationCheck` field contracts in V4. Add:

- `Emission(turn: int | None, action: Action, reason: str, prediction_valid: bool)` in `v4_types.py`.
- `PerceptionEvidence(scores: CellScores, disagreements: tuple, policy: str)`: disagreements contain `(x, y, full_winner, inset_winner)`; unselected cells remain NaN for partial-frame inference.
- `PerceptionPolicy(temperature: float, min_confidence: float, min_margin: float, view: str)`, where view is `full` or `inset_005`; saved/exported with the model.
- `PlannerAdapter(budget: EpisodeBudget, mode: str = "p2a")`, `step(state: GridState) -> Action`, `info() -> dict`: bind frozen P3, passing `reference_initial_deadline=min(budget.initial_deadline, budget.started_at + 2.0)`. Export `StructuredAgent` as its default `Agent`, and `reference_state_key` from the same backend for V4 tracking. Never dispatch `Task3Agent` for Task 1/2.

In V4, `ObservationCheck.observed_scores` carries `PerceptionEvidence | None`; consumers access probabilities through `.scores` and retain `.disagreements`. This is an explicit internal payload change, not a change to V3's record. All V4 initial/partial perception call sites must use it consistently.

Local test commands below run from repository root with `MPLCONFIGDIR=/tmp/v4-mpl` and `UV_CACHE_DIR=/tmp/v4-uv-cache` set. Use `uv run --frozen --no-sync`; do not sync/update the lock. At each task checkpoint, archive only changed V4 files and their SHA256 list under `artifacts/v4/checkpoints/`. If a properly populated branch is later established, scoped commits can replace those checkpoints. Never silently omit required untracked parent inputs.

## Task 1: Establish replayable failures and first-divergence diagnostics

**Files:** Create `tools/diagnose_v4.py`, `tools/benchmark_v4.py`, `tests/test_v4_diagnostics.py`, `tests/test_v4_harness.py`; output `artifacts/v4/baseline/` and `diagnostics/`.

**Interfaces:** `extract_regressions(report: dict) -> dict` returns `lost_jobs`, `incorrect_initial_jobs`, deduplicated `diagnostic_jobs`, and baseline hashes. `replay_job(job: dict, source: Path, output: Path, *, trace: bool) -> dict` preserves the job's layout/art/pixel bindings. The diagnostic result has `diagnostic_only: true` when tracing.

- [ ] Write failing tests `test_extracts_five_losses_and_twelve_wrong_boards`, `test_preserves_job_pixel_binding`, and `test_trace_cannot_qualify_as_timing`. Assert counts 5 and 12 (union is deduplicated); a tampered expected pixel hash fails before agent evaluation.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_diagnostics tests.test_v4_harness -v`; expect failures because the V4 tools do not yet exist.
- [ ] Implement extraction from `artifacts/v3/qualification-reviewed/final-300.json` and verify all eight current runtime hashes plus the frozen executable against the acceptance manifest. Also verify P3 frozen checksums and its 96,112-byte source; record the separate Agent (266 rows) and Task3Agent (60 rows) evidence, plus the newer user-reported Debugging pass. Store references and a pre-work protected-file hash inventory; copy no large dataset archives unnecessarily.
- [ ] Implement an isolated diagnostic runner derived from `benchmark_v3` with trace events for turn, returned action, expected turn, metadata/visual discrepancies, hypothesis count, first exception type/message/stack, planner result, and elapsed perception/reconciliation/search/replay time. Preserve the first failure even if later WAITs repeat. Save relevant image crops/metadata in evaluator output only; no file I/O or truth access enters production Agent.
- [ ] Replay the deduplicated old failing jobs with frozen V3. On the exact-board loss, call/observe `_step_image` in diagnostic mode so its first exception is visible; explicitly reproduce the production handler's WAIT behavior after capture. Maintain a separate oracle trace using evaluator-only labels. Do not attribute `cutoff` to planner insufficiency unless an exact-perception paired run supports it.
- [ ] Run the tests again; expect PASS. Save `diagnostics/attribution.json` with each loss's first causal boundary, observed evidence, and unresolved hypotheses. Do not proceed past a loss whose trace is missing; a diagnosis can remain a bounded hypothesis when nondeterminism is documented. Checkpoint the task.

## Task 2: Integrate exact frozen P3 and establish a separate control

**Files:** Create `agents/v4_backend.py`, `agents/v4_planner.py`, `agents/v4_types.py`, `tests/test_v4_backend.py`; extend `tools/diagnose_v4.py` with `package_p3_control(output: Path) -> Path`; output `artifacts/v4/baseline/p3-control/` and paired bridge reports.

**Interfaces:** `v4_backend.py` is a byte-identical copy of `artifacts/p3/frozen/task1_agent.py`, hash `d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`. `v4_planner.py` exports `BACKEND_SOURCE`, `BACKEND_SHA256`, `StructuredAgent`, `reference_state_key`, and the thin `PlannerAdapter` defined above. Initially `v4_types.py` only needs the shared `EpisodeBudget` contract; Task 3 adds the remaining records.

- [ ] Write failing tests `test_backend_matches_frozen_bytes`, `test_default_structured_dispatch_is_p3_agent`, `test_adapter_preserves_origin_and_two_second_reference_cap`, `test_reidentification_rebinds_verified_suffix`, and `test_late_fallback_never_renews_budget`. Assert t0/initial/cutoff remain t0/t0+3/t0+3.5, reference initial is <=t0+2, and failed/expired replay installs no partial cache. Reuse P3's existing deterministic slow-transition dispatch fixture to require the -387 B4 win within its existing virtual-clock bounds, without turning it into a wall-time claim.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_backend -v`; expect FAIL before integration.
- [ ] Vendor the exact frozen source and implement only the adapter bindings. Preserve P3's dispatch reserve, reference verification, resource-ID rebinding, production `p2a` mode, and late-fallback semantics. Keep the explicit Task3Agent in the byte-identical backend but never expose it as V4's default entrypoint. No planner algorithm edits or budget extension.
- [ ] Assemble the diagnostic control from exact P3 bytes, a P3-bound adapter, and unchanged V3 model/perception/tracking/orchestration logic in a new artifact. Retarget imports/constants in generated control copies only; never mutate V3 sources. Alias backend symbols before flattening and verify that every planner/helper resolves to P3. Package twice and require byte identity; save component hashes showing that only backend wiring differs from V3. This control is deliberately unrepaired.
- [ ] Run the new backend tests plus `uv run --frozen --no-sync python -m unittest tests.test_p3_adapter tests.test_p3_dispatch -v`. On the comparable CPU host using its existing reservation protocol, run 266 structured rows paired with the frozen default P3 Agent and 22 oracle rows paired with P3 PlannerAdapter on true states under the same image-adapter budget. Require all wins and paired rewards; preserve V3/P2-next winning rewards on matching structured rows. Never use the 60 Task3Agent rows as five-second evidence.
- [ ] Replay the old loss jobs with the control, recording what P3 alone changes without requiring it to solve perception failures. Freeze this old-perception/old-tracking P3 control for Task 8's matrix. Checkpoint only V4 additions; stop and diagnose if the adapter/oracle/structured gate fails before training.

## Task 3: Make recovery and every emitted action stay synchronized

**Files:** Create `agents/v4.py`, `v4_bridge.py`, `v4_tracking.py`, initial `v4_model.py` and `v4_perception.py` copies; extend Task 2's `v4_types.py` and use its existing P3 adapter; create `tests/v4_fixtures.py`, `tests/test_v4_tracking.py`, `tests/test_v4_agent.py`, `tests/test_v4_bridge.py`.

**Interfaces:** `Agent._emit_action(action: Action, *, turn: int | None, hypotheses: tuple, reason: str) -> Action` is the sole image-action return path. `Agent._last_emission: Emission | None` records every return. `reconcile(...) -> tuple[BoardHypothesis, ...]` retains the V3 arguments. `Agent.parse(observation) -> GridState` remains read-only or raises explicit perception uncertainty.

- [ ] Write independent failing behavioral tests: `test_wait_fallback_predicts_actual_wait`, `test_exception_then_valid_frame_does_not_loop_on_stale_turn`, `test_cutoff_emits_without_more_model_or_grid_calls`, `test_partial_repair_exception_does_not_commit`, `test_known_covered_exit_parses_without_mutation`, and `test_unknown_covered_layers_remain_unresolved`. Patch a single recoverable perception failure between two valid observations; apply the returned action to the true fixture with public `grid_step`.
- [ ] In those tests assert: every returned nonterminal action increments emitted-action count; a valid next prediction has `state.turn == observation_turn + 1`; broken continuity clears executable cached actions; the next supported frame can recover without a stale-history exception; the original budget tuple is unchanged. Re-run V3's key/effect identity and all-action observable-successor cases against the V4 bridge.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_agent tests.test_v4_tracking tests.test_v4_bridge -v`; expect the new recovery behaviors to fail before the fix.
- [ ] Seed V4 runtime from the hash-verified V3 modules, retaining the original model but using Task 2's frozen P3 binding; route `reference_state_key` through `v4_planner`, not `p2_next`. Implement local reconciliation/prediction followed by one atomic commit in `_emit_action`. Ordinary fallback WAIT advances a usable current conditional hypothesis; invalid/unknown turn or exhausted budget breaks continuity without replay. Do not repair a turn mismatch by changing the turn field. Keep terminal emission separate from a transition.
- [ ] Repair hypothesis validity checks to count a history-supported covered exit in the reconstructed state, rather than requiring an exposed exit label. A history layer is usable only when its provenance and turn remain valid. Share this path with nonmutating `parse`.
- [ ] Run the focused suite and replay the attributed exact-board failure with the unchanged V3 model and P3 backend, comparing against the frozen Task 2 control. Require removal of the traced synchronization failure and no stale-turn loop; if the episode still fails, preserve its new first divergence rather than claiming the whole loss fixed. Checkpoint with a protected-hash comparison.

## Task 4: Build exposure-aware, balanced data and freeze the test contract

**Files:** Create `tools/v4_data.py`, `tests/test_v4_data.py`; outputs `artifacts/v4/data/exposure.json`, `folds.json`, versioned corpora and `final-manifest.json`.

**Interfaces:** `build_exposure(manifests: list[Path], evidence_roots: list[Path]) -> dict`; `make_folds(exposure: dict, seed: int = 20261109) -> dict`; `prepare_corpora(exposure: dict, folds: dict, output: Path) -> dict`; `audit_v4_data(manifest: dict) -> dict`. Unknown provenance is treated as exposed, not untouched.

- [ ] Write failing tests for pixel/near-duplicate groups crossing folds, floor-group leakage, a V3 checkpoint initializing a held-group fold, old final-v5 incorrectly declared untouched, and missing class support. Assert three deterministic folds with no cross-fold group membership, and that changing only layout/appearance seeds cannot clear exposure.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_data -v`; expect FAIL.
- [ ] Implement exposure from all V3 source/manifest/checkpoint lineage, selected and nonselected training/validation evidence, and final outcomes. Treat final-v5 as development/regression, preserving originals. Reserve any genuinely untouched groups before creating folds; report counts per class and whether a strict artwork holdout is feasible.
- [ ] Create three class-stratified folds with disjoint floor groups and all 14 classes. Balance training draws by class then source group, not by the number of augmented copies. Fold model training uses the other two folds; the held fold supplies out-of-fold logits. Production fitting uses all development folds and excludes any untouched reserve. Disclose historical exposure for all fold results.
- [ ] Generate per-run training input: 200,000 augmented crops, 600 initial boards, and at most 600 reachable frames from at least 100 training layouts. Keep bounded mild V3-compatible augmentation. Save full semantic layers separately from the visible training target, with priority human, box, wall, locked, opened, gem, coin, key, boots, ghost, shield, lava, exit, floor. Add public-rendered fixture assertions for each included overlap family; actor-over-gem is human, box-over-item is box. Report/exclude an unestablished visible target. Record constituent hashes, sampling counts, and actual crop counts; audit pixels and legal initial no-overlap constraints.
- [ ] Freeze 100 final layouts disjoint from all prior V3/V4 training/selection/regression layouts, three appearances each, ten families × 30. Extend only the V4 generator with legal case variants until gem/coin/ghost/box/wall each have >=100 instances, before any model outcomes. Replay solution witnesses through public APIs. Save decoded-pixel hashes, all job fields, exposure label, and expected coverage. A truly untouched-art corpus, if feasible, is separate and bound by the same checks.
- [ ] Run tests plus dataset/board audits; expect zero violations and declared support counts. Freeze selection/qualification rules with corpus hashes. Checkpoint before training.

## Task 5: Train a bounded perception candidate and calibrate on grouped evidence

**Files:** Create `tools/train_v4.py`, `tests/test_v4_model.py`, `tests/test_v4_training.py`; outputs `artifacts/v4/runs/`, `models/selection.json`, out-of-fold logits and validation reports.

**Interfaces:** Keep `CellCNN` architecture/state-dict compatibility with `tools/train_v3.py`. `train_fold(config: dict) -> dict`, `collect_oof(run_reports: list[dict]) -> dict`, `select_policy(oof: dict, development_episodes: dict) -> dict`, `fit_production(config: dict, selection: dict) -> dict`. Selection includes a serialized `PerceptionPolicy` and all input hashes.

- [ ] Add tests asserting class→group balanced sampling, from-scratch fold initialization, rejection of final corpus paths, fixed class order, and reproducible seeds. For a synthetic logit fixture assert scalar temperature preserves argmax, and high-confidence wrong gem labels remain errors in the selection report.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_model tests.test_v4_training -v`; expect FAIL.
- [ ] Implement the unchanged 104,526-parameter RGB 48×48 architecture and new sampler/data contracts. Reuse the verified V3 optimizer/augmentation recipe; record all departures. Default experiment logging to local/offline. Verify the later selected GPU host and isolated environment before training; do not mutate the CPU lock or start jobs as part of this planning turn.
- [ ] Train exactly three fold runs, at most 20 epochs each. Record per-epoch held-fold results and checkpoints. Choose one shared epoch `e*` by aggregate held-fold exact boards, then worst-class recall, lower NLL, and earlier epoch; collect all three folds' full/inset logits at `e*` with group IDs. Report per-class/group recall, composite localization, confidence errors, and runtime estimates separately. Abort extra experimentation beyond the four-run total without a new bounded plan.
- [ ] Evaluate the spec's full-only and 5% inset policies from these same runs, with V3's old selective policy retained as a baseline. Fit temperature using the 0.25–8 grid including 1.0; compare confidence {0.85, 0.90, 0.95} and margin {0.10, 0.20, 0.30}. Rank development candidates by exact boards, worst-class recall, high-confidence errors, downstream wins, then CPU cost. Do not open final outcomes.
- [ ] Fit one production model from scratch for exactly `e*` epochs on all development folds, excluding any untouched reserve, and select its final checkpoint. Validate the fixed calibration/policy on separate development layouts. If it fails transfer, diagnose and stop this training batch; do not silently change the checkpoint, reuse final data, or add runs. Run unit tests, save the selected model or explicit rejection with all four-run receipts, and checkpoint.

## Task 6: Integrate bounded visual evidence with tracking

**Files:** Update only new `agents/v4_perception.py`, `v4_tracking.py`, `v4_model.py`, `v4.py`; create `tests/test_v4_perception.py`.

**Interfaces:** `classify_evidence(observation: dict, cells: tuple | None = None) -> PerceptionEvidence`; `initial_hypotheses(evidence: PerceptionEvidence, info: dict, *, previous=None) -> tuple`. `check_observation` uses the same selected policy for changed cells and retains independently measured actor/foreground evidence.

- [ ] Write tests `test_confident_coin_still_triggers_rival_view`, `test_view_winners_survive_fused_ranking`, `test_full_policy_does_not_call_inset`, `test_structural_cells_keep_full_scale`, `test_partial_batch_nan_cells_are_not_reclassified`, and `test_three_disagreements_remain_unresolved`. Use deterministic logits/crops so these tests prove logic independently of model quality.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_perception tests.test_v4_tracking -v`; expect new evidence-contract tests to fail.
- [ ] Implement batched policy selection exactly as recorded in Task 5. For `inset_005`, trigger on full-cell top-two intersection with gem/coin/ghost/shield even at high confidence, but skip full winners floor/wall/box/exit/locked/opened/lava. Fuse selected probabilities equally and retain disagreement. Independently measured temporal actor localization may still challenge a protected class on a later frame. Retain NaNs for unobserved partial-frame cells; test a protected winner with a rival as its runner-up.
- [ ] Implement hypothesis enumeration with at most two ambiguous cells/four boards. On disagreement use both view winners as alternatives. Reject impossible configurations without using a planner's success/reward as perception evidence. Retain the all-hypothesis common-action verification and contradiction pruning from V3.
- [ ] Test the 23 known cell errors as development regressions and temporal crops from all five losses, then run all V4 recovery tests. Require no hidden-layer invention and no deadline resets. These crop tests do not replace whole-board/episode gates. Checkpoint.

## Task 7: Export and package an exact standalone candidate

**Files:** Create `tools/package_v4.py`, `tests/test_v4_export.py`, `tests/test_v4_package.py`; output `artifacts/v4/models/selected-export/` and `candidates/<revision>/`.

**Interfaces:** Reuse unchanged `tools.export_v3.export_model(checkpoint: Path, output: Path) -> dict` for the compatible CNN, with V4 paths; add policy metadata to the V4 package manifest without changing V3 outputs. `package(source_root: Path, export: Path, policy: Path, output_dir: Path) -> Path` checks backend, runtime, model and policy hashes.

- [ ] Write failing tests for mismatched policy/model hash, accidental V3 imports in the standalone file, unregistered loader namespace, eager/export parity on batches 1/7/225, lazy structured dispatch, and runtime asset/file/network denial. Assert source <2,000,000 decimal bytes and Python 3.10 grammar.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_export tests.test_v4_package -v`; expect FAIL before implementation.
- [ ] Export through the validated in-memory TorchScript path; fail on pickle/training-module fallback. Keep single-thread FP32 inference and lazy load. Read the authoritative frozen P3 bytes, verify they match `agents/v4_backend.py`, and inline them exactly once. Immediately alias its `Agent`, `PlannerAdapter`, and `reference_state_key` before the ordered V4 modules; strip only V4 import statements, never edit the protected P3 artifact. Add binding tests to catch helper-name collisions and accidental P2 imports. Preserve the generated loader bytes and ensure the final public Agent is V4 with P3 default structured dispatch.
- [ ] Package twice to new temporary destinations and assert byte equality. Verify protected P1/P2/P3/V3 hashes against Task 1. Measure fresh CPU import/load/inference and package smoke episodes; these are diagnostics until the complete target-host runs in Tasks 8–9.
- [ ] Run tests; expect PASS with identical eager/export predictions and explicit source/model/policy SHA256s. Checkpoint the immutable candidate and manifest.

## Task 8: Attribute improvements with paired development runs

**Files:** Extend new `tools/benchmark_v4.py`, `tests/test_v4_harness.py`; output `artifacts/v4/validation/`.

**Interfaces:** `run_matrix(jobs: list[dict], candidates: dict[str, Path], output: Path) -> dict` runs old/new perception × old/new tracking as four separately packaged immutable artifacts with the same P3 backend, plus the original frozen V3/P2-next as a fifth historical control. No runtime monkeypatches/overrides are permitted in timing or qualification rows.

- [ ] Add failing harness tests rejecting oracle labels passed to a real image agent, trace overhead reported as qualification timing, reused-worker model warmth, changed watchdog origin, and mixed-source report rows.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_harness -v`; expect new checks to fail, then implement the four-artifact runner and provenance checks.
- [ ] Prepare the four combinations with exact P3 bytes and identical clocks; bind each to its own source/model/policy hash. The old/old corner is Task 2's frozen V3-on-P3 control. Compare it separately with original frozen V3/P2-next to measure the backend effect; do not attribute that difference to vision or tracking. Keep diagnostic instrumentation out of all timing artifacts.
- [ ] On the designated comparable CPU host (Omarchy), inspect live load/reservation/power state first. Use its established timing lock, sequential fresh workers pinned to one P-core, one library thread, stable Performance/EPP, 5-second strict cold watchdog. Do not force power changes or interrupt another job; if the profile is unsuitable, report nonqualification evidence until available.
- [ ] Run paired development cases, all old failure jobs, public 66, retired-v5 300, structured 266 against frozen P3 default Agent, and oracle 22 against P3 PlannerAdapter on true states with the same image-adapter budget. Retain V3 reward comparisons on matching rows. Use the matrix on the development subset needed for attribution; P3's 60 Task3Agent rows remain separately labeled longer-budget evidence. Require retired-v5 300/300 exact boards/wins and no reward regression on the 295 former wins before final freeze. Re-run targeted traces only for remaining failures and preserve them separately.
- [ ] Run harness tests; expect PASS. Save selection rationale with perception/tracking effects, per-case reward changes, stage timings from separate diagnostic runs, and resource maxima. If gates fail, stay in development; do not unlock final data. Freeze the successful exact candidate hash and checkpoint.

## Task 9: Qualify once, distinguish evidence, and archive the result

**Files:** Create `tools/qualify_v4.py`, `tests/test_v4_qualification.py`; output `artifacts/v4/qualification/`, `frozen/`, `supplementary/`, `HANDOFF.md`, and reproduction scripts.

**Interfaces:** `qualify(public: dict, regression: dict, fresh_layout: dict, manifest: dict, *, untouched_art: dict | None = None) -> dict`. Return `regression_qualified`, `unseen_art_status` (`passed`, `failed`, `unproven`), `original_unseen_art_gate_passed`, named checks/failures, counts/support, and source/model/policy/data hashes. Missing untouched evidence yields `unproven` and original gate false, even if regression-qualified.

- [ ] Write failing tests for: 296/300 exact boards rejected; 284/300 wins rejected; gem recall 98/100 rejected; only 99 critical-class instances rejected; missing/duplicate jobs rejected; exposed art labeled untouched rejected; hash/pixel/job mismatch rejected; timeout counted as win rejected; budget reset, runtime override, >=2 MB, or diagnostic-only evidence rejected. Test that strong fresh-layout results never imply unseen-art success.
- [ ] Run `uv run --frozen --no-sync python -m unittest tests.test_v4_qualification -v`; expect FAIL.
- [ ] Implement the spec's full gate table, preserving V3's per-row resource and corpus-binding checks. Public reward comparisons use V3 saved rows; structured comparisons use the frozen P3 default Agent and oracle comparisons use its adapter with matched budgets; retain the historical V3/P2-next no-regression check. Reject Task3Agent/11–12-second rows in Task 1/2 gates; generated optimality remains unknown. Include class/group denominators, per-family results and all failures.
- [ ] Unlock the frozen 300-case fresh-layout corpus once for the exact Task 8 candidate. If available, independently unlock the untouched-art corpus and report its separate result. Bind outputs to actual decoded pixels and manifest lineage. Any later tuning retires these outcomes; never overwrite them with a new candidate.
- [ ] Run `uv run --frozen --no-sync python -m unittest discover -s tests -v`; require the complete repository suite, scoped Ruff/ty checks on changed V4 files, standalone package checks, and all applicable gate rows. Save commands, versions, exit codes, and resource provenance. Repeat only after changes or failures justify it.
- [ ] Create immutable executable/model/policy/source/data manifests, reproducible qualification assembly, and scoped supplementary archives containing actual generated datasets and training records. Avoid duplicating old archives when a checksum-bound parent part suffices; list every required part explicitly. Verify archive contents/checksums and protected old-file hashes.
- [ ] Write `artifacts/v4/HANDOFF.md` with the measured result: either failed gates, regression-qualified with unseen art unproven, or separately evidenced unseen-art passage. Include exact hashes, all remaining losses, runtime limits and no platform/graded claim. Record the exact integrated P3 hash, oracle/structured parity, and the distinction from its longer-budget Task3Agent evidence. Checkpoint.

## Execution order and decision gates

Tasks 1 → 2 establish observable failure causes and the exact P3 integration baseline. Task 3 repairs synchronization before training. Tasks 4 → 5 → 6 establish artwork coverage and a bounded selected policy. Tasks 7 → 8 → 9 package and evaluate the actual deployable bytes. Keep the frozen P3 algorithm unchanged throughout.

The first implementation milestone is a verified P3 binding plus a frozen V3-on-P3 control. The second is recovery-only V4 using the existing model. The third is a perception candidate that fixes the old 23 cell errors without concealing them through recovery. The final milestone requires both regression and fresh-layout gates, with the artwork exposure limitation stated explicitly.

Recommended later execution: inline, with a fresh final implementation review. The user has requested a plan only; no training, implementation, remote jobs, commit/staging of unrelated work, or submission is part of this turn.
