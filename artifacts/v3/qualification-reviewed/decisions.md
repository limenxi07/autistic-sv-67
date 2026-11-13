# V3 implementation decisions

Copied from the execution ledger. Historical decisions describe candidate revisions; the final handoff identifies the evaluated bytes.

- Ruling: use scoped shared checkout — required agents, tests, tools, docs and artifacts are untracked and absent from HEAD; isolation would omit inputs, as anticipated by Global Constraints. HEAD is detached at 437772e. Preserve unrelated changes and frozen P1/P2 bytes. Checkpoint V3 files only.

- Task 1: Ruling: retain public resource IDs only once explicitly observed in image metadata, and reverify/rebind cached suffixes after re-identification — arbitrary ID replacement changes Python-set iteration and invalidates P2 caches — costs bounded reference replay per pickup; no hidden engine IDs read.

- Task 1: Ruling: cap initial reference fallback refinement at t0+2 s, within the t0+3 s ceiling — B4 failed at 3 s because rendered execution/rebinding exceeded cutoff, then retained paired -387 reward and won in 2.745 s at 2 s — future unseen fallback cases may need a better allocation; measure rather than extend deadlines.

- Task 1: Ruling: save scoped immutable source/evidence archive instead of a git commit — detached checkout has all required parent code untracked; no plan step explicitly requires a commit — archive manifest records exact sources and checksums for resumption/review.

- Task 2: Ruling: renderer map excludes dead/sleeping art for the 14-class initial-state corpus — the public renderer validates all map directories even with custom texture callbacks — dead-state parsing is not covered by these initial crops.

- Task 3: Ruling: add online shape/color regularization after measured unseen-art failure — pilot gem_1 (yellow faceted gem) was classified as coin/human, shield_3 as boots, with high confidence; validation-only error montage and constituent counts archived. Use 20% grayscale, mild saturation/brightness/contrast and <=12% shrinking/~one-pixel translation with reflection, no hue swaps/flips/severe crops — targets observed color/texture shortcuts without changing architecture or final holdout. Added a RED→GREEN foreground/grayscale contract test.

- Task 3: Ruling: canonical 14-class perception treats duplicate inert static-wall entities in one cell as one wall — 3 public builders have this constructor artifact (36 duplicated cells across66 observations), invisible to the class classifier and behaviorally equivalent under all seven actions, including phasing/expiry — other multiplicities remain significant and must not be collapsed. tests.test_v3_bridge.test_duplicate_static_walls_have_equivalent_observable_successors characterizes the public API. Normal public renderer with validation-only art: selected export66/66 semantic initial boards exact; raw full-entity multiplicity57/66 retained in report (all36 discrepancies only duplicate static walls). Report artifacts/v3/models/default-renderer-validation-selected.json. No final art/layout outcomes evaluated.

- Task4 Ruling: replace old copied backend with a thin version-pinned binding to p2_next.PlannerAdapter; pass reference_initial_deadline=min(original initial_deadline, started_at+2.0). agents/v3_planner.py now exports StructuredAgent (p2_next.Agent), PlannerAdapter, BACKEND_SOURCE and BACKEND_SHA256. User instruction supersedes original P2a-only backend requirement; preserve frozen P1/P2 and all old V3 evidence. Archived old copied planner as artifacts/v3/models/p2a-planner-before-p2-next.py. New RED→GREEN case proves V3 now handles one-key/many-item states on fast backend. Focused bridge/harness/P2-next suite passed; new22-case oracle-vs-P2-next benchmark running on Omarchy (artifacts/v3/bridge/p2-next-public.json). benchmark_v3 now names paired_backend and includes the p2_next source hash. Standalone packager must inline the exact pinned p2_next bytes, alias its PlannerAdapter as BackendPlannerAdapter before the V3 wrapper; omit unused explicit long-budget Task3Agent from submission if needed. V3 default Task1/2 budgets remain unchanged.

- Task4 Ruling: compare consumed coin/gem inventory entries as an identity-preserving multiset while retaining key order and all effect entries — observed public set-table reordering of ten inert coins triggered8 unnecessary B2 repair searches; IDs/types must still match — costs if wrong: this assumes already-consumed coin/gem list order has no future mechanics; existing all-action replay and full episode gate remain required. RED→GREEN reordered-coin check added.

- Task4 Ruling: add per-episode observed actor template as independent fallback — validation layout38 human+boots remained a high-confidence ghost after inset/foreground CNN checks; matching the actual initial actor pixels against current cells scored0.844 at the true location and<=0.214 on all negatives — fixed0.75 score and0.20 uniqueness margin may miss changed/occluded appearances; no predicted pixels or external assets used. A failed first-step boots fixture precedes implementation. No further model training or holdout evaluation.

- Pre-freeze audit Ruling: supersede untouched final300-v2 with final300-v3 before any final model outcome — generator's family named multi-gem contained only one required gem; add an opt-in multiple_gems variant for final multi-gem/combined families, preserving all frozen train/validation generation and v2 artifact bytes. Freeze strict v3 manifest and300 actual rendered boards, retaining group/layout disjointness and replayable witnesses; cost: final manifest differs from training manifest, so record lineage and both hashes. RED→GREEN ten-family witness test checks>=2 gems in those two families. This is a corpus defect correction, not tuning on final results.

- Task6 Ruling: retire that candidate and untouched final-v3 before further tuning; create a fresh100-layout/300-appearance final-v4 corpus disjoint from v3 and training/validation — preserves the single final-outcome rule after policy revision — costs more generation/evaluation; never reuse a tuned-on final outcome.

- Task6 diagnosed view-policy Ruling: use a5% inset CNN view only for gem/shield candidates when that view confidently predicts either class — RGB classified failed gem asshield0.9186 (0.99994 atT0.25), grayscale worsened it, all five measured positive insets gavegem, and both other trained seeds also predictedgem. Applying5% inset to every class damaged13boxes/1exit on validation; reject that variant, preserve structural full-cell views. Selective pair policy preserves120/120 initial validation boards and is applied during later visual checks too. No new model/training run, no objective/solvability-based relabeling. RED→GREEN failing public-crop test plus33 model/Agent/tracker/package tests pass. New source must be re-frozen and completely qualified.

- Final-review scheduling Ruling: perform the one fresh implementation review while final hardware qualification is queued — code/runtime/model are complete and frozen, all226 repository tests pass, but hardware gates and final handoff metrics are explicitly pending — reviewer checks whole V3 code, test/packaging/qualification gates; author will verify subsequent report assembly against actual outcomes. This avoids idle CPU-reservation wait; cost is metrics will be checked by the executor rather than a second reviewer. No second review dispatch.

- Final: Ruling: reviewer declined P2-next selection — explicit user instruction stands; costs unsupported backend schemas remain explicit rather than silently switching parent.

- Final: Ruling: reviewer declined duplicate-wall normalization and consumed inert-resource order — existing public-action equivalence fixtures stand; costs rely on characterized standard mechanics only.

- Final: Ruling: reviewer declined0.50s cold-load target miss — disclosed0.527s and mandatory complete-episode/strict-cold gates decide release; costs less startup margin.

- Final: Ruling: reviewer declined final corpus replacements — all retired before any final model outcome, freshv5 is frozen for revised runtime; costs extra generation/evaluation, no reused tuned-on final outcomes.

- Final: Ruling: reviewer declined pending final metrics/handoff — remain explicitly pending until fresh Performance qualification; no completion claim.

- Final: Ruling: reviewer declined external parent/P3 internals, engine implementation and private grader — preserve authorized scope and no engine-source inspection; costs private-distribution performance remains unproven.

- Task6 Ruling: preserve generated loader bytes and suppress only its I001 import-order lint — source must remain byte-identical to the qualifying candidate and vetted export; costs generated-loader style remains unformatted, no behavioral change.

Deferred minor: Agent.parse may reject a known exit covered by the actor; this affects visualization and does not establish recovery coverage.
