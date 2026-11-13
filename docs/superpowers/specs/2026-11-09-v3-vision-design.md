# V3: image perception over the frozen P2 planner

Status: proposed design, 9 November 2026. Planning only; no implementation, training, remote jobs, or submissions started. Choices and numerical targets below are proposals, not measured V3 results.

## Objective and evidence

Add Task 2 image support using a small supervised cell classifier, persistent symbolic state, and the qualified P2 planner. The user's latest clarification is **Task 1 is now also 22/22**; the earlier 20/22 statement is superseded. The exact assessment underlying that clarification is unspecified. Preserve it separately from the already recorded user-reported 22/22 Debugging passes for both variants; do not infer graded marks.

Verified source hashes during this planning pass:

- P2a: `artifacts/p2/frozen/p2a/task1_agent.py`, `e478c1ccb78c21ea4e97fa3c54ab08a28a401c9f4e4ca2846dfcb0590bb7c2cb`.
- P2b: `artifacts/p2/frozen/p2b/task1_agent.py`, `551ee00377ffe42a343d18c887b1c7f36ef4baf4fa0fff6a22b0053c9845edd0`.

Existing evidence, not rerun here: each variant won 66/66 public episodes at catalog maximum reward and 200/200 generated cases. P2a remains the documented recommendation. P2b reduced mean public runtime from 0.740 to 0.602 seconds but did not improve rewards or first-win time. Those timings exclude a vision model and are not available budget guarantees. See `artifacts/p2/HANDOFF.md` and `selection.json`.

Current P2 returns WAIT on images. Public assets contain **333 PNG files across 14 classes**, counted live: boots, box, coin, exit, floor, gem, ghost, human, key, lava, locked, opened, shield, wall. The assignment explicitly uses unseen visual variants for graded images. The local starter uses just one template per class and reconstructs only agent health; this is insufficient for temporal state tracking.

## Constraints and scope

- Tasks 1/2: 150 turns, 5 seconds for the whole episode, 1.5 GB memory, one CPU core, no GPU at submission time.
- Submission source must be below 2,000,000 bytes, including embedded model and loader. Target at most 1,600,000 bytes to leave room for fixes.
- Retain Python 3.10 grammar and prelude/no-module-registration loader checks. Development/deployment compatibility uses the repository's pinned CPU Torch 2.7.0 and NumPy/Pillow environment.
- Use public assets, assignment/tutorial, public constructors and black-box API observations. Do not inspect engine implementation source. No network, external asset reads, or model downloads in the submitted agent.
- Preserve P1/P2 source and evidence bytes. Build V3 in new files; maintain Task 1 as a separate usable submission.
- Archive dataset generation scripts, training code, configurations, checkpoints, manifests and generated datasets as required by the assignment's supplementary-files rule.
- Task 3, learned search, policy learning, world models, and expanding P2's supported simulator schema are outside the initial release. Report boss diagnostics separately if time remains.

## Recommended approach and alternatives

Use **a roughly 105k-parameter CNN trained offline, then P2a planning with visual feedback**. Start with FP32 inference; the proposed weights are about 0.42 MB before serialization/base64. Actual exported bytes and CPU measurements decide acceptance.

Template matching across all supplied sprites is a useful offline baseline, but cannot establish unseen-style generalization. Handcrafted colour/edge features plus a compact linear classifier are a deployment fallback if Torch startup dominates. A large pretrained vision model or end-to-end learned agent adds substantial packaging/training risk with no demonstrated need here. P2b is an ablation candidate for the combined vision budget, not an automatic promotion.

### 1. Prove the observation-to-planner bridge before training

First build an **oracle-perception diagnostic**: the harness supplies semantic cell labels from its separately held ground-truth state; the agent receives only the normal image/metadata in all genuine end-to-end runs. Verify that perfect visual labels plus permitted metadata reproduce P2 actions and rewards. This separates perception errors from adapter errors. Production interfaces must have no access to `env.state`, builder names, catalog solutions, hidden IDs or layout seeds used as an answer lookup.

Black-box reset checks on A0 and B4 found `info.agent` health/effects/inventory, `info.status` score/phase/turn, and `info.config` movement/objective/seed/width/height/turn_limit. A0's actual image shape was 457x640x4 for a 7x5 board. Derive crop edges from image dimensions and board dimensions, with the starter's integer partition formula; never assume square images or integer cell widths. The observed objective string was `collect_exit`, while P2 checks the public `collect_gems_and_exit` objective object: characterize and explicitly test this mapping.

Probe complete image metadata after key/coin/gem pickups, stacked powers, shield damage, box/item overlap, unlocking and expiry. Preserve stable locally assigned entity slots across frames. P2's key-sensitive inventory order and effect identity are real constraints; reconstructing a new arbitrary-ID GridState each turn can change transitions. Before trusting the bridge, compare all-action **observable** successors after re-identification with the original reference state on reachable fixtures, not just reconstructed-state replay against itself. Test permutation sensitivity explicitly. If order-sensitive distinctions cannot be observed or tracked, retain bounded alternative resource states or declare that schema unsupported; do not claim reference fallback recovers information that images did not supply. This is an early go/no-go gate.

### 2. Dataset and leakage controls

Inventory assets by path, class, SHA256, and near-duplicate visual group. Split groups per class approximately 60/20/20 into train/validation/final test **before** rendering or augmentation. Floor/background variants must follow the same split. No pixel-identical or near-duplicate sprite group crosses splits. With small per-class counts, record actual counts and limitations.

Use public rendering APIs and separately supplied symbolic labels for labeled boards and reachable rollout frames. Test the documented renderer asset-root configuration through its public API; train/validation/test asset roots contain only their assigned sprite groups. Verify provenance for every constituent sprite. Composite single-cell augmentation may supplement real rendering, but cannot substitute for checking actual renderer crops. Fail generation if assets cannot be attributed to the intended split.

Keep appearance and geometry axes separate: known-layout/unseen-art, new-layout/known-art, new-layout/unseen-art. Existing P2 generated cases are regression data; create and freeze new V3 layout seeds/families before candidate evaluation. Initial generated layouts must obey the no-overlap rule; later reachable frames deliberately include overlaps.

Start with 200,000 class-balanced 48x48 crops and real full boards. Vary legal floor combinations, scale, subpixel resampling, modest brightness/contrast/saturation, and small translations. Avoid severe crops, arbitrary recolouring, or transformations that erase semantic distinctions. Expand toward 1 million samples only if validation shows a specific coverage gap. More augmented copies do not create new artwork styles. Save per-class confusion and hard examples; validation failures may drive training, final-test failures may not silently become tuning data.

### 3. Classifier, uncertainty and deployment

Proposed network: RGB 48x48 input; four 3x3 convolution/ReLU blocks of widths 16/32/64/96 and strides 1/2/2/2; adaptive average pooling to 2x2; linear 384->64->14. Predict all 14 classes explicitly, including floor/wall; the most frequent tile is not necessarily floor. Train cross-entropy with class-balanced batches, select using validation board accuracy and downstream success. Start with 20 epochs, one seed; repeat the selected recipe with two further seeds. Cap the initial comparison at four training runs before reassessing data versus architecture.

Batch all initial cells in one inference call. Calibrate confidence on validation data. Constraints such as exactly one initial agent/exit, initial key/locked-door equality and declared maxima can reject hypotheses, but cannot prove the remaining assignment correct. Do not force a confident label to make a map solvable. Keep at most four candidate boards, varying only up to two ambiguous cells with top-two labels; larger uncertainty is explicitly unresolved. Hypothesis enumeration and planning share the same deadline. Multiple hypotheses may authorize a common nonterminal action prefix only if every retained hypothesis verifies that prefix and predicted resource safety; a win in one hypothesis does not certify another. No deliberate destructive probing in the first release.

Use CPU FP32 first; no assumption that FP16 or quantization improves CPU latency. Export using the repository's `generate_torch_loader_snippet`, requiring the self-contained TorchScript path and failing packaging if it falls back to a training-module import/pickle. Trace only tensor-only classifier forward computation in eval mode; validate eager/exported parity for batches 1, 7 and 225. PyTorch's [2.7 trace documentation](https://docs.pytorch.org/docs/2.7/generated/torch.jit.trace.html) warns that tensor-dependent Python control flow is not captured; tracking and hypothesis logic stay outside the model.

Set single-thread inference before model work, cache the loaded model, and measure a fresh cold load per episode. The new V3 packager must allow the vetted in-memory model loader and NumPy/Pillow/Torch imports; the existing P1 packager deliberately rejects them and should remain unchanged.

### 4. Persistent state with visual checks

Parse frame zero fully. Maintain a symbolic predicted state through public `grid_step` for execution, preserving entity identity, resource order, timers and the board under the agent. Search still uses P2's fast simulator and reference-verified plans. A candidate verified against a reconstructed board is valid **conditional on that board**, not proof of correct perception.

On every subsequent frame, compare observed score, health, inventory, effects, turn and phase, plus visual agent position and action-affected cells. Track the previous/destination cells, pushed boxes, doors and pickups; detect unexplained visual changes across the board with a cheap image-difference pass before deciding which crops need classification. Do not reclassify covered objects as absent. Preserve occluded layers from a corroborated history, explicitly label unresolved layers, and never use model predictions as independent observation evidence.

Later frames may show composite/corner icons. Test actor localization and changed-cell parsing on reachable composites, using the same classifier on appropriate subcrops only after this is validated. A frame-zero classifier alone is not a recovery parser. On mismatch, invalidate the cached suffix, reconcile metadata and visible layers with history/retained hypotheses, then repair within the original deadline. If hidden layers cannot be recovered, retain uncertainty rather than inventing a single exact board. WAIT is merely a legal terminal fallback after exhaustion, not a safety guarantee; report the episode as unresolved/failure when it does not win.

### 5. One shared budget

Create an absolute episode deadline before any lazy model load or parsing. Proposed sub-budgets, to validate on one Omarchy P-core:

| Component | Proposed allocation |
|---|---:|
| Model load, first parse and bridge | <=0.50 s target |
| First planning/refinement | remaining time until t0+3.00 s |
| Final verification/repair cutoff | t0+3.50 s |
| All observation checks and environment execution | whole episode <4.25 s target |
| Local hard evaluation watchdog | 5.00 s |

Do not construct an inner P2 agent after parsing and grant it a fresh 3.5 seconds. Provide the original absolute deadlines to the V3 planner adapter and reference fallback. Package-import timing is separately recorded and also included in a conservative import-to-terminal measurement; the private grader timing boundary is unknown. The numerical allocations are rejection criteria for experiments, not predictions that Torch will satisfy them. If cold-load cost is too high, test the compact feature baseline before expanding scope to a custom neural runtime.

## Evaluation and promotion gates

1. Bridge gate: all 22 public layouts under oracle perception match baseline wins/rewards; all-action observable successor comparisons and temporal-resource fixtures pass. Identity-sensitive unsupported cases are explicit. An unsupported required public case blocks release.
2. Perception diagnostics: per-class recall/confusion, calibration, exact full-board accuracy, critical-object accuracy, and later-frame localization/recovery. On 225 independent cells, 99.5% per-cell accuracy would imply only about 32% exact boards; independence is illustrative, not assumed in evaluation.
3. Freeze a final corpus of 100 new layouts across ten mechanics families, each with three held-out appearance assignments, before final evaluation. Proposed gates: >=99% exact initial boards, >=95% end-to-end wins on those 300 episodes, and no unresolved simulator/metadata contradictions on accepted execution. Report raw rewards where optima are unknown. Report each geometry/appearance axis separately; high aggregate counts cannot hide missing classes.
4. Exact candidate Task 2 public suite: 22 levels x three predetermined appearance seeds = 66 episodes, all wins, no timeouts/errors, each effective performance >=0.50. This is a strict local target, not the grader's three-instance aggregation. Measure downstream reward with the true evaluation environment; reconstructed-state replay is not an evaluator.
5. Structured-input regression: the V3 dispatch path must preserve frozen P2a's wins and per-case rewards across its 66 public episodes and existing generated regression corpus, without loading the vision model. If P2b is selected for image input, qualify that choice separately and retain Task 1 P2a behavior.
6. Runtime/package gate: one CPU/core and one library thread; every qualification episode <4.25 s and <1 GB peak RSS; final source <2,000,000 bytes; no external files/network; prelude and no-module-registration execution; exact source/model/config/manifest hashes. Cold import-to-terminal measurements must also fit 5 s before claiming conservative cold-start compliance.
7. Recovery tests: wrong initial tile with identical counters, unexpected position, picked-up/covered items, box on an item, multiple effects, timer expiry, key order, and mismatch near cutoff. Require detection before blindly issuing the stale next action; report which synthetic corruption cases can actually recover within the remaining budget.

Run architecture/augmentation/consistency ablations on validation splits only. Unlock the final holdout once a source/model/config is selected. If it fails, preserve the report; any tuned successor requires a fresh holdout. Local synthetic/artwork holdouts do not establish performance on the private asset distribution.

## Compute and sequence

Use the offered **M5080 for offline training**, after checking its actual device identity, VRAM, supported Torch/CUDA build, free capacity and current users. Do not infer hardware from the hostname or install over a shared environment. Keep GPU training in a separate environment; export/reload and CPU qualification use the pinned deployment environment. Begin with a ten-minute throughput/validation pilot and record samples/s, memory and startup time before estimating the full run. CPU rendering may be the bottleneck; cache or prefetch crops and parallelize offline generation only within available capacity. No rental or multi-GPU work is needed for the initial experiment.

Proposed calendar, subject to the bridge gate:

- 9 Nov: image/metadata contract, oracle bridge, grouped asset splits, baseline parser and export timing feasibility.
- 10 Nov: dataset generation, first M5080 training run, validation and CPU export.
- 11 Nov: tracker/recovery integration and limited augmentation/consistency ablations; compare P2a/P2b under the same total budget.
- 12 Nov: freeze candidate, final holdout and exact-source qualification, supplementary archive, user-run Debugging handoff.
- 13 Nov: retain submission buffer before the repository's stated 23:59 deadline; avoid new architectural work. No graded attempt is implied by this plan.

If the bridge remains ambiguous after the first focused day, resolve observability before training more. If unseen-sprite validation fails, inspect class confusions and split integrity before buying compute. If whole-episode runtime fails, remove optional inference and reward refinement before reducing validation coverage. Keep the frozen Task 1 checkpoints available throughout.

## Planning provenance

Read: `capstone-project.py`, `utils.py`, `tutorial.py`, `docs/approaches.md`, P2 source/qualification/freeze records and tests. Only reset-time public-API image probes ran during planning. The hosted game documentation could not be retrieved; local assignment/tutorial and observed public APIs are the current basis. No engine source was inspected. The written plan is a proposal for review, not implementation authorization.

Independent builder check completed: no important unanswered questions prevent planning. The reviewer agreed that identity observability, renderer split feasibility and cold CPU cost belong in early experimental gates rather than additional user questions. Implementation plan: `../plans/2026-11-09-v3-vision.md`.
