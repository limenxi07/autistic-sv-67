# Grid Adventure V1 Capstone: Approach Options and Reading List

Originally compiled 14 September 2026. Reviewed and updated 8 November 2026 against `capstone-project.py`, `utils.py`, `public_levels/catalog.py`, and research retrieved through the alphaXiv connector. The original engine timings and renderer-source observations below are historical notes, not measurements repeated in this review.

**Post-v4 decision, 9 November 2026:** use staged parallelism after v4 is qualified and frozen. Establish a shared benchmark, then compare exact simulation with learned guidance against one compact learned-world-model challenger. Keep code tuning as a bounded control and adversarial level generation as shared infrastructure. Sections 3.4 and 4 record the supporting alphaXiv paper review, comparison, sequencing, and stop rules. These are proposed experiments; this update reports no new training or benchmark results.

**Recommendation:** build a validated symbolic simulator, a deadline-aware reward planner, and a supervised cell parser first. Add learned guidance only after it improves end-to-end results under the actual limits. A simulator can verify a plan's execution and reward from a given state; it does not certify the parsed state, optimality, or a grading threshold.

**Main review corrections:** learned A\* guidance is not automatically admissible; fixed-point residual is not a correctness certificate; Searchformer/Dualformer headline results use up to 64 sampled plans; the NCA paper distinguishes normalized accuracy from complete-path success. Runtime and grade predictions have been replaced with explicit validation requirements. Paper descriptions report published experiments; “How it maps here” and fit assessments are project-specific proposals, not results on Grid Adventure.

Scope of this document:

1. What the assignment actually rewards and constrains.
2. Conventional options for planning and perception, ranked by ambition.
3. Experimental directions: world models, energy-based models, diffusion planners, and learned search, each with an honest fit assessment.
4. A recommended stack, the post-v4 experiment decision and parallel tracks, and acceptance tests.
5. The unhinged tier: rule induction, amortised search, program evolution, adversarial level design, cellular automata, and one coherent maximally ambitious stack.
6. A grouped reading list with links.
7. Compute estimates, current on-demand/spot rental costs, alternative accelerators, multi-GPU scheduling, data-generation budgets, and FP8/NVFP4 training feasibility.

---

## 1. The assignment in numbers

### Grading

The authority for this section is the **Grading Scheme** in [`capstone-project.py`](../capstone-project.py). The local implementation is `get_performance` / `evaluate_level` in [`utils.py`](../utils.py), with failure handling in the notebook's `evaluate_public_levels`. The private Coursemology grader is not in this repository, so this audit verifies the documented rules and local behavior, not the hidden implementation.

| Item | Value |
|---|---|
| Task 1, structured `GridState` input | 10 test sets, 1 mark each, normalized-performance threshold 0.85 |
| Task 2, RGBA image input | 10 test sets, 1 mark each, normalized-performance threshold 0.50 |
| Task 3, boss level, either input | 2 test sets, 2 marks each, threshold 0.85, bonus |
| Raw total and cap | 24 raw, capped at 20; divide the capped total by 2 for the contribution out of 10 course marks |
| Per successful, on-time test case | `p = max(0, R - M) / (O - M)`, with reward `R`, optimal reward `O`, minimum reward `M`, and `O > M` |
| Per test set | `w * clip((p1 + p2 + p3) / (3*t), 0, 1)`, with threshold `t` and weight `w = 1` for Tasks 1/2, `w = 2` for Task 3 |
| Failure to win, timeout, or crash | performance 0 for that case |

Compute each case's performance using **its own** optimum and minimum, then average the three performances, divide by the task threshold, clip, and apply the set weight. Do not average raw rewards or clip each case to its task threshold before averaging. Sum the 22 set marks, cap at 20, then divide by 2. Tasks 1 and 2 alone can reach the cap; boss marks can compensate for lost marks elsewhere, up to that cap.

#### What 0.85 and 0.50 actually permit

The notebook describes `M` as 50% worse than optimal. The public harness implements `M = 1.5 * O`, and all 24 reference rewards in [`PUBLIC_LEVEL_MAX_REWARDS`](../public_levels/catalog.py) are negative. In this setting, let optimal net cost be `C* = -O > 0` and the extra net cost of a winning plan be `delta = O - R >= 0`. Then:

```text
p = max(0, 1 - 2*delta/C*)
For a positive case-level target t: delta <= (1-t)*C*/2
```

| Case-level target | Maximum extra net cost relative to the optimum | If `O = -100`, `M = -150` |
|---|---|---|
| 0.85, Tasks 1/3 | 7.5% of `C*` | `R >= -107.5`; with integer rewards, at least -107 |
| 0.50, Task 2 | 25% of `C*` | `R >= -125` |
| Zero performance | Extra net cost at least 50% of `C*`, or any failure | `R <= -150`, even if the plan wins |

Thus “0.85 of optimal” does **not** mean 15% extra path cost is allowed, and “0.50” does **not** mean 50% extra cost is allowed. These are case-level equivalents for interpreting the thresholds; full test-set marks depend on the three-case mean. They concern **net reward cost**, including coins, not path length alone.

For a general supplied minimum, use `R >= M + t*(O-M)` for a positive target `t`. Do not extrapolate `1.5 * O` to arbitrary generated levels: `O = 0` gives a zero denominator; `O > 0` makes this supposed minimum exceed the optimum. Such levels require a separately defined valid normalization with `M < O`. The hidden grader's treatment of nonnegative optima is not specified here.

#### Three-case aggregation and failures

These examples assume the listed nonzero case performances are achieved and failures score zero:

| Three case performances | Task 1 set marks, out of 1 | Task 2 set marks, out of 1 | Task 3 set marks, out of 2 |
|---|---|---|---|
| 1, 1, 1 | 1 | 1 | 2 |
| 1, 1, 0 | 0.784314 | 1 | 1.568627 |
| 1, 0.5, 0 | 0.588235 | 1 | 1.176471 |
| 1, 0, 0 | 0.392157 | 0.666667 | 0.784314 |

Task 2 can therefore receive full set marks despite one failed case if the other two performances sum to at least 1.5. Tasks 1/3 cannot reach full set marks with one failure when the other cases are bounded by the true optimum (`p <= 1`). Every case still matters: failures remove reward quality that might otherwise compensate for weaker cases.

The literal case formula has **no upper clip**. For example, `R = -99, O = -100, M = -150` produces 1.02 locally. With a genuine optimum and identical rules, a plan cannot exceed it; a value above 1 should trigger investigation of the reference reward, level configuration or reward accounting. Do not silently alter the documented formula. Set marks remain capped by the set-level clip.

#### Reward accounting and probe budgets

For a winning episode starting at score zero, with `T` executed actions including the winning action and `K` collected coins, the documented reward is `R = 5*K - 3*(T-1)`. Count pickups, key-use actions, waits and ineffective moves as actions too. Entering a coin's tile does not itself collect it: in a direct API check, moving onto a coin costs -3, `PICK_UP` contributes net +2, and moving onto the exit on the winning turn contributes 0. Lava damage affects health and feasibility, not a separate score penalty.

For two winning plans, collecting `k` additional coins with `a` additional total actions changes reward by `5*k - 3*a`; include the pickup actions in `a`. An otherwise unnecessary single `PICK_UP` on a coin improves reward by 2. One extra coin requiring two additional actions lowers reward by 1. This explains why neither “collect everything” nor “minimize moves” alone optimizes score.

With unchanged coin collection and no other effects, `k` extra actions lose `3*k` reward. Starting from an optimal plan, the per-case allowance is `floor((1-t)*abs(O)/6)` such actions under the public negative-optimum normalization:

| Public reference | `O` | Extra cost-only actions at target 0.85 | At target 0.50 |
|---|---|---|---|
| A0 Basic Movement | -9 | 0 | 0 |
| A3 Multiple Required Gems | -63 | 1 | 5 |
| B3 So Close Yet So Far | -315 | 7 | 26 |
| C0 boss | -68 | 1 | 5 |

These are score-only ceilings relative to an optimal winning route, not extra turn-limit allowance or a guarantee that probes are safe. A0 demonstrates the sensitivity: four `RIGHT` actions win with -9; adding one `WAIT` still wins but scores -12, giving performance 1/3. Even Task 2's threshold does not make a wasted action affordable on every case. For a plan already below optimal, subtract its existing reward deficit from the allowance.

#### Local evaluator differences and audit evidence

- `get_performance` checks `win`, but has no timeout/error arguments. `evaluate_level` can therefore return raw `performance = 1` together with `timeout = True`. Use the notebook's **`effective_performance`**, which zeros errors and timeouts.
- The local timeout check uses unrounded `runtime >= time_limit`; displayed runtime is rounded to two decimals. A 4.999-second run can display 5.00 and pass, while exactly 5 seconds is flagged. Read `status`/`timeout`, not the printed time, and retain headroom for the real grader.
- Local timing begins after `create_env` and includes `evaluate`, which resets the environment, constructs the agent, steps it and records history. It does not measure environment construction or module-import startup. The remote grader's exact timing boundary is not available here.
- `get_minimum_total_reward` simulates waiting until termination. It is **not** the grading floor used by the public suite; that suite explicitly passes `1.5 * optimal`. Do not substitute the wait-to-lose reward into the grading formula.
- `summarize_public_results` averages individual effective performances. It applies neither three-case grouping, task thresholds, boss weights nor the final cap. The 48-row “Overall” mean is a diagnostic, not marks out of 20 or 10.

**Scoring audit, 8 November 2026:** eight reward/win examples and four controlled timeout/error scenarios passed against the existing scoring/evaluation functions; eight exact-arithmetic assertions checked aggregation and normalization. Four real simulator traces checked A0 with/without a wait and a one-coin grid with/without pickup. These checks do not certify all public reference optima or exercise the private grader. The historical timing table below was not rerun.

### Limits

| Constraint | Value |
|---|---|
| Time | 5 s per whole episode for Tasks 1 and 2, 15 s for Task 3 |
| Turns | 150 per episode |
| Memory | 1.5 GB |
| Upload | 2 MB, models must be embedded in source as base64 |
| Compute | single CPU core, no GPU |
| Grid | up to 15x15 |
| Entities | health at most 5, at most 5 gems, 5 key/door pairs, 5 boxes, 3 of each powerup, unlimited coins and lava |
| Initial state | no overlapping entities, all doors locked, keys equal doors |
| Rewards | step cost 3, coin +5, lava damage 2, final winning turn free |
| Powerups | boots 5 steps at double speed, ghost 5 steps of phasing, shield 5 uses |
| Actions | UP, DOWN, LEFT, RIGHT, USE_KEY, PICK_UP, WAIT |
| Graded images | held-out visual variants of the same 14 entity classes |

The time budget covers the whole episode. Spend most planning effort up front, cache the plan, and reserve time for observation checks and repair. Include model decoding/loading and preprocessing in local end-to-end measurements. Blind replay is fragile when the initial image parse is wrong.

### What the engine costs

Historical measurements reported by the original draft on B3; no benchmark artifact, hardware description, or repeat-run distribution is attached here:

| Operation | Time |
|---|---|
| One step on the raw engine `State` | 0.043 ms |
| One step through the `GridState` wrapper | 1.119 ms |
| Hashing a state by `str(grid)` | 0.22 ms |

The starter agent does breadth-first search over full `GridState` objects with string hashing:

| Level | Baseline result |
|---|---|
| A0 Basic Movement | win, 0.03 s |
| A3 Multiple Required Gems | win, 6.8 s, timeout |
| A10 Combined Mechanics | win, 3.9 s |
| B1 Hypnosis | win, 20.6 s, timeout |
| B9 Manhattan | win, 41.9 s, timeout |
| C0 boss | crashed inside 5 s |

These numbers suggest wrapper and hashing overhead, but do not establish an end-to-end speedup. Search-space growth also matters: tracking optional coin subsets can create exponentially many distinct states. The current starter searches again on each call and uses `str(next_state.grid)` as its visited key; that key omits non-grid resources such as health, effects, inventory and remaining turns. Optimizing it requires a sufficient state representation as well as faster hashing.

The current public suite has **48 checks**: 22 non-boss levels in each observation mode, plus 2 boss levels in each mode. `evaluate_level` records a timeout **after** the run returns; it does not interrupt a hanging search. The notebook turns errors and completed timeouts into effective performance zero. These are smoke tests, not a reproduction of hidden cases or three-case grading.

### What the renderer does

The original draft recorded these implementation observations from `grid_universe/renderer/image.py`; this review did not re-audit that file or the asset count. They are useful hypotheses to test through rendered observations, not assignment guarantees:

- The image is 640 px wide; cell size is 640 divided by grid width. Assets are 128x128 RGBA, 14 classes, 333 files.
- For each image, one variant per class is chosen from the seed. Every instance of a class in one image uses the same variant. Floor and wall variants also change with the seed.
- Entities are alpha-composited on the floor tile. Walls are opaque background tiles.
- When entities overlap, the highest-priority entity is drawn full size and collectibles become 40% corner icons. This never happens in the initial frame.
- Key/door recolouring only triggers with more than one key group. All keys share id `A`, so no recolouring occurs.
- The `info` dict carries health, active effects with remaining limits, inventory including keys, score, phase, turn, and grid size. It does not carry positions.

The notebook guarantees no initial overlaps and held-out visual variants, but not that floor is the most frequent tile or that exact pixel equality identifies a class. Derive crop boundaries from the actual image shape and `info['config']` dimensions, as the starter does, rather than hard-coding 640 px. The notebook asks students to use the tutorial and documentation and to avoid inspecting engine source; future implementation work should use those interfaces and black-box transition tests.

---

## 2. Conventional options

### 2.1 Planning options, from sane to unhinged

**P1. Patch the baseline.** Cache plans between calls, replace string hashing with a sufficient compact state key, and avoid copying whole action lists at every expansion by storing parent pointers. Profile the documented transition interface before replacing it. The historical raw/wrapper step ratio is about 26x, but is not a measured planner speedup. BFS minimizes action count; with optional coin rewards it need not maximize score.

The first Task 1 milestone is specified in the [P1 implementation plan](superpowers/plans/2026-11-08-p1-task1-baseline.md): preserve a bounded cached-search checkpoint, evaluate score improvements, and validate an exact single-file candidate before a graded attempt. It requires CPU time only; no model training or GPU rental.

**P1 Task 1 baseline measured, 8 November 2026:** the new isolated benchmark
([evidence](../artifacts/p1/README.md), [all 22 results](../artifacts/p1/starter.json))
ran the unchanged starter on Omarchy's Intel Core Ultra 9 275HX with one CPU
affinity, 150 turns, and an external five-second evaluation watchdog. It won
11/11 intro maps (mean effective performance 0.948052); all 11 challenge maps
timed out. A2 performance was 0.428571; A3 took 4.403 seconds, exceeding the
planned 4.25-second candidate headroom target. Eight harness tests pass on
Linux and macOS. This established the measurement checkpoint before state-key, bounded-search
and exact-candidate implementation.

**P1 best-first checkpoint selected, 8 November 2026:** the bounded cached
BFS checkpoint P1a solved 10/22 public maps; P1b solved 15/22, including all
11 intro maps with effective performance 1.0. Mean public performance rose
from 0.428571 to 0.659441 with no lost wins/errors/timeouts. On the fixed
50-layout tuning split, wins improved from 44 to 50; the unchanged selected
source then solved all 50 held-out layouts in one sweep. Generated optima
were not established, so only their wins/raw rewards are reported. Separate
profiling attributed approximately 86% of P1b evaluation time to the public
simulator, 9% to state keys and under 1% to heap/heuristic work. Retain this
checkpoint rather than make a custom simulator part of P1.

**P1c guided follow-up,8 November2026:** the website first reported10/22 after compatibility cleanup, then 20/22 after guided search. The latter failed only Ow Oof Owie and Warehouse. Current first-win detour ordering reduces their public-simulator work 491→119 and 720→210calls. The new full-state key also preserves carried reference-ID/counter associations, verified by a stacked-shield regression and public roundtrips.

Current exact source: `a51380b9771c6cb1c8614a09bbb79bc5a1a06a8dddcb761883b599924c13aea5` (37,287 bytes). All 68 tests, Ruff lint/format and ty with warnings as errors pass. Three exact-export public sweeps reached **22/22 wins and catalog maximum rewards each** (66 wins; mean effective performance 1.0), with no errors/timeouts, maximum episode 3.199105 s and peak RSS 322.6 MB. Fixed tuning and heldout corpora each replayed 50/50 wins; these are revalidation of the existing corpus, not new independent heldout evidence. The user reported all 22 Debugging tests passed for the latest delivered source; P1 is now frozen. See `artifacts/p1/candidate/platform-validation.json`.

The original Manhattan-only P1b remains frozen. Current guided/lazy ordering uses cached wall/resource geometry and target tours; all actions remain available, state keys retain score/turn/resources, and every accepted route is replayed through the public simulator. A stronger all-phase box penalty regressed an optional-coin reward and was rejected; the final score-refinement phase restores the earlier box weight. The user-reported Debugging pass is recorded separately; local checks do not establish hidden-test marks. See the [candidate handoff](../artifacts/p1/candidate/HANDOFF.md) and [requirement audit](../artifacts/p1/completion-audit.md).

**P2. Custom simulator plus reward-aware search.** Implement documented movement, pickups, unlocking, damage, box pushing and effect expiry. A sufficient state includes agent position, remaining gem/coin/key/powerup locations, inventory, door states, box positions, health, active effect counters and remaining turns; keep accumulated reward as the path value. A key count alone does not identify which keys remain on the board. Compare transitions with the reference through its public API, including invalid actions and interactions between mechanics.

Use A\* for subproblems whose costs and heuristic assumptions are established. Plain Manhattan distance in **cells** can overestimate remaining **actions** with double-speed boots; obstacle distances can overestimate under phasing. Even a lower bound on action count is not automatically a lower bound on reward cost: collecting a coin can make an action's net cost negative. A* termination/closed-list rules designed for nonnegative costs cannot simply be reused for `3 * turns - 5 * coins`.

For the full game, an anytime branch-and-bound search over the finite remaining-turn horizon is a defensible starting point. First find a winning incumbent, then improve reward. For example, if `C` coins remain and `L` is a proved lower bound on actions needed to win, an optimistic final-reward bound is `score + 5*C - 3*max(0, L-1)` (the last winning turn is free). This bound is weak but safely ignores collectible reachability and other costs; tighten it only with a proof. Prune against the incumbent using valid bounds, and preserve distinct resource states. Extra remaining speed is not necessarily beneficial, so dominance rules for effect timers need particular care.

**P3. Hierarchical planner.** Treat required items and useful interactions as subgoals. Held–Karp subset DP is attractive for a small set of required gems when travel costs are fixed. In the full game those costs depend on keys, opened doors, boxes, health, effect timers and items collected en route. A single shortest path per pair and door configuration can discard necessary alternatives. Use that abstraction for optimistic bounds or candidate routes, or retain resource-sensitive alternatives and verify every macro with the simulator. Keys are interchangeable consumables, not five prescribed key-to-door assignments. Five boxes can still cause large, irreversible search branches; a local solver is a bounded attempt, not a completeness guarantee. Always retain a validated winning incumbent once one is found, with an explicit deadline reserve for execution.

**P4. Unhinged tier.**

- *Mixed-integer programming.* Formulate coin orienteering as a MILP and solve with the HiGHS solver that ships in scipy. Elegant, but risky within 5 seconds on 50-coin levels.
- *Neural-guided search.* Train on planner traces and use predictions for tie-breaking or candidate ordering while retaining valid bounds. Replacing an admissible heuristic with a neural value estimate can lose optimality. See Section 3.10.
- *End-to-end reinforcement learning from pixels.* Low priority under these constraints: it adds training and generalization risk where documented dynamics are already available. Whether it beats a particular bounded planner is an empirical question, not something determinism settles.

### 2.2 Perception options, from sane to unhinged

**V1. Template matching against all public assets.** Match rendered composites across available variants, not just the starter's first variant. Use as a baseline; success on those assets does not establish generalization to the held-out assets.

**V2. Handcrafted features plus scikit-learn.** Colour/HSV statistics, edges and foreground estimates can feed a small classifier. The original sprite's alpha border is not directly available after compositing onto an opaque floor. Measure the serialized classifier: large forests and support-vector sets can exceed 2 MB despite a simple feature extractor.

**V3. Small CNN with label-preserving augmentation.** Train on rendered cells with varied floors, scale, colour and compositing, then export with `generate_torch_loader_snippet`. Hold out entire source sprite variants before augmentation; random frame splits can leak the same sprites into training and validation. Measure whole-board exact parsing as well as per-class accuracy. Parameter count is only a size estimate: with decimal 2 MB and uncompressed base64, the weight-only ceilings are about 375k float32 or 750k float16 parameters before source/serialization overhead. Compression is data-dependent, and float16 need not be faster on CPU.

**V4. Exploit repeated appearance where supported.** The initial frame has no overlaps by assignment guarantee. If the historical same-variant renderer behavior holds, use it to share evidence:

1. Group visually similar normalized crops to share predictions within a frame, after testing that renderer behavior supports this. Repeated crops are evidence of similarity, not semantic labels.
2. Classify floor and wall explicitly; a wall-heavy level can make wall the modal cell. Fractional crop boundaries, scaling and overlaps can also break exact pixel equality.
3. Use the initial-state count constraints: one agent, one exit, equal keys and locked doors, and the documented maxima. They can reject impossible assignments, but multiple wrong assignments may satisfy all constraints.
4. Preserve classifier uncertainty rather than forcing every cluster to a confident label. A constrained maximum-score assignment is still a hypothesis.
5. Probe only when expected information justifies action cost and risk. A probe can consume an item, cause damage, push a box irreversibly or end the episode; it is not guaranteed recovery.

Parse frame 0, simulate internally, and compare observed status plus visually localized agent/changed cells with predictions. `info` agreement alone cannot detect two different positions or layouts with identical counters. On mismatch, reconstruct current inventory/effects as well as the board and replan within the remaining episode budget. Later frames may contain overlaps, so a frame-0-only parser is not sufficient for recovery.

### 2.3 Recommended conventional stack

- Custom simulator with differential fuzzing.
- Reward-aware anytime planner, P2, with P3 abstractions only where their assumptions hold.
- Supervised cell parser, V3 or V2, with V4 consistency checks and visual feedback.
- Tooling: a procedural level generator for stress tests, a single-file bundler that inlines helpers and models and enforces the 2 MB cap, and a benchmark table over the 48 public checks.

---

## 3. Experimental directions

The reference simulator already supplies the dynamics for Task 1. A learned dynamics model is therefore an optional research experiment, not a prerequisite. For Task 2, learning a semantic observation model is useful; a parser alone is not a complete learned world model. In either task, networks can propose plans or search orderings, but checking a candidate through a correct simulator establishes feasibility and reward **conditional on a correct starting state**, not optimality.

Keep separate: transition correctness, perception correctness, plan feasibility, reward quality, and meeting the deadline. A fallback helps only if it can finish within the remaining budget.

### 3.1 Latplan lineage: learn propositional symbols from images, then plan classically

**What it is.** Latplan uses a discrete state autoencoder to connect images to symbolic planning. The extended paper includes both a PDDL construction from transitions and a learned action autoencoder/discriminator for successor generation. Its experiments cover image-based 8-puzzle, Towers of Hanoi and LightsOut. STRIPS-WM learns an abstract transition graph, predicates and operators, then grounds the predicates back into a visual classifier for planning. Neither establishes correctness for arbitrary unseen images. [Latplan, abstract and methods](https://www.alphaxiv.org/abs/1705.00154); [STRIPS-WM, p. 1](https://www.alphaxiv.org/abs/2606.06832).

DWMR learns Boolean states without requiring a pixel decoder, using variance, correlation, coskewness and locality regularization. Its **clean MNIST 8-puzzle** per-cell linear-probe F1 scores are 91 for encoded states and 84 after one imagined step, versus DeepCubeAI's 79 and 77. The corresponding **clean IceSlider** values are 99/95 versus 92/85. These are probe scores, not whole-board parsing accuracy or solved-level rates. IceSlider uses fixed sprite patches; noise robustness is not evidence of generalization to the capstone's held-out visual variants of known entity classes. [DWMR, pp. 8–11, Table 1](https://www.alphaxiv.org/abs/2603.01748).

**How it maps here.** Start with supervised semantic cell labels obtainable from locally generated states. Add temporal consistency as an ablation. Unsupervised Boolean bits need not correspond to “wall”, “gem” or inventory facts; semantic grounding must be learned or supplied. A hard decode followed by a Python simulator is not differentiable, so it is a validation/pseudo-label mechanism unless a differentiable surrogate or explicit gradient estimator is introduced. Dynamics consistency alone admits collapsed or semantically wrong representations and does not guarantee appearance invariance.

**Verdict.** High conceptual fit for symbolic perception; medium implementation priority after a supervised parser. Encode observations outside the search loop, not once per successor.

### 3.2 Latent-graph planning from pixels: PPGS

**What it is.** Planning from Pixels in Environments with Combinatorially Hard Search Spaces learns encoder, forward and inverse models and searches a latent graph with state re-identification. The full planner replans after prediction disagreement and stores observed transitions, using its learned forward model for unseen observation/action pairs. Its encoder uses ResNet-18 and a 16-dimensional normalized latent. [PPGS, §2.3 and Appendix C](https://www.alphaxiv.org/abs/2110.06149).

**How it maps here.** The useful transfer is the observe–predict–check–replan loop. Use known symbolic transitions after perception rather than learned latent successors. Re-identifying two embeddings is an empirical decision: latent aliasing can merge distinct states. Comparing only status counters has a similar limitation.

**Constraint check.** The published encoder exceeds the upload budget; its goal-reaching graph search does not directly solve optional-coin reward optimization. Replanning and model inference must be charged to the episode budget.

**Verdict.** Strong architectural inspiration, not a drop-in implementation.

### 3.3 Object-centric contrastive world models: C-SWM

**What it is.** C-SWM represents states through object slots and relations, predicts transitions with a graph neural network, and trains with a contrastive energy loss. Its experiments include interacting blocks. [C-SWM, abstract and §2](https://www.alphaxiv.org/abs/1911.12247).

**How it maps here.** Object factoring suggests separating boxes, inventory and static geometry. A cell representation avoids a fixed small slot count when many coins are present. The translation bias is a modeling choice, not an exact description of blocked movement, pushing or consumable effects.

**Verdict.** Medium fit as representation background. Re-learning known transitions needs a measured benefit.

### 3.4 JEPA-style latent world models and learned planners, the 2026 crop

**What they are.** Delta-JEPA reconstructs the executed **action from latent displacement**, encouraging action-sensitive representations without reconstructing pixels. AdaJEPA adapts its latent model using observed transitions inside an MPC loop. Hierarchical Planning with Latent World Models divides long tasks into shorter subproblems. [Delta-JEPA, p. 1](https://www.alphaxiv.org/abs/2606.31232); [AdaJEPA, p. 1](https://www.alphaxiv.org/abs/2606.32026); [Hierarchical Planning, p. 1](https://www.alphaxiv.org/abs/2604.03208).

Reinforced Planning learns a critic and plan-update operator on pretrained latent models. Its reported rollout savings compare particular planner configurations; fewer model rollouts do not imply the same wall-clock speedup on a single CPU. The relevant theoretical warning is that predictive accuracy alone does not identify a latent Euclidean distance that represents planning cost. [Reinforced Planning, abstract and latent-geometry analysis](https://www.alphaxiv.org/abs/2608.18669).

**How they map here.** Action-sensitive perception and feedback are useful ideas. The full learned-model control stacks add complexity when exact discrete transitions are already available.

**Verdict.** Complete the parser and symbolic baseline first. After v4, compact latent planning is the strongest research challenger to exact simulation with learned guidance. The LeWM follow-ups below justify testing learned representations and planning costs; their published models and timings do not establish deployment feasibility here.

#### LeWorldModel and follow-ups: post-v4 shortlist

**LeWorldModel (LeWM)** jointly learns a visual encoder and action-conditioned latent predictor with next-embedding prediction and SIGReg anti-collapse regularization. Its planner samples action sequences with CEM, scores predicted endpoints against an encoded goal image, executes a short plan and replans. The original model has approximately 15M parameters; its published planner samples 300 candidates over 10–30 optimization iterations, with training and planning on an NVIDIA L40S. Its four original evaluation tasks use continuous actions. The published model exceeds our upload budget, and its GPU timing does not establish single-CPU feasibility. A compact adaptation needs separate evaluation. [LeWM, §§3–4 and Appendix D](https://www.alphaxiv.org/abs/2603.19312).

The useful distinction is between **prediction quality, the cost used to rank futures, and planning across timescales**. Several follow-ups improve the cost while leaving the world model frozen. That is evidence for testing learned guidance with our exact simulator as well as with learned dynamics.

| Paper | Published improvement and limitation | Proposed use here |
|---|---|---|
| [TEMPO: How Long, Not How Close](https://www.alphaxiv.org/abs/2610.04988), October 2026, §§3–4 | A small temporal-distance network augments the cost of a frozen world model. Three-plan-away TwoRoom success rises from 36% to 99%; difficult Push-T improves more modestly, from 22% to 28%. Training targets elapsed trajectory time, not certified shortest-path distance. | First planning-cost inspiration: rank useful progress beyond a short search horizon. Adapt the target to resources and reward. |
| [Time-to-Goal (T2G)](https://www.alphaxiv.org/abs/2611.05847), 4 November 2026, §§2–3 | A 182K-parameter temporal-ranking metric raises average success from 32.7% to 63.0% across four tasks with a frozen flat model. Cube also changes planning horizon, so its gain is not a cost-only comparison. | Alternative progress supervision to compare separately with TEMPO-inspired regression. The published head is not a required architecture or size. |
| [H-JEPA](https://www.alphaxiv.org/abs/2610.06805), October 2026, revised 1 November, §4 | Different latent spaces and temporal strides support top-down subgoals. Three-level Visual AntMaze success rises from 18% to 73% using less planner compute. Deeper hierarchies hurt Push-T in the reported setting. | Conditional second-stage experiment when accurate short rollouts still fail to select distant objectives; begin with two levels. |
| [SALT: State-Affine Latent Transition](https://www.alphaxiv.org/abs/2609.33595), September 2026, revised 29 October, §4 and Appendix B | Structured transitions and recursive rollout training raise matched average success from 82.7% to 92.7%. Predictor size falls from 11.58M to 0.70M parameters, excluding the visual encoder; experiments use an H100. One-step error is worse despite better planning. | Compact dynamics candidate after a simple predictor baseline; assess decision quality and multi-step failures, not just one-step loss. |
| [Traj-LeWM](https://www.alphaxiv.org/abs/2608.14125), August 2026, §§1–2 | Learned whole-trajectory cost and preference supervision improve four task success rates by 3–14 percentage points. It changes representation training as well as candidate scoring. | Score intermediate pickups, resource consumption and failure risk; isolate scorer effects from changes to representation learning. |
| [Parallel Predictive World Models](https://www.alphaxiv.org/abs/2610.08627), October 2026, §6 | Joint causal trajectory prediction removes within-span decoded-state recursion. Reports 3.30× average CEM speedup at horizon 10 in its evaluation. | Consider only if rollout computation becomes the measured bottleneck; CPU benefit and discrete-game fidelity remain untested. |
| [Neighborhood Graph Cost](https://www.alphaxiv.org/abs/2610.25013), October 2026, §5 and Appendix A | Replaces Euclidean goal distance with graph distances without retraining. Gains persist with an oracle predictor. Its published 5,000-node representation and all-pairs distances occupy about 104 MB. | Offline reference for cost geometry; its stored graph is unsuitable for the 2 MB source budget. |

These are within-paper comparisons under different protocols, not a common leaderboard or measured results on Grid Adventure. New entries are additions to this document, not all papers published since its previous review.

**Objective adaptation.** Grid Adventure has many acceptable terminal states: collect the required gems and win with good reward. A single target screenshot can impose unnecessary constraints. Temporal distance alone also misses optional-coin reward and irreversible actions. TEMPO and T2G use symmetric embedding distances; consuming a key, pushing a box or expiring an effect need not have a reversible counterpart. Neither paper supplies an admissible reward bound for this game. A proposed scorer should condition on remaining objectives, inventory, health, effects and remaining turns, and learn feasible progress and reward preferences. Simulator-generated outcome supervision makes this a supervised hybrid adaptation, rather than a reproduction of reward-free pixel-only LeWM.

**First comparison.** Hold v4 perception fixed and initially supply correct structured observations to both planning tracks. A compact state encoder and latent predictor then test the value of learned planning dynamics. A later pixels-to-latent model is a separate perception/representation experiment. Failure of the structured adaptation would not prove that all visual world models are ineffective, but would weaken the case for spending the remaining capstone budget on one.

### 3.5 Learning to plan with a learned model: Thinker and DRC

**What it is.** Thinker lets a learned agent interact with a world model through imaginary actions, learning how to plan before acting. Its Sokoban and Atari experiments demonstrate learned search behavior; comparisons against DRC and other agents concern those training setups, not an optimality theorem or a CPU deployment guarantee. [Thinker, abstract and experiments](https://www.alphaxiv.org/abs/2307.14993).

**How it maps here.** Useful for studying allocation of planning effort, but training a search policy is a larger project than engineering a bounded symbolic solver.

**Verdict.** Read for research context; low implementation priority.

### 3.6 Energy-based dynamics models for planning

**What it is.** Model-Based Planning with Energy-Based Models learns transition energies and infers trajectories in state space, with actions recovered through inverse dynamics. Its continuous-control experiments motivate flexible trajectory inference. [Du, Lin and Mordatch, abstract and methods](https://www.alphaxiv.org/abs/1909.06878).

**How it maps here.** A learned transition energy could rank competing parses, but known rules are a stronger validity check once the state is identified. An invalid movement command may be a valid no-op transition in the game; do not label all blocked actions as impossible transition negatives.

**Verdict.** Low fit as a replacement simulator; optional perception-consistency experiment.

### 3.7 Implicit policies: Implicit Behavioral Cloning

**What it is.** Implicit BC learns an energy over observation/action pairs and selects low-energy actions. Its main motivation includes multimodal and discontinuous continuous-action policies. [Implicit BC, abstract and methods](https://www.alphaxiv.org/abs/2109.00137).

**How it maps here.** With seven discrete actions, score all actions in a batch. Compare this against an ordinary seven-logit categorical policy: with energies `E(o,a)`, a softmax over `-E(o,a)` already represents a categorical distribution. No continuous optimizer is required, and an EBM advantage here must be measured. Use the ranking for successor ordering without pruning alternatives, or for short candidate rollouts checked by the simulator.

**Verdict.** A small, tractable experiment; categorical behavior cloning is the simpler baseline. Neither policy is guaranteed to meet a reward threshold.

### 3.8 Energy as learned scoring: Energy-Based Transformers

**What it is.** Energy-Based Transformers refine candidate predictions by optimizing a learned energy and can select among multiple candidates. Their “self-verification” is model-based scoring, not a proof that an answer or trajectory is correct. [EBT, abstract and inference method](https://www.alphaxiv.org/abs/2507.02092).

**How it maps here.** A compact energy could rank proposed plans. For this game, replaying a candidate through a validated simulator directly establishes its modeled reward and success; whether a learned ranker saves enough evaluations to pay for itself is untested.

**Verdict.** Low priority. Keep learned scores separate from execution verification.

### 3.9 Diffusion planners

**What it is.** Diffuser denoises state/action trajectories, using conditioning/inpainting for constraints and guidance for reward optimization. Its Maze2D results are normalized benchmark scores, not success percentages or the capstone's performance formula. Energy-based Compositional Diffusion Planning addresses composition of short trajectory segments for longer robotic plans. [Diffuser, methods and Table 1](https://www.alphaxiv.org/abs/2205.09991); [Compositional Diffusion, p. 1](https://www.alphaxiv.org/abs/2606.21646).

**How it maps here.** Discrete diffusion over action tokens is a proposed adaptation, not the published Diffuser formulation. Condition on a parsed symbolic state, sample candidate sequences, replay them, and repair useful candidates with search. Benchmark a categorical autoregressive proposer and simple search first. A 150-token horizon, multiple samples and repeated denoising can consume the whole CPU budget; “20 steps” does not establish feasibility.

**Verdict.** Optional generate-and-verify experiment after the baseline; no evidence here that it beats bounded search on this game.

### 3.10 Recurrent and equilibrium maze solvers, and learned search guidance

**What the papers establish.** Bansal et al. combine recurrent convolution, repeated access to the input (“recall”), and progressive training to extrapolate from small to much larger mazes. Their 74% result on 201×201 mazes is exact output accuracy; examples require about 2,000 iterations. This is evidence for algorithmic extrapolation, not inexpensive inference. [Recurrent algorithm synthesis, §4 and Appendix A](https://www.alphaxiv.org/abs/2202.05826).

Anil et al. connect generalization to **path independence**, meaning similar limiting behavior across initializations. Fixed-point alignment compares different initializations; a small one-run residual measures something else. They explicitly show convergence to a fixed point is not necessary. Neither measure certifies path correctness or heuristic admissibility. [Path Independent Equilibrium Models, §§7–8 and Appendix G](https://www.alphaxiv.org/abs/2211.09961).

Neural A\* learns guidance costs and reports an **optimality–efficiency tradeoff**. It does not preserve shortest-path optimality under the original costs merely by retaining an A\* module. Learning Heuristics for A\* uses penalties for constraint violations; near-perfect empirical constraint satisfaction is not a guarantee on all states, and its reported path costs can be suboptimal. [Neural A\*, §§2–4](https://www.alphaxiv.org/abs/2009.07476); [Learning Heuristics for A\*, §§2–4, Table 1](https://www.alphaxiv.org/abs/2204.08938).

A newer relevant paper, Learning Empirically Admissible Neural Heuristics for Combinatorial Search, uses an asymmetric loss and a validation-calibrated offset. Its claim is no observed admissibility violations under its evaluation protocol, not certification on unseen states. This distinction remains important even when a method explicitly targets admissibility. [Sahay, abstract and calibration method](https://www.alphaxiv.org/abs/2606.04860).

**How it maps here.** For an A\* subproblem with valid nonnegative costs and a proved admissible heuristic, use learned values only to break ties in the primary `g+h` ordering; retain the appropriate reopening rules. For the full reward problem, learned rankings may order branches or suggest incumbents while valid optimistic bounds control pruning and termination. A directly substituted learned heuristic is an approximate planner unless separately certified. Residual gating does not restore the missing guarantee. A spatial cost map also omits inventory, box and effect state unless those are explicitly encoded.

**Constraint check: arithmetic estimates, not timing measurements.** Four dense 3×3 convolutions with equal input/output width on 15×15 cells require approximately:

| Width | MACs per iteration | MACs for 50 iterations | Recurrent weights only |
|---|---|---|---|
| 64 | 33.2M | 1.66G | 147,456 |
| 128 | 132.7M | 6.64G | 589,824 |

These omit input/output heads, recall concatenation, biases and runtime overhead. Weight sharing keeps parameter count small while repeated inference remains expensive. Measure CPU wall-clock for all subgoals and compare against a cached reverse BFS/distance map on the same relaxed graph. Fewer search expansions alone are not a speedup.

**Verdict.** A useful optional ablation after profiling shows a search bottleneck. Tiny action ranking is a lower-cost first experiment than a large recurrent solver.

---

## 4. Recommended experimental stack and acceptance tests

Build Section 2.3 first. Evaluate component changes separately within the same resource budget. After v4 is frozen, the two tracks below may be developed in parallel; use development data for selection and reserve final evaluation for a frozen candidate.

1. **Semantic perception first.** Train a small supervised parser with source-variant splits. Add within-frame grouping and temporal consistency separately. Report per-class confusion, whole-board exact parsing, and downstream wins/reward, including recovery after a deliberately wrong initial parse. A high per-cell score can still produce many wrong boards.
2. **Learned ordering second.** Train a seven-action policy or compact state ranker on planner traces. Compare ordinary supervised behavior cloning against the implicit-BC formulation before trying recurrent value maps. Keep unproved neural estimates out of pruning bounds. Report total time including inference, node expansions, win rate and reward, with identical search deadlines.
3. **Optional plan proposals.** Try a small proposer only if it fits alongside the parser. Replay every candidate, retain the best winning incumbent, and compare with spending that same time on search. Report sample count, valid-plan rate, reward gap and time to first win. An invalid proposal rejected by the simulator has still consumed time.

### Post-v4 decision: shared preparation, two parallel tracks

**Choice:** exact simulation plus learned guidance is the main candidate for improving the submission. A compact world model is the strongest research challenger and should receive one bounded parallel experiment. If only one track can be supported, choose exact simulation plus learned guidance. Keep offline code tuning as an inexpensive control, and use adversarial level generation to support both tracks. These priorities are engineering judgments, not measured success probabilities.

| Approach | Expected value for this project | Main uncertainty | Decision |
|---|---|---|---|
| Exact simulator plus small ranker/value model | Learn which branches deserve computation while preserving known transitions | Whether better decisions save more time than inference costs | Main track |
| Compact WM plus learned planning cost | Learn abstractions and potentially evaluate useful imagined futures cheaply | Rollout errors, CPU cost, and preservation of keys, boxes and timers | Parallel challenger |
| Offline code evolution, U3 | Improve ordering and time allocation with little deployment overhead | Evaluation overfitting and potentially incremental gains | Bounded control within the main track |
| Adversarial level generation, U4 | Supply failures and traces to every approach | Producing representative, legal and solvable cases | Shared preparation |
| Distilled action or short-plan proposer, U2 | Find verified winning incumbents quickly | Compounding mistakes and generalization beyond teacher routes | Next option if a useful planner is too expensive to deploy |
| NCA or recurrent maze solver, U5/§3.10 | Compact learned spatial computation | Benefit over cheap classical paths on resource-sensitive problems | Defer |
| Diffusion/EBM planner, full Dreamer/Thinker, rule induction | Broader research opportunities | Larger integration/training effort and weak evidence under our limits | Defer |
| Test-time parser adaptation, U6 | Possible response to residual appearance errors | Harmful updates and competition with planning time | Reconsider only if failures after v4 qualification justify it |

The learned-cost evidence from TEMPO/T2G supports Track A as well as Track B. SALT supplies a reason to test compact learned dynamics, but its GPU control results do not establish superiority over our compact exact simulator. Known dynamics are a teacher, verification mechanism and experimental control; their availability does not rule out useful learned representations.

#### Step 1: freeze v4 and identify the remaining bottleneck

Preserve the qualified v4 source, model and evidence, along with the frozen P1/P2/P3/V3 artifacts. Do not infer v4 completion from this plan. Profile its exact frozen candidate and classify development failures into observation errors, inability to find a win, poor winning reward, and excessive computation. Use correct structured states to isolate planning, then compare complete image episodes behind the same frozen v4 perception/recovery implementation.

The saved P3 evidence includes 266 default-entrypoint episodes; its additional Task3Agent results use longer budgets and cannot be counted as five-second evidence. Generated-case wins also do not establish optimal reward. See the [v4 design's baseline accounting](superpowers/specs/2026-11-09-v4-vision-recovery-design.md) and [P3 freeze record](../artifacts/p3/frozen/README.md).

If residual failures are predominantly perceptual, record that planner improvements do not directly address them. A separate research result must not be described as repairing the image pipeline without end-to-end evidence.

#### Step 2: prepare a shared corpus and evaluator

- Mix ordinary generated levels with difficult combinations: irreversible box pushes, key allocation, effect expiry, health constraints, coin detours and long routes. Avoid selecting only cases that favor one proposed method.
- Reject illegal levels and retain executable winning witnesses verified through the public simulator. Use certified optima on tractable cases; otherwise label teacher returns best-known feasible rewards.
- Collect successful routes, losing branches, blocked/no-op actions and counterexamples. Include action costs, pickups, resource changes and terminal status, not only geometric paths.
- Separate training, development/selection and final layouts. Adversarial generation and candidate feedback use training/development only. Freeze the final corpus before inspecting candidate outcomes there.
- Preserve artwork exposure history. V3's consumed final artwork cannot become unseen again through new seeds; fresh-layout and genuinely unseen-art claims remain separate.
- Give both tracks the same data access, observation inputs and episode limits. Record corpus hashes, generation seeds, solver budgets and source/model hashes.

#### Step 3: develop two tracks in parallel

**Track A — exact dynamics, better decisions.** First try a bounded change to existing branch/subgoal ordering or time allocation as the code-only control. Then train one small ranker/value model to score consequential branches, subgoals or short routes using resource state and expected reward. Compare its extra inference cost against the existing ordering. Prefer selective or batched calls where they earn their cost; do not assume every node needs neural inference. Keep transition rules, verification, sound pruning bounds and the episode clock fixed. Learned scores order exploration and propose incumbents; they do not become unproved pruning bounds.

[G-RRM, §5](https://www.alphaxiv.org/abs/2607.02491) excludes neural inference from its reported solver timings. [DiBS, Tables 1–2](https://www.alphaxiv.org/abs/2606.06518) illustrates the practical risk: its always-call ablation reduces search work but increases mean runtime by 162.64%. Both motivate complete episode timing here. [SPO](https://www.alphaxiv.org/abs/2609.31179) is useful background for evolving executable route-repair operators, but reproducing its full discovery machinery is outside this first bounded comparison.

**Track B — compact learned dynamics.** Start with a simple compact state encoder/predictor, discrete action embeddings and short latent rollouts. Use the same resource-aware progress/reward objective as Track A. Evaluate multi-step behavior, including rare interactions; low one-step prediction loss alone is insufficient. First establish a compact baseline, then add at most one improvement justified by the observed failure: a learned temporal/trajectory score, SALT-style transition structure, or another targeted correction. Do not combine all paper ideas in the first candidate.

Use the WM to propose candidate routes. Replay executable candidates through the public simulator, charge verification and rejected proposals to the original budget, and retain the best verified winning incumbent. Model-predicted wins and rewards cannot authorize execution or establish pruning thresholds. Verification establishes feasibility conditional on the supplied state; image-mode uncertainty remains v4's responsibility.

For a controlled dynamics ablation, fix the state encoder and scoring head while comparing exact successor states encoded into that space with predicted latent successors. Include encoding costs on the exact path. If joint WM training changes either representation or scorer, report that as a separate ablation rather than attributing the whole gain to learned dynamics. This initial structured-state study is a LeWM-inspired adaptation; pixels-to-latent learning remains a later experiment.

Training and data preparation may overlap on independently available resources. Keep source changes and result directories isolated by candidate, and preserve shared inputs. Run qualification timing sequentially on the designated CPU host with the same affinity/thread settings and no competing workload. Parallel experiments do not permit multiple inference cores or larger episode budgets.

#### Step 4: compare the actual deployable candidates

| Candidate | Dynamics | Guidance | Question answered |
|---|---|---|---|
| Frozen v4/P3 | Exact | Existing | What does the accepted baseline achieve? |
| Tuned planner | Exact | Revised code | Can bounded engineering close the gap? |
| Learned-guidance planner | Exact | Learned score | Does learning improve decisions after its cost is included? |
| WM planner | Learned proposals, exact verification | Matched learned score where practical | Does learned imagination add value beyond the scorer? |

For the last two, hold candidate-generation settings fixed where practical and disclose unavoidable differences. Compare both correct-state planning and full image episodes. Report wins/failures, paired rewards, time to first verified win, reward versus elapsed time, complete episode and cold-load time, peak memory, and serialized source bytes. For the WM also report verification rejection rate and the first consequential rollout mismatch. Search expansions and latent MSE are diagnostics, not the promotion objective.

Use the documented effective-performance formula only where a valid optimum and minimum are available, zeroing failures/timeouts. On generated levels with unknown optima, report wins and paired raw rewards or gaps to an explicitly best-known reference. Do not manufacture normalized performance or hidden-grade estimates.

Select on development data, requiring evidence of useful reward/reliability improvement within the same limits and preservation of accepted regression behavior. Repeat timing sufficiently to distinguish a real gain from deadline noise. Freeze the selected candidate before final evaluation; any subsequent tuning retires that final corpus to development. A method that reduces nodes but loses complete-episode reward or breaches limits is not an improvement.

#### Step 5: let the result determine the next experiment

| Observed result | Next action |
|---|---|
| Learned guidance helps; learned dynamics adds no useful gain | Keep the exact-dynamics candidate. |
| WM proposals help, but rollout errors dominate | Test SALT-style transitions or targeted counterexample training, one change at a time. |
| Short rollouts are reliable, but distant objective selection fails | Try a two-level H-JEPA-inspired planner; explicitly preserve consumable resources and timers across abstraction. |
| WM plans well but inference is too expensive | Distill its decisions into a small ranker or short-plan proposer; requalify the student independently. |
| Rollout computation is the measured bottleneck after a useful WM baseline | Consider parallel trajectory prediction; verify actual single-core benefit. |
| Neither track beats the tuned baseline | Keep v4/the qualified code-only improvement and the stronger corpus; stop adding architectures to this campaign. |

The repository records a **13 November 2026, 23:59** deadline in [README.md](../README.md). For this capstone window, prioritize Track A integration and bound Track B to one compact baseline plus one evidence-driven improvement. Larger H-JEPA, NCA, diffusion and model-based RL campaigns remain separate research choices. This sequence defines what to try; it does not claim training has started, v4 has passed, or new compute has been reserved.

### Evaluation protocol

| Check | Required evidence |
|---|---|
| Simulator fidelity | Transition-by-transition comparison through the public API; targeted combinations of speed, phasing, damage, pickups, doors, boxes, expiry and terminal turns; retain minimized counterexamples. Fuzzing is evidence, not proof. |
| Appearance generalization | Split source sprite variants before rendering/augmentation; keep layout splits separate; vary floors and rendering seeds. Synthetic variants do not reproduce the unknown held-out dataset. |
| Planning quality | Compare with certified optima on small tractable instances; on larger instances report best-known feasible reward and any valid bound separately. A slower solver is not automatically an optimal oracle. |
| Robustness | Run all 48 public checks plus frozen generated cases; keep adversarial training levels separate from final evaluation. Check recovery on overlapping later frames. |
| Deployment cost | Measure complete episodes with one CPU thread, cold model load, parsing, search, verification and repair; report worst-case time, memory and exact final submission bytes. Use an external hard timeout for local benchmarking. |
| Grade reporting | Compute the documented normalized reward formula and zero failures/timeouts; distinguish individual public checks from hidden three-case aggregation. No public-suite score is a hidden-grade estimate. |

### Budget summary

| Constraint | Implication |
|---|---|
| 2 MB submission | Measure the actual serialized source after compression and base64. For compressed payload size `B`, base64 needs `4 * ceil(B/3)` bytes, plus loader/source/container overhead. Parameter ceilings are not acceptance checks. |
| 5 s / 15 s per episode | Reserve time for execution and recovery. Limit model calls and all candidate rollouts with one shared deadline. |
| Zero score when not won | Prioritize a winning incumbent; no method can promise one before it has found and validated it. |
| Held-out asset variants | Evaluate semantic generalization, not repeated appearances of training sprites. |
| Supplementary files | Keep generation scripts, training code and generated datasets as required by the notebook; preserve seeds and provenance for reproducibility. |

---

## 5. The unhinged tier

These are research extensions, not prerequisites for a competitive submission. Prefer expensive work offline and small validated artifacts at runtime. “Shippable” below describes an adapted idea; it does not mean the cited model fits unchanged.

### 5.1 Tier 1: unhinged ideas with a possible small deployment

**U1. Rule induction: infer a simulator from transitions.** Collect action-labeled transitions, synthesize candidate rules, and compare their predictions against the reference through its public interface. PoE-World combines weighted programmatic experts and assumes symbolic observations; it does not solve raw-image perception for this project. From Gameplay Traces to Game Mechanics studies VGDL induction on nine games, but preference for generated explanations/rules is not execution equivalence. [PoE-World, methods](https://www.alphaxiv.org/abs/2505.10819); [Gameplay Traces, abstract and evaluation](https://www.alphaxiv.org/abs/2602.00190).

Use targeted traces for rare combinations, not only random play. Fuzzing cannot establish that every induced rule is correct, and it cannot catch shared bugs if the “oracle” is the same induced implementation. Keep a separately validated documented-rule simulator available. If the rule inducer sees documentation or the reference implementation, do not describe the experiment as discovering rules from observation alone.

**U2. Amortised search: imitate A\* traces.** Searchformer trains on search traces and bootstraps toward shorter traces that still yield good plans. Its 45M-parameter Sokoban result is 93.7% optimal under **Any-optimal-64**: at least one of 64 sampled responses has an optimal plan. The reported 26.8% trace reduction is conditional on optimally answered tasks, not a wall-clock comparison with native A\*. [Searchformer, Table 1, Figure 3 and Appendix D](https://www.alphaxiv.org/abs/2402.14083).

Dualformer drops parts of training traces to support fast, slow and automatic modes. Its 15M-parameter maze model reports 80% optimal in fast mode and 97.6% in slow mode on 30×30 mazes using **1-Optimal-64**, again allowing 64 sampled responses. The published models exceed 2 MB even with ordinary 8-bit weight storage. A much smaller distilled proposer is an untested adaptation; measure one-sample and fixed-time performance explicitly. [Dualformer, §4 and Tables 4.1–4.2](https://www.alphaxiv.org/abs/2410.09918).

**U3. LLM-evolved planner code, offline.** AlphaEvolve uses proposed code changes, evaluators and an evolutionary program database. The transferable idea is to optimize small planner components against reproducible measurements, then ship ordinary source. [AlphaEvolve, overview and methods](https://www.alphaxiv.org/abs/2506.13131).

Freeze the evaluator and a held-out level set. Validate candidate code for transition fidelity and time/memory limits before rewarding speed. Evolved heuristics or reward bounds are not automatically admissible; keep them out of correctness-critical pruning until justified. Plain Python still has runtime and upload constraints. Check and disclose AI assistance under the course's applicable policy.

**U4. Adversarial level generation.** PAIRED uses the return difference between an antagonist and protagonist to guide environment generation. It does not require an omniscient optimal antagonist. Here a high-budget solver can provide a stronger feasible baseline, but its return is only best-known unless optimality is established. [PAIRED, regret formulation](https://www.alphaxiv.org/abs/2012.02096).

Mutate valid levels to expose reward loss, timeouts or perception failures in the bounded agent. Reject constraint violations and independently verify solvability. Track reward gap separately from a known-optimum performance score. Procedural Content Metageneration evolves whole generator programs; its solver-based fitness is useful inspiration, but neither reported dollar cost nor solver success is a universal guarantee. Keep an untouched evaluation set to detect overfitting to the adversarial pool. [Metageneration, method and evaluation](https://www.alphaxiv.org/abs/2608.17947).

**U5. A neural cellular automaton as a proposer.** Pathfinding NCA presents hand-coded BFS/DFS constructions and learned iterative networks. Its shared 48-channel model has 22,464 parameters. Table 2 reports 96.61 normalized accuracy and 96.92% complete paths on 16×16 test mazes, but **88.83 normalized accuracy** on 32×32 mazes; the latter is not an 89% solve rate. Accuracy is defined relative to an all-zero output, whereas “pct. complete” requires exact rounded target matching. [Pathfinding NCA, Table 2 and p. 7](https://www.alphaxiv.org/abs/2301.06820).

That model's raw float32 weights occupy about 90 kB before serialization/base64. Compact weights do not imply fast iteration or exact transitions. Use an NCA for candidate paths or branch rankings, compare it with ordinary BFS, and verify paths. A second learned NCA simulator is a separate fidelity experiment, not a validated replacement for the reference.

**U6. Test-time parser adaptation.** TTT jointly trains a supervised task and a self-supervised auxiliary task, then adapts using the auxiliary signal at test time. Tent instead minimizes prediction entropy, adapting normalization statistics and channel-wise affine parameters. Neither is simply “make structural pseudo-labels and fine-tune everything”. [TTT, §2](https://www.alphaxiv.org/abs/1909.13231); [Tent, §§2–3](https://www.alphaxiv.org/abs/2006.10726).

A grid has few, highly correlated crops and possibly many repeats of one class. Constraint-consistent pseudo-labels can still be wrong, and entropy reduction can reinforce them. Treat adaptation as optional: benchmark a frozen classifier first, reset weights per episode, cap update time, and test harmful-update rates on held-out variants. No sub-second CPU claim has been established here.

| Component | Published artifact or workload | Deployment implication |
|---|---|---|
| Searchformer / Dualformer | Tens of millions of parameters; up to 64 sampled responses in headline metrics | Published configurations do not fit. Distillation and fixed-time decoding need separate evidence. |
| Pathfinding NCA | 22,464 parameters in the cited configuration | Weight size is plausible; iteration cost and Grid Adventure quality remain unmeasured. |
| PoE-World / code evolution | Program synthesis with LLM calls | Offline experiment; ship validated rules/code and retain provenance. |
| PAIRED-style adversary | Repeated comparisons between solvers | Offline stress testing; cost scales with solver budgets and generation volume. |
| Parser adaptation | Backpropagation during evaluation | Competes directly with planning time; require an improvement over the frozen parser. |

### 5.2 Tier 2: larger side experiments

- **Full model-based RL from pixels.** Dreamer-style or Thinker-style agents could be research baselines, but require training and deployment budgets that have not been established here. TD-MPC2 targets continuous control, so a seven-action version needs adaptation.
- **Gradient-based latent planning.** Universal Planning Networks and Reinforced Planning motivate optimizing through a learned model. Relaxed action vectors must eventually be converted to legal discrete actions and replayed. There is no evidence here for either universal superiority or universal failure relative to search.
- **SAT or MILP.** Encoding size and solution time depend on the formulation; a fixed horizon alone does not establish “millions of clauses”. No SAT solver is declared in `pyproject.toml`; SciPy MILP is available through the declared dependency. A bounded coin-routing MILP may help, but an incumbent must be feasible and a time-limit return is not proof of optimality.
- **QUBO / simulated annealing.** Possible as a coin-ordering proposal generator. Compare equal-time reward and feasibility with simpler insertion/local-search and branch-and-bound baselines before claiming an advantage.

### 5.3 Tier 3: the coherent maximally unhinged stack

Call it the agent that discovers the game, with an explicitly experimental rule-learning track:

1. **Learn and ground symbols.** Train a visual representation and semantic decoder; induce rules from action-labeled observations. Separate any documentation-assisted training from a rules-withheld experiment.
2. **Find failures.** Generate valid solvable levels that expose shortcomings of the bounded planner; independently reserve levels and sprites for final evaluation.
3. **Tune offline.** Evolve proposal ranking and time-allocation code against the training pool, preserving trusted transitions and proved bounds.
4. **Compare proposers.** Evaluate a distilled sequence model, an NCA map and a categorical/energy action ranker separately before combining them. Replay proposals, retain winning incumbents, and preserve deadline reserve for the baseline planner.
5. **Report the evidence.** Show induced rules, counterexamples, parsing failures, fixed-time reward curves, serialized sizes and actual runtimes. Distinguish observed feasibility, best-known reward and certified optimality.

Perception, simulation, search and runtime budgeting all remain on the critical path. Differential testing and a fallback reduce risk but do not guarantee the 0.85 performance threshold. GPU resources may help offline training; they do not relax the grader's CPU limit. Preserve supplementary materials as required by the notebook.

---

## 6. Reading list

Most links point to alphaXiv. For the substantive corrections above, this review used retrieved paper text, not just alphaXiv's AI-generated overviews; page/section references identify the supporting passages. Optional reading-list entries are not an exhaustive validation of every claim in those papers. DeepCubeAI was not resolved by the connector's exact-title lookup, so its official RLJ proceedings record is linked instead.

Priority is relative to a chosen research direction: **core** means read before implementing that direction, **support** means consult as needed, **curiosity** means optional context. None of the neural papers is a prerequisite for implementing the baseline. For the baseline, start with the environment docs, PPGS's feedback loop, and the search-guarantee discussion in Section 3.10. For the post-v4 comparison, read LeWM and TEMPO/T2G first; consult SALT, Traj-LeWM and H-JEPA when the measured failure warrants them. The November 9 additions were checked through alphaXiv's retrieved paper text during the planning discussion.

### 6.1 The environment

| Resource | Why |
|---|---|
| [Grid Adventure V1 documentation](https://grid-universe.github.io/grid-adventure-v1/) | Primary mechanics reference; the notebook asks students to use the documentation/tutorial and avoid engine-source inspection. |
| [`tutorial.py`](../tutorial.py), [`capstone-project.py`](../capstone-project.py) | API walkthrough, project constraints, grading formula and supplementary-file requirements. |
| [`utils.py`](../utils.py), [`public_levels/catalog.py`](../public_levels/catalog.py) | Local evaluation semantics, serialization helpers and the 48-check public-suite composition. |

### 6.2 Discrete and symbolic world models from pixels

| Paper | Link | Why | Priority |
|---|---|---|---|
| Classical Planning in Deep Latent Space, Asai and Fukunaga, 2018 | https://www.alphaxiv.org/abs/1705.00154 | Early pixel-to-symbolic planning; discrete autoencoder, PDDL export and successor-function planning. | core |
| Discrete World Models via Regularization, 2026 | https://www.alphaxiv.org/abs/2603.01748 | Boolean representation learning; probe metrics do not establish semantic grounding or whole-board accuracy. | support |
| Planning from Pixels in Environments with Combinatorially Hard Search Spaces, Bagatella et al., 2021 | https://www.alphaxiv.org/abs/2110.06149 | Closest paper to the assignment; latent-graph search with state re-identification and closed-loop replanning. | core |
| STRIPS-WM: Learning Grounded Propositional STRIPS-style World Models from Images, 2026 | https://www.alphaxiv.org/abs/2606.06832 | Visual transition graph to predicates/operators and a grounded classifier; visual rearrangement experiments. | support |
| Learning Discrete World Models for Heuristic Search, Agostinelli and Soltani, RLJ/RLC 2024 | https://rlj.cs.umass.edu/2024/papers/Paper225.html | DeepCubeAI: discrete latent dynamics, a learned heuristic and search. | support |
| Contrastive Learning of Structured World Models, Kipf et al., ICLR 2020 | https://www.alphaxiv.org/abs/1911.12247 | Object slots, graph-network transitions, TransE-style energy loss on pushable-block grids. | support |

### 6.3 Latent world models and learned planners

| Paper | Link | Why | Priority |
|---|---|---|---|
| LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels, 2026 | https://www.alphaxiv.org/abs/2603.19312 | End-to-end predictive representations with SIGReg; baseline for the compact WM research adaptation, not a deployable model under our limits. | core |
| How Long, Not How Close: A Learned Temporal Metric for Planning in Latent World Models, 2026 | https://www.alphaxiv.org/abs/2610.04988 | TEMPO: small temporal-distance head on frozen latents; trajectory time is not certified optimal cost. | core |
| Time-to-Goal: Learned Temporal Metrics for Long-Horizon Planning with JEPA World Models, 2026 | https://www.alphaxiv.org/abs/2611.05847 | T2G: temporal ranking with a frozen WM; compare separately from dynamics changes and account for the Cube horizon change. | core |
| Beyond One-Step Accuracy: State-Affine Latent Transition for Reliable Visual Planning, 2026 | https://www.alphaxiv.org/abs/2609.33595 | SALT: compact structured predictor and recursive rollout training; published parameter savings exclude the encoder. | support |
| Traj-LeWM: Path-Aware World-Model Planning via Latent Trajectory Cost, 2026 | https://www.alphaxiv.org/abs/2608.14125 | Whole-route preference supervision and scoring; isolate representation-training effects from scorer effects. | support |
| H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning, 2026 | https://www.alphaxiv.org/abs/2610.06805 | Separate latent spaces at different timescales; conditional extension for long-horizon objective selection. | support |
| Parallel Predictive World Models for Accurate and Efficient Long-Horizon Planning, 2026 | https://www.alphaxiv.org/abs/2610.08627 | Parallel causal future prediction; consider after rollout computation is a measured bottleneck. | support |
| Latent World Models Should Plan with Distances Along the Latent Manifold, 2026 | https://www.alphaxiv.org/abs/2610.25013 | Neighborhood Graph Cost separates goal-cost errors from predictor errors; published graph storage exceeds our upload budget. | support |
| Reinforced Planning with Latent World Models, 2026 | https://www.alphaxiv.org/abs/2608.18669 | Learned plan updates and critic; predictive accuracy alone does not identify a planning metric. | support |
| Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding, 2026 | https://www.alphaxiv.org/abs/2606.31232 | Decode actions from latent differences without pixel reconstruction. | curiosity |
| AdaJEPA: An Adaptive Latent World Model, 2026 | https://www.alphaxiv.org/abs/2606.32026 | Test-time adaptation of a latent world model. | curiosity |
| Hierarchical Planning with Latent World Models, 2026 | https://www.alphaxiv.org/abs/2604.03208 | Subgoal decomposition against compounding latent error. | curiosity |
| Orthogonal JEPA: Factorized Predictive States, 2026 | https://www.alphaxiv.org/abs/2608.20065 | Factorised latents, relevant to per-cell encodings. | curiosity |
| A Path Towards Autonomous Machine Intelligence, LeCun, 2022 | https://openreview.net/forum?id=BZ5a1r-kVsf | The JEPA position paper. | curiosity |
| DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning, 2024 | https://www.alphaxiv.org/abs/2411.04983 | Planning with pretrained visual features; background for separating perception and dynamics. | curiosity |
| Thinker: Learning to Plan and Act, Chung et al., NeurIPS 2023 | https://www.alphaxiv.org/abs/2307.14993 | Learning to search inside a learned model on Sokoban; read for the analysis of learned search behaviour. | curiosity |
| An Investigation of Model-Free Planning, Guez et al., ICML 2019 | https://www.alphaxiv.org/abs/1901.03559 | DRC, the Sokoban model-free planner Thinker compares against. | curiosity |
| Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model, Schrittwieser et al., 2019 | https://www.alphaxiv.org/abs/1911.08265 | MuZero; the reference point for learned-model tree search. | curiosity |
| World Models, Ha and Schmidhuber, 2018 | https://www.alphaxiv.org/abs/1803.10122 | Neural visual representation, dynamics and controller pipeline. | curiosity |

### 6.4 Energy-based models and diffusion planners

| Paper | Link | Why | Priority |
|---|---|---|---|
| Implicit Behavioral Cloning, Florence et al., CoRL 2021 | https://www.alphaxiv.org/abs/2109.00137 | Action energies; compare with a seven-logit categorical policy for this discrete game. | core |
| Model-Based Planning with Energy-Based Models, Du, Lin, Mordatch, CoRL 2019 | https://www.alphaxiv.org/abs/1909.06878 | EBM over transitions, planning in state space by sampling, online contrastive training. | support |
| Planning with Diffusion for Flexible Behavior Synthesis, Janner et al., ICML 2022 | https://www.alphaxiv.org/abs/2205.09991 | Diffuser; planning as inpainting, Maze2D results, warm-start trick. | support |
| Energy-Based Transformers are Scalable Learners and Thinkers, Gladstone et al., 2025 | https://www.alphaxiv.org/abs/2507.02092 | Prediction refinement and candidate scoring with learned energy; not a correctness certificate. | support |
| Energy-based Compositional Diffusion Planning, 2026 | https://www.alphaxiv.org/abs/2606.21646 | Stitching short trajectories into long plans through energies. | curiosity |
| Structured Denoising Diffusion Models in Discrete State-Spaces, Austin et al., 2021 | https://www.alphaxiv.org/abs/2107.03006 | D3PM, the discrete-diffusion machinery needed for action-token plans. | support |
| Video Models Reason Early: Exploiting Plan Commitment for Maze Solving, 2026 | https://www.alphaxiv.org/abs/2603.30043 | Emergent maze solving in video diffusion; fun context for generate-and-verify. | curiosity |

### 6.5 Learned search, test-time compute, and neural algorithmic reasoning

| Paper | Link | Why | Priority |
|---|---|---|---|
| G-RRM: Guiding Symbolic Solvers with Recurrent Reasoning Models, 2026 | https://www.alphaxiv.org/abs/2607.02491 | Guidance without deleting feasible alternatives; reported solver speedups exclude neural inference. | support |
| DiBS: Diffusion-Informed Branch Selection, 2026 | https://www.alphaxiv.org/abs/2606.06518 | Selective guidance and explicit inference-overhead ablations; fewer nodes need not mean lower runtime. | support |
| Reasoning with Neural Cellular Automata, 2026 | https://www.alphaxiv.org/abs/2609.36126 | Compact maze models with substantial recurrent computation; deferred alternative to the post-v4 WM challenger. | curiosity |
| End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking, Bansal et al., NeurIPS 2022 | https://www.alphaxiv.org/abs/2202.05826 | Recall and progressive loss; extrapolation can require thousands of iterations. | support |
| Path Independent Equilibrium Models Can Better Exploit Test-Time Computation, Anil et al., NeurIPS 2022 | https://www.alphaxiv.org/abs/2211.09961 | Path independence differs from residual convergence; neither certifies correctness. | support |
| Path Planning using Neural A\* Search, Yonetani et al., ICML 2021 | https://www.alphaxiv.org/abs/2009.07476 | Learned guidance and the optimality–efficiency tradeoff; no original-cost optimality guarantee. | core |
| Learning Heuristics for A\*, 2022 | https://www.alphaxiv.org/abs/2204.08938 | Graph-network heuristics with empirical constraint penalties, not certified admissibility. | support |
| Learning Empirically Admissible Neural Heuristics for Combinatorial Search, Sahay, 2026 | https://www.alphaxiv.org/abs/2606.04860 | Asymmetric training and validation offsets; empirical admissibility must not be promoted to a guarantee. | support |
| Neural Algorithmic Reasoning, Veličković and Blundell, 2021 | https://www.alphaxiv.org/abs/2105.02761 | The programme behind learned algorithm execution. | support |
| Value Iteration Networks, Tamar et al., NeurIPS 2016 | https://www.alphaxiv.org/abs/1602.02867 | Value iteration as a convolutional module; the classic grid-world planner-in-a-net. | support |
| iA\*: Imperative Learning-based A\* Search for Path Planning, 2024 | https://www.alphaxiv.org/abs/2403.15870 | A newer learned-A\* variant. | curiosity |
| Tackling GNARLy Problems: Graph Neural Algorithmic Reasoning through Reinforcement Learning, 2025 | https://www.alphaxiv.org/abs/2509.18930 | NAR beyond supervised traces. | curiosity |

### 6.6 Classical planning and combinatorial optimisation

| Resource | Why |
|---|---|
| Held and Karp, A Dynamic Programming Approach to Sequencing Problems, 1962 | Subset DP for small ordering subproblems with fixed travel costs. |
| Vansteenwegen, Souffriau, Van Oudheusden, The Orienteering Problem: A Survey, 2011 | Prize-collecting routing background; the game's resource-dependent transitions require extensions. |
| Russell and Norvig, Artificial Intelligence: A Modern Approach, chapters on informed search and planning | Admissible heuristics, A\*, anytime variants. |
| SciPy `optimize.milp` documentation | MILP support through the declared SciPy dependency; measure runtime and inspect solver status. |
| Culberson, Sokoban is PSPACE-complete, 1997 | Worst-case complexity of generalized Sokoban; not a runtime prediction for these bounded grids. |

### 6.7 The unhinged tier: rule induction, amortised search, program evolution, environment design, cellular automata

| Paper | Link | Why | Priority |
|---|---|---|---|
| PoE-World: Compositional World Modeling with Products of Programmatic Experts, Piriyakulkij et al., NeurIPS 2025 | https://www.alphaxiv.org/abs/2505.10819 | World model as weighted product of LLM-written causal-law programs; the recipe for Section 5.1 U1. | core |
| From Gameplay Traces to Game Mechanics: Causal Induction with Large Language Models, 2026 | https://www.alphaxiv.org/abs/2602.00190 | Compare direct and causal-structure-assisted VGDL rule induction from traces. | support |
| WorldCoder, a Model-Based LLM Agent, 2024 | https://www.alphaxiv.org/abs/2402.12275 | Program-based world models; a comparison point for PoE-World. | support |
| Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping, Lehnert et al., 2024 | https://www.alphaxiv.org/abs/2402.14083 | Searchformer; A\* trace imitation and bootstrapping on maze and Sokoban. | core |
| Dualformer: Controllable Fast and Slow Thinking by Learning with Randomized Reasoning Traces, Su et al., ICLR 2025 | https://www.alphaxiv.org/abs/2410.09918 | Fast/slow/auto modes; headline maze metrics allow 64 samples and published models exceed 2 MB. | core |
| Stream of Search: Learning to Search in Language, Gandhi et al., 2024 | https://www.alphaxiv.org/abs/2404.03683 | The same idea on the Countdown game. | curiosity |
| Transformers Can Navigate Mazes With Multi-Step Prediction, 2024 | https://www.alphaxiv.org/abs/2412.05117 | Multi-step objectives for maze planning in transformers. | curiosity |
| AlphaEvolve: A coding agent for scientific and algorithmic discovery, Google DeepMind, 2025 | https://www.alphaxiv.org/abs/2506.13131 | Evolving marked code blocks against an automatic evaluator; the recipe for U3. | support |
| SPO: Discovering Adaptive Large Neighborhood Search Operators via Stackelberg Program Optimization, 2026 | https://www.alphaxiv.org/abs/2609.31179 | Offline discovery of executable destroy/repair operators; borrow bounded operator ideas before reproducing the full discovery campaign. | support |
| Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design, Dennis et al., NeurIPS 2020 | https://www.alphaxiv.org/abs/2012.02096 | PAIRED; regret-driven level adversaries on a 15x15 grid; the recipe for U4. | core |
| Procedural Content Metageneration via Program Search and Continual Abstraction Discovery, Siper, Khalifa, Togelius, 2026 | https://www.alphaxiv.org/abs/2608.17947 | LLM-evolved Python level generators validated by an A\* solver. | support |
| COvolve: Adversarial Co-Evolution of LLM-Generated Policies and Environments, 2026 | https://www.alphaxiv.org/abs/2603.28386 | Policies and environments co-evolved as a zero-sum game. | curiosity |
| Pathfinding Neural Cellular Automata, Earle, Yildiz, Togelius, Hegde, 2023 | https://www.alphaxiv.org/abs/2301.06820 | Hand-coded and learned NCAs for BFS and DFS; adversarial maze evolution; the recipe for U5. | core |
| Neural Cellular Automata for ARC-AGI, 2025 | https://www.alphaxiv.org/abs/2506.15746 | NCAs as rule learners on grid puzzles. | curiosity |
| Growing Neural Cellular Automata, Mordvintsev et al., Distill 2020 | https://distill.pub/2020/growing-ca/ | The NCA architecture the pathfinding paper builds on. | curiosity |
| Programmable Cellular Automata, 2026 | https://www.alphaxiv.org/abs/2609.06102 | Cellular automata as a programmable local-computation substrate. | curiosity |
| Test-Time Training with Self-Supervision for Generalization under Distribution Shifts, Sun et al., 2020 | https://www.alphaxiv.org/abs/1909.13231 | Per-instance adaptation; the recipe for U6. | support |
| Tent: Fully Test-Time Adaptation by Entropy Minimization, Wang et al., ICLR 2021 | https://www.alphaxiv.org/abs/2006.10726 | Entropy-based adaptation of normalization statistics and affine parameters. | support |
| Universal Planning Networks, Srinivas et al., ICML 2018 | https://www.alphaxiv.org/abs/1804.00645 | Gradient-based planning through a learned latent, Tier 2. | curiosity |
| Kautz and Selman, Planning as Satisfiability, ECAI 1992 | no arXiv record | The SAT encoding discussed in Tier 2. | curiosity |

---

## 7. Compute estimates and training precision

**Planning estimate, 8 November 2026:** reserve roughly **10–30 H100 GPU-hours** for the recommended parser plus small learned action ranker, including a modest set of reruns and ablations. A parser-only conventional stack can use less. Reserve **30–100 GPU-hours total** if adding one or two compact research components such as an NCA, a discrete representation model or a short-plan proposer. Large search-trace teachers and full model-based RL are separate budgets below. None of these estimates predicts a grade or successful convergence.

**Post-v4 scope note, 9 November:** the historical estimates below do not constitute a budget for the newly selected WM challenger, and do not authorize rentals or jobs. Size that bounded experiment from its chosen architecture, dataset and a measured training/inference pilot. Parallel tracks need independent resources; qualification timing remains sequential. The November 8 price snapshots remain dated snapshots.

These are prospective workload envelopes, **not measured H100 runtimes**. They are retained as historical estimates rather than refreshed against the current training implementation. They assume an **H100 SXM 80 GB**, BF16 mixed precision, batched/cached training data, and a working GPU training stack. Other H100 variants and host configurations can differ. NVIDIA lists the SXM memory capacity in its [H100 specifications](https://www.nvidia.com/en-us/data-center/h100/).

One GPU-hour means one GPU occupied for one hour: four H100s for eight hours consume 32 GPU-hours. The estimates exclude CPU-only data generation before training, queue time, infrastructure setup and external LLM API use. They include no assumed FP8 discount. During training, data-loader stalls still consume allocated GPU time.

### 7.1 Workload estimates

Each row describes a proposed **single run**, not a reproduction of every experiment in its cited paper. Ranges allow for different batch sizes, input pipelines and stopping points; the endpoints are not confidence intervals. Do not add all rows unless intending to implement every direction.

| Component | Assumed workload for one run | H100 GPU-hours per run | Deployment implication |
|---|---|---|---|
| Supervised cell parser, V3 | 100k–250k parameters; 0.2–1M crops at 32–64 px; 20–50 epochs | 0.25–2 | Plausible small student; share the 2 MB budget with the planner/ranker. |
| Categorical or energy action ranker, Section 3.7 | 50k–150k parameters; 0.5–5M labeled symbolic states; 5–20 epochs | 0.5–3 | Train on solver traces; teacher-label generation is a separate CPU cost. |
| NCA / compact recurrent guidance | 20k–150k parameters; about 50k updates, batch 32–128, 16–64 recurrent iterations | 0.5–6 | Small weights, potentially expensive CPU inference. |
| DWMR-style representation experiment | 0.1–1M observation transitions at 64–96 px; 20–40 epochs; compact encoder/predictor, roughly 0.2–2M total parameters | 1–8 | Only a suitably small encoder/semantic head can ship; exclude training-only heads from export. |
| Short-plan diffusion or sequence proposer | 0.2–2M parameters; roughly 100k plans of up to 150 actions; 50–100 epochs | 2–12 | Larger versions are teachers; candidate sampling and simulator evaluation need additional time. |
| Searchformer/Dualformer-style teacher | 15M–50M parameters; roughly 100k problems, 100k–400k optimizer updates, capped 1k–4k-token traces | 8–80 | Not directly shippable. Add distillation and generated-label costs; long contexts can exceed this budget. |
| Full Thinker-style RL run | Around 50M real environment steps plus imagined steps; budget 2 H100s for roughly 50–200 hours | 100–400 | Research-scale allowance, not a validated Grid Adventure configuration. Three seeds alone would consume 300–1,200 GPU-hours. |

The small-model ranges are engineering estimates informed by the workload sizes and the published baselines below, not GPU-to-GPU benchmark conversions. For a concrete recommended campaign, six parser runs at 0.25–2 hours, four ranker runs at 0.5–3 hours, and 2–6 hours for profiling, validation and export-related reruns give **5.5–30 GPU-hours**; reserve **10–30** rather than relying on the most optimistic endpoint. This is enough for a bounded comparison, not an exhaustive hyperparameter sweep.

One H100 has ample parameter-storage capacity for these small models. Activation memory depends on batch size, image resolution, sequence length and recurrent unrolling; tune those before assuming that every batch fits. With 2–4 H100s, run independent seeds or variants in parallel first. Distributed training of a tiny model can spend more time synchronizing than it saves. `total_GPU_hours / GPU_count` is only an ideal wall-clock lower bound when the jobs are independent and data is ready.

### 7.2 Published compute anchors

| Source | Reported workload and hardware | What it establishes |
|---|---|---|
| [Recurrent algorithm synthesis, Appendix A.2](https://www.alphaxiv.org/abs/2202.05826) | Maze models train in up to 8 hours on one RTX 2080 Ti. | Recurrent maze learning can fit within a single-GPU experiment; does not measure an H100 port. |
| [Pathfinding NCA, Appendix A.3](https://www.alphaxiv.org/abs/2301.06820) | 50,000 updates per trial; up to 5 hours on a GPU comparable to RTX 2080 Ti; five trials per configuration. | Seed/architecture sweeps multiply an otherwise small run's cost. |
| [DWMR, pp. 9–10](https://www.alphaxiv.org/abs/2603.01748) | IceSlider uses 40,000 training transitions and 20 epochs; batch 256; selected results averaged over ten runs. | The paper's dataset is small; broad tuning and expanded capstone data are additional costs. |
| [Dualformer, §3 and §4](https://www.alphaxiv.org/abs/2410.09918) | 30×30 maze training takes 30 hours on eight V100 32 GB GPUs: 240 V100-GPU-hours. | This is a substantially larger trace-learning experiment than training a cell classifier. |
| [Searchformer, Appendix C and §4.2](https://www.alphaxiv.org/abs/2402.14083) | Sokoban traces were capped at 10,000 tokens; bootstrapping adds generated traces and fine-tuning. | Training-token count and autoregressive rollout generation matter more than parameter count alone. |
| [Thinker, Appendix D, p. 22](https://www.alphaxiv.org/abs/2307.14993) | About 7 days on two A100s for the reported Sokoban setup: 336 A100-GPU-hours; an actor-critic MLP variant takes about 3 days. | Full RL and architecture choices can dominate the project budget. |

Do not divide these wall-clock times by advertised tensor-core TFLOPS ratios. Rendering, environment stepping, small operations, optimizer updates and autoregressive decoding do not scale like one large matrix multiplication. A full Searchformer bootstrapping reproduction also falls outside the 8–80-hour teacher estimate: Dualformer describes three rounds totaling **9.6M sampled responses**, before accounting for their token lengths. [Dualformer, §4.1.2, p. 6](https://www.alphaxiv.org/abs/2410.09918).

### 7.3 CPU data generation, memory and storage

H100 access does not remove the cost of generating good solver labels. As a planning configuration, pair one training GPU with **16–32 CPU cores, 64–128 GB host RAM and 50–200 GB of local SSD space**, then size from a pilot. These are suggested allocations, not measured requirements. Parallelize independent offline levels; the submission itself must still obey the single-core rule.

The dominant CPU budget is approximately `number_of_levels * mean_solver_seconds / 3600` core-hours for a single-core teacher. For **100,000 levels**:

| Mean teacher time per level | CPU core-hours | Ideal elapsed time on 32 cores |
|---|---|---|
| 0.1 s | 2.8 | 5.2 min |
| 1 s | 27.8 | 52.1 min |
| 10 s | 277.8 | 8.7 h |

This excludes rendering, rejected/unsolved levels, memory contention and scheduling overhead. Hard levels have long tails; set per-level budgets and record whether each label is optimal or only best-known. For adversarial level/code search, 100 candidate planners × 1,000 levels × a five-second evaluation cap is up to **138.9 CPU core-hours** of solver runtime, excluding candidate compilation and generation. The cost is on the evaluator even if the proposal-generating model is fast.

Avoid saving every full-resolution frame. One million 64×64 RGBA crops occupy about **16.4 GB** uncompressed; one million pairs of 640×640 RGBA frames occupy about **3.28 TB**. These are illustrative fixed-size arrays, not promises about actual renderer dimensions or compression. Store symbolic states, actions, sprite IDs and seeds, then cache crops or compressed transitions as needed. More augmented crops do not create more independent held-out sprite variants.

### 7.4 Can FP8 or NVFP4 speed training?

**On H100, FP8 is a viable experiment; native NVFP4 is not available.** H100 supports FP8 E4M3/E5M2 arithmetic. Blackwell adds native NVFP4; the stable Transformer Engine NVFP4 training documentation lists SM 10.0/10.3, rather than H100's Hopper architecture. [NVIDIA FP8/FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html); [NVFP4 supported devices](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices).

| Training format | H100 suitability | Decision for this project |
|---|---|---|
| BF16 mixed precision | Native support; establish the throughput/quality baseline. | Default for the small CNN, ranker, NCA and representation experiments. |
| FP8 mixed precision | Native support, but casting/scaling overhead and matrix shapes determine benefit. | Trial on sufficiently large linear layers in a teacher or proposer; keep the compute budget unchanged until measured. |
| Native NVFP4 mixed precision | Requires supported Blackwell training hardware, not H100. | Do not budget an H100 arithmetic speedup from enabling this recipe. |
| FP4 compression with FP8 compute | Specialized software can compress storage/communication on Hopper. | Relevant to large distributed MoEs; not a reason to complicate these small dense models. |

**FP8 evidence and limits.** TorchAO reports **1.10×–1.25×** throughput over BF16 for Llama3-8B on eight H100s, depending on scaling/communication settings. Its analysis explicitly shows that small matrix shapes can be slower in FP8. Those results do not establish a speedup for a 100k-parameter model. The documented automatic conversion targets `nn.Linear`, not an arbitrary Conv2d network; it is not a drop-in acceleration for the parser/NCA. [TorchAO quantized training: H100 benchmarks and performance analysis](https://docs.pytorch.org/ao/stable/workflows/training.html).

H100 hardware does support FP8 convolutions, but that does not establish support or a gain in the chosen PyTorch training path. Benchmark the actual operators. For this project's small networks, budget **1.0× relative to BF16**, with possible regressions; a larger trace teacher is the more promising FP8 candidate. [NVIDIA FP8 introduction](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html#introduction-to-fp8).

**NVFP4 evidence and limits.** On Blackwell, NVIDIA reports **1.31×–1.73× over FP8** for Llama 8B/405B configurations on GB200/GB300. These are large-model, long-sequence training-throughput comparisons, not H100 results. The recipe applies lower precision selectively and retains higher-precision outputs/master weights. The NVFP4 paper demonstrates convergence at 12B parameters and 10T tokens; it does not validate a tiny grid parser. [NVIDIA MaxText benchmark](https://developer.nvidia.com/blog/train-models-faster-with-jax-and-maxtext-using-nvfp4-on-nvidia-blackwell/); [NVFP4 pretraining paper](https://www.alphaxiv.org/abs/2509.25149).

There is a relevant exception to “FP4 cannot help Hopper”: **Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs** stores activations and communicates in **MXFP4**, then executes core matrix operations in **FP8**. It reports a 12.5% throughput improvement at 671B parameters. This is not native NVFP4 compute and its communication/memory bottleneck is absent from our small single-GPU models. [Hopper FP4 paper, abstract and method](https://www.alphaxiv.org/abs/2603.02731).

Training precision, optimizer-state precision, checkpoint storage and CPU inference format are separate choices. FP8/NVFP4 training does not automatically halve total training memory or produce a smaller CPU-ready submission. High-precision weight copies, optimizer states and saved activations still matter. Export and test against the pinned CPU runtime and the combined 2 MB limit. [Transformer Engine memory considerations](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/performance_considerations/performance_considerations.html).

### 7.5 Calibration and practical choice

The current `pyproject.toml` explicitly selects **CPU PyTorch wheels**. Use a separate GPU training environment with compatible CUDA/PyTorch and, if trialing FP8, a compatible Transformer Engine or TorchAO release. Retain the CPU environment for final submission checks; exporting from a newer training stack is not proof of compatibility with the grader's pinned PyTorch 2.7. No environment changes or GPU jobs were launched for this estimate.

Before committing the larger budget:

1. Run a **15–30 minute pilot per architecture**, after compilation/warmup, with the real augmentation and data pipeline. Record samples/tokens per second, peak memory, data-loader wait and validation time. Measure short-job compilation/startup overhead separately.
2. Estimate `training_seconds = epochs * samples / measured_samples_per_second` (or updates × measured step time), then add validation, checkpointing and generation costs. Extrapolate at the intended image size, sequence length and recurrent depth.
3. Compare FP8 with the same BF16 model, batches, data and optimizer. Use matched quality: whole-board parse accuracy, candidate-plan validity, reward and timeout-adjusted performance. For sensitive regularizers, output heads and recurrent updates, retaining BF16/FP32 is a reasonable baseline choice. Faster steps with worse convergence are not a useful speedup.
4. Spend at most **1–2 GPU-hours initially** on FP8 profiling for a promising linear-heavy model. Keep it only if a repeatable wall-clock gain survives validation and export. For tiny CNNs, prioritize batching, cached crops and fused/compiled execution before a custom low-precision implementation.

A pipeline calculation illustrates the limit: if training consumes 30% of elapsed work and CPU data generation/evaluation consumes 70%, making training 1.25× faster improves the whole pipeline by only `1 / (0.70 + 0.30/1.25) = 1.064×`, about **6.4%**. This is an example, not a measured breakdown. Start calibration on **one H100 in BF16**, then use **2–4 H100s for independent experiments** once the data pipeline is ready. Keep the useful-work budget at 10–30 GPU-hours and account separately for scheduling losses, as below.

### 7.6 Current rental prices and campaign cost

**Public pricing checked 8 November 2026; all amounts are USD, before tax.** These are provider-published rates, not reserved capacity or a verified checkout quote. Regional stock, account terms and extra storage can change the bill. Keep SXM and PCIe estimates separate: Section 7.1 assumes H100 SXM, so a cheaper PCIe price cannot be substituted without measuring its runtime.

| Provider / offer | H100 SXM price per GPU-hour | Billing interpretation |
|---|---|---|
| [Hyperstack on-demand](https://www.hyperstack.cloud/gpu-pricing) | $3.20 | Its current SXM catalog lists 8-GPU instances: **$25.60 per node-hour** at the published rate. |
| [Runpod Pods](https://www.runpod.io/pricing) | $3.99 | Published Pod rate; 2/4/8 GPUs at this rate imply $7.98/$15.96/$31.92 per hour. Confirm the selected Pod quote. |
| [Verda on-demand](https://verda.com/pricing) | $4.938 | Published single-GPU instance rate. |
| [Verda spot](https://verda.com/pricing) | $2.469 | Lower price with interruption risk; include checkpoint/restart work and confirm availability. |

Hyperstack's [flavor catalog](https://docs.hyperstack.cloud/docs/hardware/flavors/#h100) distinguishes its eight-GPU SXM configuration from its 1/2/4/8-GPU PCIe configurations. Runpod supports [single-node multi-GPU Pods](https://www.runpod.io/articles/guides/multi-gpu-training-runpod-ddp-deepspeed-instant-clusters); its separate H100 **Clusters** product is quote-based, so the Pod price is not a multi-node cluster quote.

Two distinctions prevent misleading comparisons. [Lambda's public page](https://lambda.com/pricing) currently displays instance rates effective **16 November 2026**, directing users to log in for prices before then; those future rates are not used here. [Vast.ai](https://vast.ai/pricing) uses changing marketplace offers; the live interruptible quotes collected subsequently are recorded separately in Section 7.10, with their requested disk size and bid-floor interpretation. The table above remains a comparison of published provider rates.

The following multiplies the Section 7.1 workload budgets by current rates. It assumes all paid GPU-hours are useful work, with no extra setup, idle allocation or interrupted work. In particular, Hyperstack's lower per-GPU price does not mean that a partly idle eight-GPU node achieves these costs.

| Workload | Useful H100 GPU-hours | At $3.20/GPU-hour | At $3.99/GPU-hour | Spot at $2.469/GPU-hour, before interruptions |
|---|---|---|---|---|
| Recommended parser + ranker campaign | 10–30 | $32–$96 | $39.90–$119.70 | $24.69–$74.07 |
| Campaign including 1–2 compact research components | 30–100 total | $96–$320 | $119.70–$399 | $74.07–$246.90 |
| Trace teacher, one run, excluding bootstrapping | 8–80 | $25.60–$256 | $31.92–$319.20 | $19.75–$197.52 |
| Thinker-style RL, one seed | 100–400 | $320–$1,280 | $399–$1,596 | $246.90–$987.60 |
| Thinker-style RL, three seeds | 300–1,200 | $960–$3,840 | $1,197–$4,788 | $740.70–$2,962.80 |

The 30–100-hour campaign is a total budget, not an addition to the 10–30-hour campaign. The precision section's recommendation is unchanged: no assumed FP8 discount, and no native NVFP4 speedup on H100.

### 7.7 How multiple GPUs change elapsed time and cost

**For this project's small models, parallelize experiments first.** Assign one parser/ranker seed or ablation to each GPU, cache the training data, and schedule the next ready run when a GPU finishes. This reduces campaign elapsed time without adding gradient synchronization. Runpod's guide describes DDP as replicating a model and synchronizing gradients; that is a different operation from running independent experiments. [Multi-GPU training guide](https://www.runpod.io/articles/guides/multi-gpu-training-runpod-ddp-deepspeed-instant-clusters).

For the parser/ranker campaign, define `H` as the single-GPU run-hours summed across ready jobs, `N` as the rented GPU count, `u` as useful scheduling efficiency, and `r` as the hourly price per GPU:

```text
campaign_hours ≈ H / (N * u)
compute_cost ≈ r * H / u
extra_setup_cost = N * r * setup_hours
```

Here `u` describes job imbalance, idle slots and contention relative to running the same jobs individually; it is not the utilization percentage printed by nvidia-smi. The next table is an **illustrative scheduling model**, not measured scaling. The decreasing efficiencies are assumptions to expose the cost of keeping a larger fixed node allocated during a short campaign.

| Configuration | Whole allocation per hour | Assumed `u` | Elapsed for 10–30 useful GPU-hours | Compute cost, before extra setup |
|---|---|---|---|---|
| 1 H100, Runpod rate | $3.99 | 100% baseline | 10–30 h | $39.90–$119.70 |
| 2 H100s, Runpod rate | $7.98 | 90% | 5.56–16.67 h | $44.33–$133.00 |
| 4 H100s, Runpod rate | $15.96 | 80% | 3.13–9.38 h | $49.88–$149.63 |
| 8 H100 SXM node, Hyperstack rate | $25.60 | 65% | 1.92–5.77 h | $49.23–$147.69 |

The lower endpoints require enough short, independent jobs. Actual elapsed time is at least the longest individual run and any serial dependency chain. A parser-selection step, new teacher-label generation or final CPU evaluation cannot disappear by dividing GPU-hours by eight. A ready queue matters more than the node's GPU count. Separately rented single-GPU instances can also release capacity as jobs finish; a fixed eight-GPU node keeps charging for unused slots.

Adding an illustrative **30 minutes of allocation time** for setup/data staging raises four-GPU compute cost to **$57.86–$157.61**, and eight-GPU cost to **$62.03–$160.49**. This is additional infrastructure time, not a second charge for profiling already included in Section 7.1. If only one job remains on the $25.60/hour node, its effective allocation cost is $25.60 per useful GPU-hour until the other slots are filled or released.

For a larger trace teacher, benchmark DDP on 1, 2 and 4 GPUs on the same node before scaling further. Preserve the intended global batch/optimization schedule when comparing time to the same validation quality. If four GPUs achieve 75% scaling efficiency, speedup is `4 * 0.75 = 3×` and GPU-hours/cost at the same per-GPU rate increase by `1 / 0.75 = 1.33×`. That is an example, not an expected result for this teacher. Prefer a verified NVLink/NVSwitch topology for frequent communication; do not infer it solely from “multi-GPU.” Multiple nodes add network requirements and are unnecessary for the initial campaign.

The Thinker allowance already assumes a **two-GPU run** lasting 50–200 hours. It cannot be divided by the GPU count again to claim new scaling. Independent seeds can overlap: three such seeds could occupy six GPUs for 50–200 hours, retaining 300–1,200 useful GPU-hours. Renting a fixed eight-GPU node for that schedule bills **400–1,600 GPU-hours**, or **$1,280–$5,120 at $3.20**, if the two spare GPUs stay idle.

### 7.8 Practical cash budget

Reserve roughly **$200 before tax for the initial parser/ranker campaign on 2–4 H100s**, assuming the data and code are ready before the GPU allocation. This allows the four-GPU scenario above plus a modest host/storage margin. Reserve roughly **$600** for the 30–100-hour expanded campaign: at four GPUs, 80% scheduling efficiency and $3.99/GPU-hour, compute plus 30 minutes of setup is **$157.61–$506.73**. These are spending allowances derived from uncertain runtime estimates, not cost ceilings for achieving a target score.

CPU-only preparation can be much cheaper than leaving GPUs allocated. [Verda lists 32-vCPU/128-GB instances at $0.384/hour](https://verda.com/pricing). If the 100,000-level teacher workload from Section 7.3 achieves ideal 32-worker throughput, its 0.087–8.68 hours cost about **$0.03–$3.33** for mean per-level solve times of 0.1–10 seconds. A vCPU is not necessarily a dedicated physical core; measure throughput and include rendering/rejected levels. Additional dataset searches can multiply this budget.

Persistent storage is separate: [Runpod standard network storage](https://www.runpod.io/pricing) is $0.07/GB-month below 1 TB, or **$7 for 100 GB retained for a month**. Taxes, data transfer where charged, external model APIs and engineering time are excluded. The provider snapshot and all cost arithmetic were checked locally; no capacity was reserved and no training was started.

### 7.9 Other GPUs and accelerators

The H100 budgets answer the original hardware assumption; **H100 is not a minimum requirement**. For the small parser/ranker, start a price/performance pilot with RTX 3090, 4090 or 5090 and use 2–4 independent workers if the campaign has enough ready jobs. Their 24–32 GB memory is a plausible starting point, subject to measured activation memory and batching. For recurrent models, unrolling can matter more than parameter storage. A CPU ranker baseline is also worth timing before assuming it needs an accelerator.

Current published on-demand references, checked 8 November 2026, provide a fallback if spot capacity is unavailable:

| Accelerator | Provider's listed memory | USD per device-hour | Project fit to test |
|---|---|---|---|
| RTX 3090 | 24 GB | $0.50 | Low-cost BF16 baseline for small models. |
| RTX 4090 | 24 GB | $0.89 | Small CNN/ranker and independent seeds. |
| RTX 5090 | 32 GB | $1.19 | More activation headroom; compare actual throughput against 4090. |
| L40S | 48 GB | $1.09 | More memory for recurrence or longer sequences. |
| A100 SXM | 80 GB | $1.79 | Larger batches/teachers and BF16 training. |
| RTX PRO 6000 Blackwell | 96 GB | $2.49 | High-memory CUDA alternative; verify the exact edition. |
| H200 | 141 GB | $5.29 | Consider when memory/bandwidth limits a teacher; little reason to pay for unused capacity. |
| B200 / B300 | 180 / 288 GB | $7.99 / $8.99 | Datacenter Blackwell for a measured larger-model or NVFP4 training experiment. |

These are [Runpod Pod prices](https://www.runpod.io/pricing), not serverless rates or benchmarks. They do not imply stock for a particular node count. Compare complete instance specifications: some inexpensive GPU offers have fewer CPU workers or less host RAM for augmentation.

For an alternative with measured throughput `s` times the H100 throughput and price `r`, ignoring scheduling differences, its relative compute cost is `(r / r_H100) / s`. At the $3.99 H100 on-demand reference, a $0.89 RTX 4090 breaks even at **22.3%** of H100 throughput; a $1.19 RTX 5090 at **29.8%**. This is a break-even calculation, not a prediction of either GPU's speed. Compare time to the same validation quality, including initialization and data loading. The appropriate reference changes when an H100 spot quote is much cheaper.

**AMD:** [Hot Aisle lists MI300X 192 GB VMs at $2.99/GPU-hour](https://hotaisle.xyz/pricing) for 1/2/4 GPUs, and eight-GPU bare metal at $3.39/GPU-hour. These are on-demand references, not verified spot rates. The filtered Vast AMD spot search returned no offers; that does not establish that AMD spot capacity is unavailable everywhere. ROCm PyTorch preserves much of the `torch.cuda` API, but CUDA-specific extensions and low-precision recipes still require validation. For this small campaign, porting effort can exceed the rental saving; reconsider for a memory-heavy teacher or existing AMD access. [PyTorch HIP semantics](https://docs.pytorch.org/docs/2.15/notes/hip.html).

**TPUs:** Google publishes on-demand v5e at **$1.20/chip-hour** in Iowa and v6e/Trillium at **$2.70/chip-hour** in supported US regions. v5e has 16 GB and v6e 32 GB HBM per chip. They are valid training devices, but require a JAX or PyTorch/XLA path and CPU export checks. Compiler startup and changing shapes can undermine a short experiment. A multi-chip VM is billed for every chip; do not compare its per-chip quote to a whole GPU node. [Google TPU pricing](https://cloud.google.com/tpu/pricing), [v5e specifications](https://docs.cloud.google.com/tpu/docs/v5e), [v6e configurations](https://docs.cloud.google.com/tpu/docs/v6e), [PyTorch/XLA compilation behavior](https://docs.pytorch.org/xla/release/r2.8/learn/xla-overview.html). Spot references are below.

**Precision compatibility:** BF16 remains the portable starting point. RTX 5090 and RTX PRO 6000 Blackwell are **SM 12.0**, whereas Transformer Engine 2.20.2 documents NVFP4 **training** for **SM 10.0/10.3**, covering B200/B300-class devices. Its inference support is broader. Do not buy a consumer Blackwell rental on the assumption that this datacenter training recipe works unchanged. Other software paths need their own compatibility and convergence evidence. [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus), [Transformer Engine supported devices](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html#supported-devices).

### 7.10 Live spot and interruptible price snapshot

**Vast.ai public offers queried on 8 November 2026, around 10:44 UTC.** The search used `--type bid`, verified hosts, reliability score above 0.98, currently rentable offers and at least one direct port. It returned 131 single-GPU and 64 four/eight-GPU offers across the selected models. No bid was placed.

The totals below are **minimum compute bid plus the requested disk charge per allocation-hour**. Vast's API states that `type=bid` uses minimum-bid pricing. A floor is not a guaranteed winning bid or uninterrupted reservation; prices and availability can change immediately. The provider's reliability score is not an estimate of how often a spot job will be preempted. [Vast offer API and pricing semantics](https://docs.vast.ai/api-reference/search/search-offers), [interruptible rental behavior](https://vast.ai/article/Rental-Types).

| Single-GPU offer | Nominal VRAM | Region | Compute bid floor/hour | Total/hour with 50 GiB disk | Offer ID |
|---|---|---|---|---|---|
| RTX 3090 | 24 GB | Spain | $0.107 | **$0.125** | 58209852 |
| RTX 4090 | 24 GB | Nevada, US | $0.239 | **$0.253** | 53080136 |
| RTX 5090 | 32 GB | Alberta, Canada | $0.244 | **$0.258** | 45028228 |
| A100 SXM4 | 40 GB | Sweden | $0.333 | **$0.347** | 58855600 |
| A100 SXM4 | 80 GB | Austria | $0.520 | **$0.534** | 58940658 |
| L40S | 48 GB | South Africa | $0.467 | **$0.484** | 56268723 |
| RTX PRO 6000 WS | 96 GB | Texas, US | $0.402 | **$0.407** | 57557323 |
| H100 SXM | 80 GB | Czechia | $1.000 | **$1.047** | 41555524 |

These are the cheapest returned offers within each listed model/capacity group, not a worldwide minimum. The 40 GB A100 quote reports an older driver with CUDA capability 12.2; validate the training image before selecting it. Nominal VRAM labels are rounded product capacities; the snapshot preserves raw reported memory. Transfer fees, tax, off-node checkpoint storage and storage retained while paused are extra.

| Multi-GPU offer | Region | Whole-node floor + 100 GiB disk/hour | Equivalent per GPU-hour | Offer ID |
|---|---|---|---|---|
| 4× RTX 3090 | Taiwan | $0.828 | $0.207 | 59597888 |
| 4× RTX 4090 | Bulgaria | **$1.096** | $0.274 | 26537138 |
| 4× RTX 5090 | Vietnam | **$1.356** | $0.339 | 57894451 |
| 4× A100 SXM4 40 GB | Mississippi, US | $1.931 | $0.483 | 51341854 |
| 8× RTX 4090 | Bulgaria | $2.164 | $0.270 | 26537139 |
| 8× RTX 5090 | Alberta, Canada | $2.847 | $0.356 | 48259598 |

No four/eight-GPU H100 offer matched this filtered query at the time. The RTX nodes above report no NVLink bandwidth, so treat them as independent-worker candidates; a fast collective communication path has not been validated. Do not infer four-GPU availability by multiplying a cheap single-GPU listing. The four-GPU 5090 node costs about **23.7% more** than the four-GPU 4090 node, so it needs about 23.7% higher useful throughput to win on compute cost under equal scheduling/interruption assumptions.

Other published spot references checked during the review:

| Provider / accelerator | Published spot USD/device-hour | Price boundary |
|---|---|---|
| Verda A100 SXM4 80 GB | $1.158 | Single-GPU published instance rate; disk extra. |
| Verda L40S 48 GB | $0.9695 | Single-GPU published instance rate; disk extra. |
| Verda RTX PRO 6000 96 GB | $1.363 | Single-GPU published instance rate; disk extra. |
| Verda H100 / H200 | $2.469 / $3.220 | Published per-GPU rates; confirm instance shape and stock. |
| Verda B200 / B300 | $4.569 / $5.845 | Published per-GPU rates; confirm instance shape and stock. |
| Google TPU v5e | $0.342218 | Published spot-table reference per chip-hour with Iowa selected. |
| Google TPU v6e / Trillium | $0.682658 | Published spot-table reference per chip-hour with Iowa selected. |

Sources: [Verda pricing](https://verda.com/pricing) and [Google Spot VM/TPU pricing](https://cloud.google.com/spot-vms/pricing?hl=en). Google's numbers were retrieved from its indexed pricing table; the full page exceeded the browser tool's size limit, and no live regional billing-SKU or quota query was made. Confirm a supported zone and the actual slice price before budgeting a deployment. These are separate from Google's Flex-start prices. Four v5e chips at the listed reference are about $1.369/hour and four v6e chips about $2.731/hour, before ancillary costs. Low chip prices do not establish better time-to-quality for these models.

#### Campaign cost with spot interruptions

Let `H` be the original 10–30 H100-equivalent workload hours, `s` the measured per-device speed relative to H100, `u` scheduling efficiency, and `w` the paid-work multiplier for checkpoint overhead/replayed work. For a whole-node hourly quote `P` and `N` devices:

```text
active_hours ≈ H * w / (N * s * u)
allocation_cost ≈ P * (active_hours + extra_setup_hours)
```

Queue/preemption downtime additionally extends the calendar schedule and can incur storage charges. Neither `w` nor `s` has been measured here. Assume, illustratively, that each RTX GPU achieves **half of H100 throughput**, four-worker scheduling efficiency is **80%**, replay/checkpoint work adds **25%**, and setup consumes another **30 minutes**. Using the unrounded live node quotes:

| Scenario | Active work time, excluding setup/downtime | Allocation cost including setup and requested disk |
|---|---|---|
| 4× RTX 4090 | 7.81–23.44 h | **$9.11–$26.23** |
| 4× RTX 5090 | 7.81–23.44 h | **$11.27–$32.45** |
| 1× H100 SXM, `s=1`, `u=1`, same 25% overhead | 12.50–37.50 h | **$13.61–$39.79** |

The equal RTX timings deliberately hold throughput constant to show price effects; they are not a claim that 4090 and 5090 perform equally. A higher bid increases the cost. At these floors, a **$50–$100 initial cash allowance** is more useful than the earlier $200 H100 on-demand allowance, provided the checkpointed pipeline works and measured speed supports the scenario. This is not a guaranteed project budget. The on-demand allowance remains the fallback.

Recommended next experiment: a short BF16 pilot on one RTX 5090 (or 4090/3090 if it wins measured cost), plus an H100 reference, then schedule independent seeds on the best-priced 2–4-GPU allocation. Save model, optimizer, scheduler, RNG and data-position state regularly and copy checkpoints off-node; demonstrate resume before relying on spot for a long run. No rental, training, driver change or environment migration was performed in this review.
