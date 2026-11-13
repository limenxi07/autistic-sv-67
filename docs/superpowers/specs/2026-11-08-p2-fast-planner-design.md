# P2: compact simulator and reward-aware planner

Status: proposed design, 8 November 2026. This request is for a plan; implementation and compute jobs have not started.

## Intent and scope

Build P2 as defined in `docs/approaches.md`: a custom simulator plus reward-aware search for structured Task 1 input. The user reports P1 stable at 20/22 and offers an M5080 if useful. Task 2 image perception is a separate V3 milestone. The compact simulator should later accept a parsed board, but no image model belongs in this milestone.

The first objective is to retain P1's successful cases while making B5/B6 and unfamiliar mechanic combinations tractable on the grader's single CPU. The next objective is better reward within the same deadline. Public success is not a hidden-test score guarantee.

## Evidence and baseline

- `artifacts/p1/platform-results.json` records 20/22 on the latest delivered candidate, with B5 Ow Oof Owie and B6 Warehouse losing without reported timeout or error. Search exhaustion is a hypothesis; the platform action traces are unavailable.
- The candidate associated with 20/22 has SHA256 `ce5a70347773f3eb9609db7f6e9c04f253c1eb0fdc44754c6bb277ce9165929b`, found in the unchanged archive `artifacts/p1/pre-reference/candidate/task1_agent.py` and `artifacts/p1/guidance-score/experiment_agent.py`. The recorded platform submission hash was not explicitly confirmed. The user's latest message confirms stability at 20/22, not the exact bytes pasted.
- Both the working source and current export changed during planning. At a later check, the export hashed to `d27fc4e5eb619f3c7c2302f4772f7087fab8dbc120aec9edcb71a71019a8b3c3` and the working source to `a51380b9771c6cb1c8614a09bbb79bc5a1a06a8dddcb761883b599924c13aea5`; the platform record labels the newer export's retry pending. These are snapshots. Select the baseline by the archived `ce5a...` bytes and recheck all hashes at execution; preserve newer work separately.
- `artifacts/p1/profile-summary.json` attributes 86.2% of older P1b evaluation time to simulation and 9.2% to keys. These measurements predate the current guided ordering. Reprofile the exported candidate, including route estimates and action ordering, before claiming the current bottleneck or speedup.
- Local P1 reports already claim 22/22 on a faster host. P2 must improve constrained runtime and unfamiliar levels, not merely repeat that local score.

## Constraints

Use the assignment and tutorial, public constructors, and black-box `grid_step` experiments. Do not inspect installed engine implementation source. The documentation URL failed to load during this planning pass; local assignment/tutorial material and the reference API remain the implementation authorities.

Runtime limits are 150 turns, 5 seconds for the entire Task 1 episode, 1.5 GB memory, source below 2,000,000 bytes, one CPU core, and no GPU. Keep a self-contained source file with standard-library and allowed Grid Adventure imports. Preserve the previously tested loader compatibility: Python 3.10 grammar, no PEP 695 aliases, and successful execution with a grader prelude and without module registration. Development uses the existing locked environment.

Proposed budgets inherit P1: initial planning ends at 3.0 seconds, every search/repair/verification attempt stops by 3.5 seconds from construction, and measured whole episodes should stay below 4.25 seconds on the recorded validation CPU. No helper starts a fresh allowance. These are engineering targets, not promises about the remote host.

## Architecture and staged choices

**P2a: replace the expensive representation and transitions.** Keep one immutable per-episode board describing dimensions, topology, static walls/lava/exit, item slots and immutable entity metadata. Represent each dynamic state as primitive immutable values: position; remaining gem/coin/key/power-up masks; opened doors; box positions; health; inventory; active effects including multiplicity, relevant identity and remaining counters; exact turn/horizon and terminal flags. Preserve overlap and any ordering/identity that the public API shows affects transitions. Start conservatively; do not collapse stacked effects into one timer without evidence. Python integer masks support the whole 225-cell board and arbitrary assignment-valid coin counts.

Separate accumulated score from the dynamic key only for the documented objective whose transitions/termination do not depend on score. Keep exact turns and effects in the key. Under standard mechanics, exact turns, remaining coins and terminal flags already determine score relative to the episode start; score separation is bookkeeping, not an expected search-space reduction. A generic best-score table may reopen synthetic score improvements, but no performance claim should depend on them. More health, longer boots or more remaining time do not create cross-state dominance rules in this milestone.

Port the exported P1 search priorities and deferred action ordering to the compact representation. Compare deterministic successor choices and expansion traces under a fixed transition budget to isolate the backend change; do not rebuild a GridState for every fast successor. This checkpoint measures simulator, key and ordering costs independently.

**P2b: improve reward after a win.** Preserve the best winning incumbent and continue best-first branch-and-bound over the finite horizon. All seven actions remain available, including ineffective moves and WAIT; they can change timers. Existing route estimates may order work but cannot reject a branch. Coin pickups create negative net costs, so ordinary nonnegative-cost A* closed-list and first-goal rules do not justify optimal reward.

Use a separate proved optimistic reward bound and prune against only a reference-verified incumbent's actual reward. For current score S, C remaining coins, and a lower bound L on actions to win:

`upper_reward = S + 5*C - 3*max(0, L-1)`.

Start with L=0. The next bound may use `ceil(MST/2)` where the Manhattan minimum spanning tree spans the agent, remaining gems and exit, all obstacles are removed, and speed two is granted forever. Any real route connects those targets and moves at most two cells/action, so this relaxation cannot demand more actions than the real route. It ignores pickup costs. Use it only for the verified standard mechanics; test it against exhaustive tiny-map optima. Prune when the upper reward cannot beat the incumbent or when a proved L exceeds the remaining horizon. Reopen score improvements and discard stale queued records. Deadline/cap termination means best found, not optimal.

**Runtime verification.** Encode each live observation and compare it with the predicted next state, including score and dynamic counters. Search calls a verification callback whenever it finds a potentially better winning plan. The callback receives actions and predicted canonical keys and returns status, actual final score and actual keys from public-API replay using the episode's stable Board slots. Compare every predicted intermediate state and score, not only the final win. Only a verified candidate can replace the executable incumbent or set the pruning threshold. Replay shares the deadline; timeout retains the previous verified incumbent, and a simulator disagreement aborts fast search. Cache verified action/prediction sequences. Unexpected live observations invalidate the suffix; repair from the observed state within the original cutoff. Keep a bounded reference-backend search for unsupported schemas or a detected simulator disagreement; no fallback can guarantee rescue after the budget is spent.

Start at P1's 4,096 live states and 32,768 records, then profile compact memory before changing them. Count transposition-table entries, stale heap entries, deferred actions, parent records and verification buffers. Proposed ceilings to evaluate are 65,536 live states and 262,144 retained records, only if peak RSS stays below 1.0 GB. Caps protect the incumbent and cannot silently remove its reconstructible path.

## Correctness and evaluation

1. Differentially compare all actions at reachable states with `grid_step`, including complete semantic state, reward delta and terminal flags. The reference projection must inspect public fields independently of the compact encoder, so a shared dropped field cannot hide a mismatch. Preserve failing seeds and minimize traces.
2. Explicitly cover pickups/overlaps, multiple adjacent doors, key consumption, box chains and blocked pushes, shield consumption, stacked/repeated power-ups, timer boundary turns, boots substeps across hazards/objects/exits, phasing expiry, WAIT/invalid actions, final-turn reward, and death/win/turn-limit ordering. Determine ambiguous event ordering by API probes before coding it.
3. Target at least 100,000 compared transitions, from at least 1,000 layouts, with mandatory mechanic coverage. Include public-level traces, existing witnesses and new fixed-seed generated maps. A count alone does not establish correctness.
4. Use an exhaustive reference oracle on small finite-horizon maps to check optimal reward, duplicate handling, safe bounds and pruning. Generated large maps without proved optima report wins/raw reward; do not label a teacher score optimal.
5. Use disjoint layout families/seeds for development and final qualification. Existing P1 held-out layouts have already informed development; retain them as regression cases and create a new P2 held-out suite. Public22 and B5/B6 remain development diagnostics.

Promotion requires zero known differential mismatches, replay-valid wins, no regressions against the frozen baseline on paired suites, improved constrained performance, and exact-artifact qualification. Aim for 22/22 public wins across three repeats, including B5/B6, with per-case normalized performance at least 0.85 as a conservative local target. The real grading aggregates three different instances per set; three repeats do not reproduce it. Report all failures and resource limits rather than weakening gates after seeing results.

## Compute, alternatives and stop rules

The M5080 GPU is unnecessary for P2a/P2b. Its CPU could run offline differential tests and seed sweeps in independent workers; submitted-agent and acceptance timing stay single core. Hardware identity and current load must be checked before use; no GPU capability or VRAM is inferred from the name.

A learned action ranker is an optional later experiment only if profiling shows search ordering remains the dominant problem. Generate traces with the validated CPU planner, split by layout family, and train a small ranker on the M5080 after identifying its software/hardware. Predictions only order actions. Keep it only if CPU inference, loading and source size included, it beats equal-time unlearned search on untouched layouts. This is outside the initial P2 acceptance gate; V3 perception is the more direct later use for the GPU.

Alternative choices considered: continuing P1 heuristic tuning is cheap but does not remove simulator overhead; jumping to full hierarchical P3 risks losing resource-sensitive paths; learned guidance before a validated fast backend adds data and inference costs. Recommend P2a first, and retain that checkpoint independently if P2b adds no measurable benefit.

Planning estimate: 2–3 focused engineering days for profiling, simulator/differential validation, integration and qualification; simulator edge cases can extend it. Treat the 13 November deadline as a reason to preserve a usable checkpoint after each stage. If simulator mismatches persist after the first focused day, keep P1 operational and reassess scope before spending the remaining schedule. If P2a is correct but not faster end-to-end, reprofile; do not train a model to conceal the regression.

Builder review identified and resolved the candidate-verification interface and the redundant-score distinction. All architecture choices and acceptance targets above are proposals by the assistant, not measured P2 results or user-approved implementation decisions.
