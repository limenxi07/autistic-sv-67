# Image-recovery agent (V4): recovery plan

Session: `01a1bec7-8c71-73e0-8b62-918f04f15aeb`  
Started: 2026-11-09T05:28:31.112Z  
Export: 13 visible messages. See [conversation index](README.md).

## enxi — 2026-11-09T05:29:35.397Z
<!-- source-line: 13 -->

plan Image-recovery agent (V4) implementation to fix Image-tracking agent (V3) problems

## Assistant — 2026-11-09T05:29:38.739Z
<!-- source-line: 17 -->

I’ll find the Image-tracking agent (V3) implementation and its known problems, then use that evidence to plan v4. I’m using the brainstorming and writing-plans skills to structure the plan.

## Assistant — 2026-11-09T05:30:06.250Z
<!-- source-line: 44 -->

Image-tracking agent (V3)’s handoff records 295/300 wins, but only 288/300 exact initial boards—below its 99% acceptance gate. Most errors were gems read as coins or ghosts. One loss began with a correct board and then failed during reconciliation, so Image-recovery agent (V4) needs separate perception and tracking fixes. I’m tracing those paths before choosing the implementation sequence.

## Assistant — 2026-11-09T05:31:13.357Z
<!-- source-line: 74 -->

The code reveals a recovery hazard: several fallback paths return `WAIT` without advancing the stored prediction, while reconstruction requires a prediction for the current turn. That could explain repeated failures after the first mismatch; the plan will require a trace to confirm whether it caused the recorded loss.

I’ll keep Image-recovery agent (V4) on the pinned Boss-search planner (P2-next) backend. New layout seeds also won’t make the already evaluated artwork unseen again, so the plan will track that exposure explicitly.

## enxi — 2026-11-09T05:31:33.311Z
<!-- source-line: 82 -->

will provide you with a working Hierarchical planner (P3) implementation to work on later

## Assistant — 2026-11-09T05:31:58.680Z
<!-- source-line: 85 -->

Understood. I’ll plan Image-recovery agent (V4) against the current Boss-search planner (P2-next) backend and keep the planner interface ready for your working Hierarchical planner (P3) implementation later. Hierarchical planner (P3) integration won’t be a dependency of these fixes.

## Assistant — 2026-11-09T05:34:03.837Z
<!-- source-line: 96 -->

The design now covers recovery, artwork discrimination, and qualification as separate stages. I’m having a reviewer check it for missing implementation decisions while I turn it into tasks. The plan will distinguish “regression-qualified” from an unseen-artwork pass if no untouched artwork remains.

## Assistant — 2026-11-09T05:38:00.224Z
<!-- source-line: 118 -->

The reviewer caught three decisions that needed to be explicit: labels for overlapping sprites, which classes the inset view must protect, and how to select the final training checkpoint. I’m resolving those in the plan. The first milestone remains a recovery-only candidate using Image-tracking agent (V3)’s model, so we can verify that fix independently of retraining.

## Assistant — 2026-11-09T05:39:14.340Z
<!-- source-line: 135 -->

Saved the [Image-recovery agent (V4) implementation plan](/Users/enxi/coding/autistic-sv-67/docs/superpowers/plans/2026-11-09-v4-vision-recovery.md) and [design](/Users/enxi/coding/autistic-sv-67/docs/superpowers/specs/2026-11-09-v4-vision-recovery-design.md).

The sequence is:

1. Trace Image-tracking agent (V3)’s five losses and twelve incorrectly parsed boards.
2. Fix recovery synchronization, including fallback `WAIT` actions, using the existing model first.
3. Improve gem/coin/ghost and box/wall discrimination through bounded training and visual-policy comparisons.
4. Qualify the exact packaged candidate, separating known regressions from fresh-layout and unseen-artwork evidence.

**Hierarchical planner (P3) integration is deferred until you provide your working implementation.** Image-recovery agent (V4) currently retains pinned P2-next.

The design received an independent review. Only planning documents were added; no implementation, training, or remote jobs started.

## enxi — 2026-11-09T05:39:49.596Z
<!-- source-line: 142 -->

Hierarchical planner (P3) is done at artifacts/p3/frozen/task1_agent.py

## Assistant — 2026-11-09T05:39:52.522Z
<!-- source-line: 145 -->

I’ll inspect the frozen Hierarchical planner (P3) artifact and its qualification evidence, then update the Image-recovery agent (V4) plan around the available backend.

## Assistant — 2026-11-09T05:40:33.127Z
<!-- source-line: 165 -->

The frozen Hierarchical planner (P3) hash matches `agents/hierarchical_p3.py`. It already exposes a `PlannerAdapter` with the shared 3.0/3.5-second budget, and its freeze record includes the B4 dispatch fix and user-reported Debugging pass.

I’ll add Hierarchical planner (P3) integration as an early Image-recovery agent (V4) milestone, with bridge and deadline checks before retraining. I’ll also adjust the comparisons so planner gains remain separate from perception and recovery gains.

## Assistant — 2026-11-09T05:43:50.782Z
<!-- source-line: 228 -->

Verified the frozen Hierarchical planner (P3) artifact: `d4877284…`, byte-identical to `agents/hierarchical_p3.py`.

Updated the [Image-recovery agent (V4) implementation plan](/Users/enxi/coding/autistic-sv-67/docs/superpowers/plans/2026-11-09-v4-vision-recovery.md) to integrate Hierarchical planner (P3) before recovery and perception work, with adapter checks and separate comparisons to isolate each improvement.

The five-second image budget remains unchanged. Hierarchical planner (P3)’s frozen files are untouched; implementation hasn’t started.
