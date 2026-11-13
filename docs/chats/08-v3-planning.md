# Image-tracking agent (V3): planning

Session: `01a1bdd2-6f55-7263-ba78-67e1e945d884`  
Started: 2026-11-09T01:00:47.342Z  
Export: 11 visible messages. See [conversation index](README.md).

## enxi — 2026-11-09T01:01:59.948Z
<!-- source-line: 13 -->

Compact-state planner (P2) is stable at 20/22
both are frozen, with your 22/22 Debugging passes recorded:

- P2a — e478c1cc (artifacts/p2/frozen/p2a/task1_agent.py)
- P2b — 551ee003 (artifacts/p2/frozen/p2b/task1_agent.py)
come up with a plan for Image-tracking agent (V3) approach, you have access to a m5080 if needed

## Assistant — 2026-11-09T01:02:05.180Z
<!-- source-line: 17 -->

I’ll review the frozen Compact-state planner (P2) variants and their evidence, then build a Image-tracking agent (V3) plan around the remaining failure modes and where the M5080 could help.

## Assistant — 2026-11-09T01:02:20.860Z
<!-- source-line: 28 -->

I’m using the brainstorming and writing-plans skills to develop the Image-tracking agent (V3) proposal. I’ll treat 20/22 as your reported platform result and keep the recorded Debugging passes separate while checking the local evidence.

## Assistant — 2026-11-09T01:02:59.094Z
<!-- source-line: 60 -->

The hashes match both frozen files. The roadmap uses **Image-tracking agent (V3) for the small CNN image parser**, and Compact-state planner (P2) currently returns `WAIT` for image input. Its local evidence already records 22/22 public wins and 200/200 generated wins, so vision looks like the next useful milestone.

One integration issue matters: replaying a plan validates it against the parsed board; it cannot prove that the board was read correctly. Image-tracking agent (V3) will need visual checks during execution.

## enxi — 2026-11-09T01:03:49.680Z
<!-- source-line: 84 -->

Question: For Image-tracking agent (V3), should I focus on adding Task 2 image perception to the frozen Compact-state planner (P2) planner, or improving Task 1 beyond the reported 20/22? The earlier roadmap treats vision as the next milestone.

Answer: task 1 is now also 22/22

## Assistant — 2026-11-09T01:04:00.474Z
<!-- source-line: 90 -->

Understood—Task 1 is now 22/22 as well. I’ll focus Image-tracking agent (V3) on Task 2 vision: a compact CNN trained on the M5080, feeding the Compact-state planner (P2) planner, with checks for unseen artwork and state tracking after pickups, pushes, and power-up expiry.

I found 333 public sprites across 14 classes. The evaluation will hold out entire sprite variants before generating training examples, so it tests more than memorization of the supplied artwork.

## Assistant — 2026-11-09T01:06:50.823Z
<!-- source-line: 108 -->

The proposal now starts with an integration gate before training: use perfect cell labels to check that image metadata can reconstruct the inventory, effects, and state the planner needs. Compact-state planner (P2)’s inventory ordering makes this worth testing early.

I’m writing up the staged plan and having a fresh reviewer check for missing decisions. The proposed sequence is a roughly 105k-parameter CNN, visual state tracking, then single-core qualification of the complete source—including model loading.

## Assistant — 2026-11-09T01:09:50.139Z
<!-- source-line: 128 -->

The reviewer found no missing decisions that need your input. The plan has six stages, with the observation bridge and CPU startup checks ahead of the larger training runs. It also includes the assignment’s required training code and dataset archive.

I’m checking the documents and frozen hashes once more. No training or implementation has started.

## Assistant — 2026-11-09T01:10:26.023Z
<!-- source-line: 139 -->

With Task 1 now **22/22**, I recommend **Image-tracking agent (V3): a small CNN image parser feeding the frozen Compact-state planner (P2) planner**, with visual checks during execution.

Saved the reviewed [design](/Users/enxi/coding/autistic-sv-67/docs/superpowers/specs/2026-11-09-v3-vision-design.md) and [implementation plan](/Users/enxi/coding/autistic-sv-67/docs/superpowers/plans/2026-11-09-v3-vision.md).

The proposed sequence:

1. **Prove the image-to-planner integration first.** Supply perfect cell labels in a diagnostic harness and verify that image metadata preserves inventory, effects, timers and entity ordering. This catches integration problems before training.
2. **Build an appearance-generalization dataset.** We have 333 public sprites across 14 classes. Split entire sprite variants—including floors—before augmentation, and hold out level layouts separately.
3. **Train a compact CNN on the M5080.** Start with 48×48 crops, roughly 105k parameters and 200k training examples. Run a short throughput pilot before committing to longer runs.
4. **Track state across frames.** Parse the initial board, predict transitions, then check the observed agent position, changed tiles and resource counters. Handle overlaps and invalidate cached actions when observations disagree.
5. **Package for CPU execution.** Embed the model in one source file below 2 MB. Model loading, perception, planning and repair share the original episode budget.
6. **Qualify the exact artifact.** Require all 66 public image episodes to win, preserve structured-input results, and evaluate unseen artwork/layout combinations. Target complete episodes below 4.25 seconds on one CPU core.

The key distinction: **simulator replay validates a plan against the reconstructed board; visual feedback checks whether that reconstruction matches reality.**

Use P2a initially. Compare P2b once vision is integrated—it previously reduced total runtime without improving reward, which may become useful with perception overhead.

Proposed schedule: integration and data on **9 November**, training on **10 November**, tracking on **11 November**, qualification on **12 November**, leaving **13 November** as submission buffer.

Both frozen hashes remain unchanged. No implementation or training has started.

## enxi — 2026-11-09T01:10:39.744Z
<!-- source-line: 146 -->

why such a long timeline?

## Assistant — 2026-11-09T01:11:02.227Z
<!-- source-line: 151 -->

I padded the schedule too much. I assigned a day to each milestone without measuring the work; **there’s no evidence Image-tracking agent (V3) needs four days**.

Compact-state planner (P2) already supplies the simulator, planner and verification. The proposed CNN is small. The uncertain work is image-to-state integration and generalization to unseen sprites—not GPU capacity.

A better **provisional budget** is:

- **First 1–2 hours:** image/metadata bridge, dataset splits, CPU loading check.
- **Next 1–2 hours:** first training run on the M5080 and runnable image agent.
- **Next 2–4 hours:** visual tracking, failure-driven fixes, packaging and qualification.

So I’d target **one focused session, roughly 4–8 hours**, revising that after the first training and integration measurements. Those are estimates, not benchmarked durations.

The checkpoints are worth keeping; spreading them across calendar days wasn’t justified. Extra time should follow a concrete failure—such as unrecoverable state ambiguity or poor unseen-sprite accuracy—not be reserved automatically.
