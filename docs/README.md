# Project documentation

Start with the [assignment](assignment.md), then the stage plans below. The [training guide](../training/README.md) gives commands for reproducing the learned models.

## Implementation stages

| Stage | Implementation plan | Design or supporting record |
|---|---|---|
| Reference-state planner (P1) | [Baseline](superpowers/plans/2026-11-08-p1-task1-baseline.md) | [Implementation conversation](chats/01-p1-implementation.md) |
| Compact-state planner (P2) | [Fast planner](superpowers/plans/2026-11-08-p2-fast-planner.md) | [Design](superpowers/specs/2026-11-08-p2-fast-planner-design.md) |
| Boss-search planner (P2-next) | [Recorded work plan](chats/04-boss-search-planning.md) | [Implementation](chats/05-boss-search-implementation.md) |
| Hierarchical planner (P3) | [Measured solver](superpowers/plans/2026-11-09-p3-measured-solver.md) | [Design](superpowers/specs/2026-11-09-p3-hierarchical-planner-design.md) |
| Image-tracking agent (V3) | [Vision pipeline](superpowers/plans/2026-11-09-v3-vision.md) | [Design](superpowers/specs/2026-11-09-v3-vision-design.md) |
| Image-recovery agent (V4) | [Vision recovery](superpowers/plans/2026-11-09-v4-vision-recovery.md) | [Design](superpowers/specs/2026-11-09-v4-vision-recovery-design.md) |
| Task 2 and boss runtime repair | [Runtime repair](superpowers/plans/2026-11-13-runtime-repair.md) | [Design](superpowers/specs/2026-11-13-runtime-repair-design.md), [boss conversation](chats/15-boss-mode-debugging.md) |

Plans describe intended work at the time. Use the implementation conversations and saved reports to distinguish completed work from proposals.

## Other records

- [Conversations](chats/README.md): 15 selected implementation, planning and debugging chats, exported as Markdown in stage order.
- [Development lessons](development-history.md): what the debugging runs taught us.
- [Approaches](approaches.md): research alternatives and reading list.
- [Reproducibility](reproducibility.md): environments, artifacts and verification commands.
- [Rename ledger](renames.md) and [machine-readable mappings](rename-map.json): old and current code names.
- `reproducibility/`: archive parts, file inventory and deduplication manifests used by verification tools.

The implementation plans and design specs are retained for every documented stage. The artifact manifests support reproducibility.
