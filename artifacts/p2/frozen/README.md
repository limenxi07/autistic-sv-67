# Frozen P2a and P2b

Both variants are frozen after the user's **22/22 website Debugging pass for each** on 9 November 2026 (Asia/Singapore).

- [P2a source](p2a/task1_agent.py): `e478c1ccb78c21ea4e97fa3c54ab08a28a401c9f4e4ca2846dfcb0590bb7c2cb`.
- [P2b source](p2b/task1_agent.py): `551ee00377ffe42a343d18c887b1c7f36ef4baf4fa0fff6a22b0053c9845edd0`.
- [Website report](platform-validation.json) records the exact user message and evidence scope.
- Each variant includes its freeze record, local qualification, regression qualification and differential summary.
- [Frozen bundle](../p2-frozen-e478c1cc-551ee003.zip) includes these exact sources, current handoff records and the unchanged historical source/test/tool/evidence archive.
- The bundle's SHA256 is recorded in `../p2-frozen-e478c1cc-551ee003.zip.sha256` and `../freeze-receipt.json`.

Source and freeze records are read-only. Future development uses a separate version. Both checkpoints are preserved; the existing P2a recommendation and P2b promotion-benefit decision are unchanged by this freeze.

The original `p2-supplementary.zip` and its checksums remain intact as the historical local-qualification snapshot. Its pending-website wording predates this report; the platform-validation record here is authoritative for the new user-reported passes. No graded outcome is inferred. P1 remains unchanged.
