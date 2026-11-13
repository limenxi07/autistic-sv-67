# V4 delivery complete

The resumed dataset transfer is complete. The 5,814,445,999-byte archive and all
10,604 dataset members match their saved SHA256 manifest. Code, models, all four
training runs, generated data, raw evaluation evidence and required V3 parent
parts are now available locally and verified.

V4 remains frozen at SHA256
`fb978e3b8bc77462a537f47308701ab909d69979915e59d99f90851af53d3ff6`:
299/300 fresh-layout wins and exact initial boards, regression-qualified;
unseen-art generalization remains unproven. The recorded delivery suite passed
349/349 tests. No training or final evaluation was rerun during retrieval.

No work remains pending. See [HANDOFF.md](HANDOFF.md),
[delivery status](delivery-status.json), [verification receipt](qualification/delivery-verification.json)
and [archive parts](supplementary/parts.json).

For reproduction, follow [the restore order](reproduce/README.md) in a NEW
directory; preserve concurrent edits in the live workspace. Saved-evidence
assembly does not unlock or rerun final evaluation. The consumed final corpus
must not be reused for tuning.

The original user-requested shutdown checkpoint is preserved unchanged in
[shutdown-checkpoint.json](shutdown-checkpoint.json) and
[the paused resume note](checkpoints/RESUME-2026-11-09-paused.md).
