# Mac shutdown checkpoint — 2026-11-09

User requested local V4 work stop by **19:25 SGT** so the Mac can be switched off.
Do not shut down or change Omarchy or unrelated jobs.

V4 implementation and qualification are complete and frozen:

- Source SHA256 `fb978e3b8bc77462a537f47308701ab909d69979915e59d99f90851af53d3ff6`,693003bytes.
- Final299/300 wins and exact boards; retired300/300; public66/66; structured266/266 and oracle22/22 paired rewards equal.
-349/349 repository tests pass. Regression-qualified; unseen art unproven.
- Exact source, selected export, qualification reports and the verified307706157-byte code/model/evidence archive are local.
- All four runs' actual checkpoints/logits are in that code/model/evidence archive.

**Only bulk dataset retrieval remains.** No more training, candidate changes, or
final evaluation is needed. The consumed final corpus must not be rerun for tuning.

The complete verified dataset archive remains on Omarchy at:
`/home/enxi/coding/v4-vision-recovery-20261109/artifacts/v4/supplementary/generated-datasets.tar.gz`.
Expected5814445999bytes, SHA256
`fba0e540c26d32ea29ea1343e2a180d0d182dc9ba8edf32e5d0dfb86f32754e6`.
The Mac's interrupted prefix is
`artifacts/v4/supplementary/generated-datasets.tar.gz.partial`.
It is intentionally incomplete and is not a verified archive.

After restarting the Mac, run from this repository:

```sh
bash artifacts/v4/reproduce/resume-dataset.sh
```

The command appends remaining bytes, verifies the whole archive and every member,
and renames it only after matching the saved manifest/checksum. It does not fit a
model, alter the candidate, or evaluate a final case. Keep the three required
unchanged V3 parent parts listed in supplementary/parts.json.

For a separate reproduction checkout, restore the code/model/evidence archive
and the small source-completion part listed in parts.json. Extract into a NEW
directory first; do not overwrite concurrent changes in this live workspace.
The source-completion part supplies the plan/spec, project lock files and starter
source that were absent from the remote scratch copy.

The earlier resource-pressure timeout and its separate successful complete
development rerun remain preserved. One older, non-authoritative P3 candidate
was changed to one byte during the session and left untouched; its restoration
copy is under qualification/protected-drift. The authoritative frozen backend
and the qualified V4 source remain unchanged.

See shutdown-checkpoint.json for exact transfer/process/lock state and HANDOFF.md
for the measured result, limitations, all archive hashes, and reproduction.
