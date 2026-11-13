# V4 frozen: regression-qualified; unseen art unproven

**Delivery complete:** code, models, all four training runs, generated datasets and required parent parts are verified locally. The resumed5.81GB dataset archive and all10604 members match their SHA256 manifest. No training or final evaluation was rerun. See [delivery verification](qualification/delivery-verification.json), [archive parts](supplementary/parts.json) and [restore order](reproduce/README.md).

**V4 passes the regression and fresh-layout gates:299/300 wins and299/300 exact initial boards (99.67% each).** The original unseen-art gate is **not passed**: all333 public artwork groups were previously exposed. No private/graded outcome or platform submission is claimed.

Executable: [frozen/task2_agent.py](frozen/task2_agent.py), **693003 decimal bytes**.
SHA256: `fb978e3b8bc77462a537f47308701ab909d69979915e59d99f90851af53d3ff6`.

## Measured results

| Suite | Wins | Exact initial boards | Comparison |
|---|---:|---:|---|
| Public images |66/66|66/66|No per-case V3 reward regression|
| Retired V3-v5 corpus, clean complete attempt |300/300|300/300|All five old losses repaired; all295 former winning rewards preserved|
| Structured regression |266/266|n/a|Identical paired rewards to frozen P3.Agent; V3 rewards preserved|
| Oracle bridge |22/22|n/a|Identical rewards to matched-budget P3 PlannerAdapter|
| Single fresh-layout final |299/300|299/300|100 disjoint layouts ×3 appearances; generated optima unknown|

Qualification maximum episode **3.645784s**, strict cold **3.857221s**, peak RSS
**332279808bytes (316.9MiB)**. Every row in the qualifying suites meets the
4.25s episode target,5s strict cold watchdog and1GB RSS limit. Omarchy Intel Core
Ultra9 275HX, CPU0, one thread, stable user-selected Performance/EPP, sequential
fresh spawned workers under `/tmp/grid-adventure-timing.lock`. Inference uses CPU
Torch2.7.0; no GPU inference. The parent watchdog starts before process creation.

Critical initial-class recalls: gem120/120 (100%), coin360/360 (100%),
ghost360/360 (100%), box359/360 (99.72%), wall21453/21453 (100%). All other
classes have100% recall. [Qualification](qualification/final-single-attempt/qualification.json)
reports all14 denominators, per-artwork-group support/recall, ten-family results,
remaining losses and exact source/model/policy/data bindings. Zero-support groups
remain explicitly unmeasured.

The sole final failure is `mixed-effects/9db4df970bb6/2035544694701`, generated
layout90078, appearance seed2035544694701, resolution512. One box was classified
as an exit; the episode ended at cutoff after150 recorded actions, reward-445.
This row is both the single loss and the single incorrect initial board. No
post-final model, policy or agent-code tuning occurred. Raw outcomes are in
[final-single-attempt/fresh-300.json](qualification/final-single-attempt/fresh-300.json).

## Earlier failed attempt is preserved

The first retired-corpus attempt stopped on its first row with a strict cold
timeout of **5.004551s**, before a worker result/hash was returned. The runner's
subsequent source-mismatch exception reflected the missing worker hash; the
candidate's actual bytes were unchanged. The failure remains in
[the original attempt](validation/development/attempts/thermal-memory-pressure/retired-v5-300.json),
with its stopped qualification snapshot preserved separately.

Kernel logs shortly before that timeout show package thermal throttling and
memory-pressure/cache flushing. A later10-second idle preflight showed stable
throttle counters and no recent CPU/memory pressure. One separate complete
retired300 development attempt then passed, without row-wise merging, additional
fits, model/policy changes or longer deadlines. Its host bookends remain in
[clean-retired-once](validation/development/attempts/clean-retired-once).
Thermal counters can increase under load; this does not prove the exact cause of
the earlier spike. The qualifying complete attempt met every per-row limit.

One separately labeled diagnostic replay of the original timeout won with an
exact board in1.502584s cold, with0.942562s first-frame perception and0.077829s
planning. It does **not** replace the recorded failure or count as qualification.
See [diagnostic](diagnostics/cold-stop/first-case.json) and
[host events](diagnostics/host-events-cold-timeout.log). Final data was unlocked
only after all22 pre-final checks passed, and consumed exactly once.

## What changed

The backend is the exact frozen P3 source,96112bytes, SHA256
`d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4`.
Default structured dispatch is P3.Agent. Its60 longer-budget Task3Agent rows
remain separate and are excluded from5-second gates. The historical user-reported
P3 Debugging pass is not a V4 platform result.

Every emitted nonterminal action, including WAIT, passes through transactional
prediction and commit. Invalid continuity clears; turns are never rewritten;
terminal inputs do not advance state. One original clock governs initial
planning through+3.00s, verification/repair through+3.50s, and initial reference
refinement through+2.00s. Known covered exits parse from independently checked
history; unknown hidden layers remain unresolved.

The104526-parameter CNN was retrained from scratch with class/group balancing.
Exactly three20-epoch folds (seeds20261109/10/11) and one14-epoch production fit
(seed20261112) were run. Shared epoch14 used pre-calibration full-cell raw board
accuracy, worst-class recall, NLL, then earlier epoch. Held-fold raw board result
was326/360, worst-class recall0.94529; these are development metrics on previously
exposed public art. Production fixed-policy transfer passed120/120 actual-parser
admissible boards and27530/27530 raw cells.

Selected policy: full view, temperature1.8340080864093418, confidence.90,
margin.10. Its temperature grid is geomspace.25–8,41points plus1. The initial
raw-versus-admissible scoring mistake counted120 raw boards despite four parser
rejections. It was corrected from saved logits without another fit, epoch change
or different production tensor weights. Standard reproduction entrypoints now
require actual board contracts for policy admission. Original receipts and exports
remain preserved.

Checkpoint SHA256: `45a6768c467d0bacac481f70e033b4f1d71560c5db49c2553ae4ba1861207690`.
CPU model export SHA256: `d526fd31b579bca243b4d5acdead9a51efe9ddd8f524625aed1ca41bf256e059`.
Policy SHA256: `b765aa605351e9554ff5d1890efdd71026f137750facde7ea9419ceb5021bda6`.
FP32 eager/export logits match exactly for batches1/7/225. The standalone loads
embedded TorchScript bytes lazily and uses no runtime files/assets/network.

The selected model corrects all23 known V3 initial-cell errors. The paired13-case
development union won8/13 with historical V3,8/13 P3-only,8/13 old perception/new
tracking,12/13 new perception/old tracking and13/13 full V4. Both new-perception
corners parsed13/13 initial boards exactly. A repeated unchanged ambiguity now
preserves a verified plan while reconciliation/common-action checks continue;
uncertain actors invoke independent localization even when human is already the
top class. Each repair has a failing-then-passing regression test.
[Attribution](validation/attribution.json) includes all paired reward changes.

## Verification and preservation

The recorded implementation delivery-tree suite passed **349/349**,129.616s.
This retrieval continuation changed delivery metadata only; the suite was not rerun. Scoped Ruff and type checks
pass. Repackaging is byte-identical, including Python3.10 grammar, unregistered
namespace, lazy structured dispatch, model parity and runtime-access denial
checks. [Check receipts](qualification/checks.json) and
[full test log](qualification/all-tests-delivery.log) preserve commands/results.
The fresh implementation review's Important parser-scoring issue was fixed;
the initially deferred trace-stage omission was subsequently fixed when needed
for diagnosis. No Minor findings remain. See [review](qualification/final-review.md)
and exhaustive [execution decisions](qualification/decisions.md).

All qualified V4 runtime component hashes and the authoritative frozen P3/V3
sources match their accepted bytes. The latest read-only delivery inventory
found5889/5889 protected parent files matching. During implementation, the older
generated copy `artifacts/p3/b4-repair/candidate/task1_agent.py` had temporarily
changed to one byte `x` at10:40:21UTC. It is not referenced by V4. This work left
that concurrent edit untouched; the resumed check now observes its original
96112-byte hash again. The earlier observation and exact pre-work copy remain in
[protected-drift](qualification/protected-drift/receipt.json). No broad staging,
merge, push or unrelated README edit was performed.

## Data and reproduction

The fresh corpus was frozen before fits and independently audited disjoint from
905 historical geometries, with100 layouts ×3 appearances, ten families and at
least100 instances of every critical class. All333 artwork groups are exposed;
`unseen_art_status=unproven` and `original_unseen_art_gate_passed=false` remain
explicit even though regression/fresh-layout qualification passes.

Actual generated datasets, all four runs' checkpoints/logits/configs/histories,
original trainer snapshots, current code, tests, manifests, diagnostics and raw
qualification evidence are preserved in the supplementary parts. Their per-file
manifests are stream-verified; [parts.json](supplementary/parts.json) lists sizes,
SHA256s and the three required unchanged V3 parent parts without duplicating their
large archives. The V4 data and model/evidence archives are distinct from the
693003-byte standalone.

[Reproduction instructions](reproduce/README.md) cover CPU checks, exact source
repackaging, the four-fit GPU environment and the saved evaluation recipes. Run
`PYTHONPATH=. UV_CACHE_DIR=/tmp/v4-uv-cache uv run --frozen --no-sync python artifacts/v4/reproduce/assemble.py`
from the repository root to reassemble saved evidence. It never reruns or unlocks
final data. Do not reuse the consumed final corpus for later tuning; a changed
candidate requires a newly frozen final corpus.

The final-delivery metadata part is extracted last, after the immutable code/model,
generated-data and source-completion parts. It preserves these completed delivery
records and the original paused checkpoint. [Delivery status](delivery-status.json)
records the completion time and verification boundaries.
