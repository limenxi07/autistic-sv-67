# Frozen P2a boss baseline and fast-path coverage

Measured 2026-11-09 on Omarchy, Intel Core Ultra 9 275HX, CPU 0 (5.3 GHz P-core), sequential fresh workers, one library thread. 150 turns; 15-second evaluation watchdog and separate 30-second setup watchdog. No engine implementation source was inspected. All 12 acceptance traces independently replay status and reward. Cold wall includes fresh process startup/import, evaluation and teardown; it is not added to evaluation time. The OS page cache was not flushed.

| Source / internal initial-cutoff | Boss | Wins | Reward | Median episode s | Median cold wall s | Peak RSS bytes | Inner stop |
|---|---|---:|---:|---:|---:|---:|---|
| baseline-frozen / 3.0-3.5 | C0 | 3/3 | -70 | 3.0313 | 3.6833 | 300199936 | deadline |
| baseline-frozen / 3.0-3.5 | C1 | 0/3 | -450 | 3.7383 | 4.3924 | 310583296 | deadline |
| baseline-budget15 / 11-12 | C0 | 3/3 | -68 | 10.0062 | 10.7242 | 398991360 | state_cap |
| baseline-budget15 / 11-12 | C1 | 0/3 | -450 | 12.2557 | 12.9483 | 446066688 | deadline |

Both sources use reference fallback on both bosses. Raising the watchdog alone leaves the 3.0/3.5-second internal search unchanged. The budget copy changes exactly four outer/reference constants. C1 remains a loss; budget-only C0 reaches catalog -68. C0 stops on the 4096 frontier cap with the larger budget. C1 stops on deadline below that cap. Records remain capped at 32768. These observations justify a bounded simulator-coverage change before hierarchy or cap changes.

## First fully reference-verified win (separate diagnostic)

The diagnostic copies add reconstruction and independent public replay at each improving reference incumbent, comparing every state key and final score/status. This costs time and changes throughput: short-budget C0 finishes at -72 rather than acceptance -70. These are not exact-source acceptance timings. In acceptance runs, first verification occurs during the independent post-episode replay and is not presented as time to first search win.

| Instrumented source | C0 first verified win s (three runs, from agent construction) | C1 |
|---|---|---|
| diagnostic-frozen | 0.531564, 0.527936, 0.498892 | no verified win |
| diagnostic-budget15 | 0.524715, 0.536721, 0.523463 | no verified win |

## Rejection audit

| Frozen corpus | Eligible frames | Rejected frames and reason |
|---|---:|---|
| Initial public24 | 21/24 | 3 large keyed inventory (B4, C0, C1) |
| Public22 frozen traces | 736/803 | 67 large keyed inventory |
| Existing generated P2-v2 traces | 6716/6856 | 80 simultaneous pickups/overlapping locks; 60 large keyed inventory |
| Boss acceptance traces (repeat zero) | 0/178 | 178 large keyed inventory |

All 222 legacy public/generated traces preserve recorded status/reward on replay. Eligibility is not all-action equivalence. The guard counts keys + coins + gems, although both bosses have exactly one key.

Every explicit rejection site is enumerated with source lines in `coverage-frozen.json` (24 sites):

- Schema: unknown observation fields/configuration; unknown components/fields/entity shapes; nonstandard reward/damage/speed amounts; nonpositive or noninteger power counters; duplicate identities; wrong agent count.
- Order/overlap: simultaneous collectible pickup or overlapping locks; large keyed resource history. Other overlaps (box/item, boxes, lava/effects) remain represented and need all-action checks.
- Live observation: changed geometry; unknown/duplicate entity; changed kind, lock identity or immutable component; off-board agent; inconsistent reference sets; unknown carried item/effect; invalid active counter; moved static entity; changed uncollected item; missing persistent entity.

Inventory distinctions: competing matching key identities can affect which key is consumed and future locks. With zero or one key, lock traversal order still matters but inventory order cannot choose a different key. Coins/gems never match locks. Their presence, identity and quantity remain represented. Effect identities and their individual counters remain exact; status presentation order alone is normalized by the existing simulator. The candidate must validate this semantic quotient, not emulate arbitrary large-set history.

## Provenance and reproduction

- `artifacts/p2/frozen/p2a/task1_agent.py`: `e478c1ccb78c21ea4e97fa3c54ab08a28a401c9f4e4ca2846dfcb0590bb7c2cb`
- `artifacts/p2-next/p2a-budget15.py`: `2ab981e6990c03dc856fd065839608f863abd37fc486c2401375ead718012e0b`
- `artifacts/p2-next/p2a-diagnostic.py`: `6c7d3c668f9d10a1581bdb53524ae7dff034baeabd9e830319c2080e097ae73f`
- `artifacts/p2-next/p2a-budget15-diagnostic.py`: `30c3edb5ded22783b97ab99439bcb7a7c722024022b52a15ec9ee07459f97972`
- `tools/benchmark_boss.py`: `b72015634c6d7fd8a48d436c112bb6172c01ed4c28bb04f8d53c4e2936e8cc98`

```sh
cd /home/enxi/coding/p2-next-20261109
PY=/home/enxi/coding/p2-fast-planner-20261108/.venv/bin/python
taskset -c 0 "$PY" -m tools.benchmark_boss --source artifacts/p2/frozen/p2a/task1_agent.py --repeat 3 --output artifacts/p2-next/reproduce-frozen.json
taskset -c 0 "$PY" -m tools.benchmark_boss --source artifacts/p2-next/p2a-budget15.py --repeat 3 --output artifacts/p2-next/reproduce-budget15.json
taskset -c 0 "$PY" -m tools.benchmark_boss --source artifacts/p2/frozen/p2a/task1_agent.py --diagnostic-copy artifacts/p2-next/reproduce-instrumented.py --output artifacts/p2-next/reproduce-diagnostic.json
```

Raw reports retain setup/import time, cold wall, episode time, RSS, actions, source/package hashes, affinity, thread settings, fallback use and nested search stop/counters. The runner refuses to overwrite evidence. Budget-only copies and diagnostic copies are experimental and are not V3 integrations or submissions.
