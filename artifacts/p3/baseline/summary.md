# Structured boss baseline and fast-path coverage audit

Measured 9 November 2026 on Omarchy, Intel Core Ultra 9 275HX, CPU 0 (verified P-core), one library thread. Twenty-four sequential fresh-worker episodes: two bosses, four artifacts, three seeds/repeats. All used a 15-second episode watchdog and 150 turns. No timeout/error; all captured traces independently reproduced status and reward. These are structured-input local results, not image-agent or graded results.

Unchanged artifacts retain internal 3.0/3.5-second limits. `budget15` artifacts change only four outer/reference initial/cutoff constants to 11.0/12.0 seconds. Frozen files are untouched. See [manifest](manifest.json), [raw episodes](boss.json), [coverage](coverage.json), and [runner](run_baseline.py).

| Variant | Boss | Wins | Reward (all repeats) | Median episode s | Max RSS MB |
|---|---|---:|---:|---:|---:|
| p2a-frozen | C0 | 3/3 | -70 | 3.031 | 298.7 |
| p2a-frozen | C1 | 0/3 | -450 | 3.741 | 308.7 |
| p2a-budget15 | C0 | 3/3 | -68 | 9.602 | 398.8 |
| p2a-budget15 | C1 | 0/3 | -450 | 12.209 | 451.2 |
| p2b-frozen | C0 | 3/3 | -70 | 3.031 | 298.4 |
| p2b-frozen | C1 | 0/3 | -450 | 3.743 | 310.3 |
| p2b-budget15 | C0 | 3/3 | -68 | 9.509 | 398.8 |
| p2b-budget15 | C1 | 0/3 | -450 | 12.211 | 451.3 |

C0 reaches catalog reward -68 only with the larger budget in these runs; frozen P2 reaches -70. C1 loses at -450 and emits WAIT for all 150 turns in every variant. Both bosses initially reject compact encoding with `Large keyed inventory iteration history`; both use the same reference search, so this comparison does not exercise P2b reward pruning on bosses. More time alone does not resolve C1.

## Coverage

| Corpus | Traces/states | Entire trace fast-eligible | Frames supported / total |
|---|---:|---:|---:|
| initial-public24 | 24 | 21 | 21 / 24 |
| public | 22 | 21 | 736 / 803 |
| p2-v2 | 200 | 160 | 6716 / 6856 |
| boss | 24 | 0 | 0 / 2142 |

Initial public exclusions are B4, C0 and C1, all large keyed inventories. Public22 traces have 67 rejected frames, all that reason; all 22 replay as wins with original rewards. P2-v2 has 80 rejected frames for simultaneous pickups/overlapping locks and 60 for large keyed inventory; 160/200 traces stay entirely eligible, all 200 replay with original win/reward. Boss traces have 2,142 rejected frames, all large keyed inventory. Trace coverage is eligibility from re-encoding each frame, not proof of all-action simulator equivalence or automatic backend switching. Frozen P2 stays on reference fallback once selected.

## Separate instrumented search diagnostic

Two root reference searches used an 11-second deadline with cProfile and internal timing enabled. This changes throughput and is not acceptance timing. See [profile](reference-profile.json) and [runner](profile_reference.py). No engine implementation source was read; function names/times came from runtime profiling.

- C0: stop `deadline`, win True, 6403 transitions, frontier peak 1292/4096, records 2201/32768; simulation 9.622 s, keys 1.039 s. First reference winning successor: 1.959081866996712 s (not independently timed path verification).
- C1: stop `deadline`, win False, 2365 transitions, frontier peak 575/4096, records 907/32768; simulation 9.671 s, keys 0.956 s. First reference winning successor: None s (not independently timed path verification).

C1 spent about 88% of its profiled search in reference simulation, hit its deadline below either cap, and found no winning successor. This supports investigating large-keyed fast-path coverage/reference transition cost first. It does not prove that simulator acceleration alone solves C1. Cap increases and hierarchy are deferred until comparable diagnostics show a remaining need.

## Next implementation decision

Preserve caller deadlines and resource-ID suffix rebinding first. Characterize the large-keyed resource-order boundary with public API probes and independent all-action checks. Keep exclusions in force unless correctness is established. Re-run the boss comparison after a bounded coverage/transition-cost change; only then select further search machinery. Preserve prior fixed simulator edge cases as regressions.

## Verification

Checked 24 unique variant/level/repetition results, source hashes via manifests, 15-second watchdogs, CPU 0 affinity and unprofiled acceptance flags. All 24 boss and all 222 legacy traces reproduce terminal status/reward under their recorded seed. Frozen P2 source hashes were checked locally and remotely. Budget variants are diagnostics and have not received full Task 1/V3 qualification.
