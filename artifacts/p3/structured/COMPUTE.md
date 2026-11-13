# Compute receipt

Host: Omarchy, Intel Core Ultra 9 275HX. CPU0 is a P-core (5.3 GHz advertised maximum), no SMT sibling. Fresh episode workers inherit taskset CPU0 and pin to its single allowed core. Five numerical-library thread limits are set to 1 before each worker spawns. No runtime GPU.

Initial process audit at 12:29 local host time found no ML/benchmark Python work. Reproduction/cap/gem/event development runs used the shared /tmp/grid-adventure-timing.lock, but V3 did not yet use that lock. Therefore early times are exploratory only. At approximately 04:39 UTC the ongoing V3 task requested exclusive timing reservation, confirmed it would use the same lock, and P3 yielded the host. Exact-source final acceptance is scheduled only after explicit release and shared-lock acquisition. Per-episode process checks inspect actual Python processes; waiting flock shell commands cannot cause a mutual wait.

Local diagnostic cProfile times are not acceptance times. A profiled case011 reaching the diagnostic deadline does not contradict the unprofiled development win. Only final locked Omarchy results qualify latency/RSS.

Final correction/receipt: the user selected Performance before P3 actually acquired the lock. Boss suite started 2026-11-09T04:48:14.784692+00:00 and public at 04:48:23.088599+00:00; both recorded profile=performance, EPP=performance. At the first profile audit all completed episodes had performance both before/after. V3 explicitly confirmed it would queue behind P3, with no compute contention. The user-selected profile is left unchanged. Final JSON contains the definitive suite/episode receipts.
