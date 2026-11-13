# Task 2 and boss runtime repair

User authorized implementation after platform probes confirmed B4 exhausting its 4.65-second execution budget around turn 95 of a 130-action verified route. B3 prepared a winning action at 4.348 seconds. A separate synthetic boss reproduction showed GridState dispatch stopping at 12 seconds with a verified route remaining despite a 15-second evaluator limit.

Preserve model weights, calibration, independent observation checks, public-engine winning replay and frozen P3 bytes. Optimize repeated validation and image bookkeeping without relying on predicted labels as visual evidence. Task 2 deadlines stay 3.0/3.5/4.65 seconds. Boss planning stays 11/12 seconds, with already verified matching structured routes permitted through the existing 14.65-second execution deadline.

Approaches: extending Task 2 deadlines adds too little time for 35 remaining moves; removing checks weakens correctness. Use equivalent computations, bounded reuse and faster dispatch of independently verified states instead. Profile each change, retain only measured improvements, and validate mutations, mismatches, reference fallback and notebook scope.

Current submission bytes are preserved under artifacts/v4-runtime-repair-20261113/baseline. Build candidate artifacts first. Promotion requires the repository tests, exact packaged public runs, and slowdown checks with full evaluator overhead. Local/synthetic results are not platform certification. Use unlimited Debugging for the next platform check; no scored submission is authorized.
