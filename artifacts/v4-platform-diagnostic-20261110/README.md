# Platform failure diagnostic

The 95e43962 notebook compatibility repair still receives the same user-reported
11 failed image cases on the platform. Read-only inspection of the open Debugging
page confirms **11/22 image cases pass**, correcting the reported 11/12 count.
The successful image cases mean a universal image-path failure is not established.
The previous notebook-scoping bug was real locally, but did not resolve these
platform outcomes.

`task2_agent_diagnostic.py` contains the unchanged repaired source followed by a
diagnostic Agent subclass. It intentionally raises `V4_PLATFORM_DIAG_95e43962`
after the first image action. Use the ungraded Debugging assessment only. This is
not a candidate for graded submission or a performance-qualified artifact.

The report contains the original recovered exception message, emission reason,
constructor-to-first-step gap, first-step duration, input image dimensions/dtype,
metadata contract, Python/Torch versions and Torch thread counts. It does not
change the model, policy, clock or planning code before the report.

Four real-source scenarios were verified: normal module execution and normal
function-scoped notebook execution report RIGHT/no error; a host that has already
configured Torch interop reports its non-reconfigurable interop RuntimeError;
a delayed first call reports cutoff and a four-second pre-step gap. Those are
local reproductions, not a claim that either explains the platform failures.

Production source files and both frozen artifacts remain unchanged in this
investigation. No final corpus, model training or benchmark qualification was
rerun. Await the actual platform diagnostic before choosing another repair.
