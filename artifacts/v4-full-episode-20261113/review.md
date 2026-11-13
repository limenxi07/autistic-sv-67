# Read-only review

Reviewer: /root/episode_cost_review. No important correctness or package-binding defects found.

37 focused backend, episode-cost and packaging tests passed; function-scoped notebook execution included. Reviewer independently checked 22,198 key/transition comparisons across22 public levels,8 random trajectories per level, seed38113. Exact raw-pixel keys include shape,dtype,bytes; configure resets cache and byte accounting; returned values do not alias cached arrays. Static reuse excludes dynamic components; original native engine clones inputs. Late dispatch keeps original key/deadline guards.

The reviewer requested a shared-wall mutation test; it was added and passes. The serialization-cost test initially patched only the backend alias; both backend and V4 aliases are now patched and the focused test passes.

Verification boundary: action sequences are replayed separately through the public native engine. Reference search and replay share the optimized converter. Their common converter is independently checked against original specialized public grid_step by repository differential tests and the review experiment; runtime replay alone cannot detect a defect common to both conversions.
