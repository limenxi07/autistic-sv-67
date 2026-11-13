# Focused runtime review

A separate read-only reviewer examined CNN equivalence, in-memory loading,
package validation, tests and absolute-budget recovery. It made no edits or
remote requests. No unresolved Critical or Important findings remained.

Important resolved finding: package validation previously allowed path-based
np.load or allow_pickle=True. A new regression failed both unsafe variants; the
AST validator now requires exactly an io.BytesIO argument and allow_pickle=False.
The candidate loader already met this requirement, so its bytes did not change.

The reviewer found convolution ordering, padding, strides, pooling and flattening
consistent with the trained CellCNN. All exported tensor bytes remain identical;
the recorded numerical comparisons support the conversion. Recovery preserves
the original absolute deadlines.

Validation limitation: NumPy BLAS threading is not a demonstrated blocker. The
local default-thread measurements use Accelerate; Linux evaluation with thread
environment limits does not establish platform-default BLAS behavior. Keep the
platform retest explicitly pending until the exact candidate is run there.
