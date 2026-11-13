# Independent review

Read-only reviewer search_repair_review found one deadline propagation gap: after shrinking the refinement deadline, CompactOrdering still held the original deadline. Fixed in both fast and reference search. Regression reproduces an expensive ordering call after a verified win: old elapsed 101.66 ms; fixed 25.04 ms. No remaining important correctness finding.

Packaged-loader regression separately caught a duplicated old loader that disabled caching. Current release has exactly one _NumpyCNN and one get_model and its model has the bounded cache. Earlier candidate measurements are retained but are not evidence for cached release bytes.
