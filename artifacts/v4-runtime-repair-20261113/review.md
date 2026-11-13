# Runtime repair review

The final candidate preserves the frozen P3 backend, CNN weights, calibration,
uncertainty checks, public-engine route replay, and absolute episode deadlines.

Independent read-only review covered static conversion, observation validation,
pixel bookkeeping, selected-cell slicing, and boss dispatch. Findings resolved:

- Removed an experimental cached-observation shortcut whose mutation guard did
  not capture all component associations and counter types.
- Restricted direct native static allocation to exact wall/exit classes; custom
  subclasses retain their component iterators.
- Used explicit `v4_observe_key` calls because the submission inliner strips
  local import aliases. Module and notebook tests now check that V4 uses this
  function while frozen P3 still uses its original observer.
- Reproduced and fixed a boss cutoff race: if the frozen delegate returns WAIT
  after crossing 12 seconds, retry the same observation only for an unchanged,
  unconsumed verified route. A real planned WAIT cannot consume two actions.

Final independent review reported no additional actionable findings. The full
suite and exact packaged episodes provide execution evidence separately.

`v4_observe_key` duplicates the validation logic of frozen P3 `read_state`
(SHA256 d487728413d4ec7d24f7a8cbf52f4769f6830653db1549d22c82afb1254bc2f4),
with a direct plain-static-marker branch. Maintain the public-route differential
and mutation tests if either implementation changes. Do not rebind the frozen
observer in the flattened package.
