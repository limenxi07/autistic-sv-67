
# Diagnostic only: intentionally raises after the first image action.
class Agent(Agent):
    def _step_image(self, observation):
        try:
            return super()._step_image(observation)
        except Exception as exc:
            self._platform_diagnostic_error = type(exc).__name__ + ': ' + str(exc)
            raise

    def step(self, observation):
        if isinstance(observation, GridState):
            return super().step(observation)
        import json
        import sys

        before = perf_counter()
        self._platform_diagnostic_error = None
        action = None
        try:
            action = super().step(observation)
        except Exception as exc:
            self._platform_diagnostic_error = type(exc).__name__ + ': ' + str(exc)
        after = perf_counter()
        torch_module = sys.modules.get('torch')
        info = observation.get('info', {}) if isinstance(observation, dict) else {}
        image = observation.get('image') if isinstance(observation, dict) else None
        report = {
            'error': self._platform_diagnostic_error,
            'reason': self._stats.get('reason'),
            'seconds_before_first_step': round(before - self._budget.started_at, 6),
            'step_seconds': round(after - before, 6),
            'action': str(action),
            'hypotheses': self._stats.get('hypotheses'),
            'unresolved': [str(h.unresolved)[:500] for h in self._hypotheses[:2]],
            'torch_version': getattr(torch_module, '__version__', None),
            'torch_threads': [torch_module.get_num_threads(), torch_module.get_num_interop_threads()] if torch_module is not None else None,
            'python': sys.version.split()[0],
            'image_shape': list(getattr(image, 'shape', ())),
            'image_dtype': str(getattr(image, 'dtype', None)),
            'observation_type': type(observation).__name__,
            'info_keys': sorted(str(key) for key in info),
            'status': info.get('status'),
            'config': info.get('config'),
        }
        raise RuntimeError('V4_PLATFORM_DIAG_95e43962 ' + json.dumps(report, default=str, separators=(',', ':')))
