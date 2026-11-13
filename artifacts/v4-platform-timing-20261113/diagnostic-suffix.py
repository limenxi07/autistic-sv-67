
# DIAGNOSTIC ONLY: intentionally raises after the first image action.
_v4_timing = {"cnn": [], "planner": [], "errors": []}

def _v4_timed(original, name):
    def measured(self, value):
        started = perf_counter()
        try:
            return original(self, value)
        finally:
            _v4_timing[name].append({
                "seconds": perf_counter() - started,
                "shape": list(getattr(value, "shape", ())),
                "parsed_board": [[visible_kind(value.grid[x][y]) for x in range(value.width)] for y in range(value.height)] if name == "planner" else None,
            })
    return measured

_NumpyCNN.__call__ = _v4_timed(_NumpyCNN.__call__, "cnn")
PlannerAdapter.step = _v4_timed(PlannerAdapter.step, "planner")

class Agent(Agent):
    def _step_image(self, observation):
        try:
            return super()._step_image(observation)
        except Exception as exc:
            _v4_timing["errors"].append(type(exc).__name__ + ": " + str(exc))
            raise

    def step(self, observation):
        if isinstance(observation, GridState):
            return super().step(observation)
        import json
        import os
        import sys
        import numpy as np

        started = perf_counter()
        action = super().step(observation)
        elapsed = perf_counter() - started
        report = {
            "base_sha256": "832a8450b61fec09076d66bb03a324b10452a6bae0932d5fbb455238c9862c2e",
            "before_first_step": started - self._budget.started_at,
            "step_seconds": elapsed,
            "action": str(action),
            "stages": _v4_timing,
            "stats": self._stats,
            "hypotheses": len(self._hypotheses),
            "unresolved": [str(h.unresolved)[:1000] for h in self._hypotheses],
            "python": sys.version,
            "numpy": np.__version__,
            "image_shape": list(observation["image"].shape),
            "info": observation.get("info"),
            "cpu_count": os.cpu_count(),
            "thread_environment": {k: os.environ.get(k) for k in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "blas": getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas"),
        }
        try:
            from threadpoolctl import threadpool_info
            report["threadpools"] = threadpool_info()
        except Exception as exc:
            report["threadpools_unavailable"] = str(exc)
        raise RuntimeError("V4_TIMING_832a " + json.dumps(report, default=str, separators=(",", ":")))
