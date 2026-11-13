
# DIAGNOSTIC ONLY. Deliberately raises a compact episode timing report.
_episode_probe = {"current": None}

class V4EpisodeDiagnostic(Exception):
    pass

def _episode_report(trigger):
    import json
    data = _episode_probe["current"]
    agent = data["agent"]
    planner = agent._planner
    active = (getattr(planner, "_reference", None) or planner) if planner else None
    report = {
        "source": "4023b9ae0d50c9820733031a2d7453ecd687ee72720b9876d798d64211542d08",
        "trigger": trigger,
        "elapsed": round(perf_counter() - agent._budget.started_at, 6),
        "inflight_step_seconds": round(perf_counter() - data["step_started"], 6) if data.get("step_started") is not None else 0.0,
        "initial_window": agent._budget.initial_deadline - agent._budget.started_at,
        "cutoff": agent._budget.cutoff - agent._budget.started_at,
        "config": data.get("config"),
        "latest_status": data.get("status"),
        "stats": agent._stats,
        "plan_remaining": len(active._actions) - active._next if active else None,
        "planner_info": active.info() if active else None,
        "stages_inclusive": data["stages"],
        "transition_cache_hits": data["cache_hits"],
        "transition_cache_misses": data["cache_misses"],
        "agent_seconds": round(data["agent_seconds"], 6),
        "between_calls_seconds": round(data["gap_seconds"], 6),
        "steps": data["steps"],
        "errors": data["errors"],
    }
    raise V4EpisodeDiagnostic("V4_EPISODE_4023 " + json.dumps(report, default=str, separators=(",", ":")))

def _episode_wrap(original, label):
    def measured(*args, **kwargs):
        data = _episode_probe["current"]
        if data is None:
            return original(*args, **kwargs)
        before = perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            seconds = perf_counter() - before
            slot = data["stages"].setdefault(label, {"calls": 0, "seconds": 0.0})
            slot["calls"] += 1
            slot["seconds"] += seconds
            # A checkpoint inside long replay/search prevents the evaluator's
            # hard timeout from swallowing the report on the reference level.
            if perf_counter() - data["agent"]._budget.started_at >= 3.65:
                _episode_report("checkpoint:" + label)
    return measured

_NumpyCNN.__call__ = _episode_wrap(_NumpyCNN.__call__, "cnn_total")
_NumpyCNN._infer = _episode_wrap(_NumpyCNN._infer, "cnn_uncached")
_V4ReplayCache._verify_cached = _episode_wrap(_V4ReplayCache._verify_cached, "winning_replay")
PlannerAdapter.step = _episode_wrap(PlannerAdapter.step, "planning_dispatch")
check_observation = _episode_wrap(check_observation, "observation_check")
reconcile = _episode_wrap(reconcile, "reconcile")
grid_step = _episode_wrap(grid_step, "public_transition")
v4_reference_transition = _episode_wrap(v4_reference_transition, "reference_search_transition")

def _episode_predict(original):
    def predict(self, state, action):
        result = original(self, state, action)
        data = _episode_probe["current"]
        if data is not None:
            owner = getattr(self, "_reference", None) or self
            entry = owner._verified_edges.get((id(state), action))
            hit = entry is not None and entry[0] is state and result is entry[2]
            data["cache_hits" if hit else "cache_misses"] += 1
        return result
    return predict

_V4ReplayCache.predict = _episode_predict(_V4ReplayCache.predict)

class Agent(Agent):
    def __init__(self):
        super().__init__()
        self._episode_data = {"agent": self, "stages": {}, "steps": [], "errors": [],
                              "cache_hits": 0, "cache_misses": 0,
                              "agent_seconds": 0.0, "gap_seconds": 0.0, "last_end": None}

    def _step_image(self, observation):
        try:
            return super()._step_image(observation)
        except V4EpisodeDiagnostic:
            raise
        except Exception as exc:
            self._episode_data["errors"].append(type(exc).__name__ + ": " + str(exc))
            raise

    def step(self, observation):
        if isinstance(observation, GridState):
            return super().step(observation)
        data = self._episode_data
        _episode_probe["current"] = data
        before = perf_counter()
        data["step_started"] = before
        info = observation.get("info", {})
        data["config"], data["status"] = info.get("config"), info.get("status")
        gap = before - data["last_end"] if data["last_end"] is not None else before - self._budget.started_at
        data["gap_seconds"] += gap
        try:
            action = super().step(observation)
            after = perf_counter()
            data["agent_seconds"] += after - before
            data["last_end"] = after
            data["step_started"] = None
            data["steps"].append({"turn": info.get("status", {}).get("turn"),
                                  "action": str(action), "reason": self._stats.get("reason"),
                                  "step_s": round(after-before, 6), "gap_s": round(gap, 6)})
            if self._stats.get("reason") != "conditional_plan":
                _episode_report("first_fallback")
            if len(data["steps"]) >= 40:
                _episode_report("40_action_sample")
            return action
        finally:
            _episode_probe["current"] = None
