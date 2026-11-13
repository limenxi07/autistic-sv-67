
# DEBUGGING ONLY: report completion or fallback; no per-stage wrappers.
class Task2EndDiagnostic(Exception):
    pass


class Agent(Agent):
    def __init__(self):
        super().__init__()
        self._end_probe = dict(agent_seconds=0.0, between_calls_seconds=0.0,
            actions=0, first_step_seconds=None, last_end=None, tail=[])

    def _end_report(self, observation, trigger, inflight=0.0):
        import json
        data = self._end_probe
        planner = self._planner
        active = (getattr(planner, "_reference", None) or planner) if planner else None
        report = dict(
            source="0b61f5c7c594541b12656995aa2389642ce37a48ad9f5a2bc3544360c45d92d5",
            trigger=trigger,
            elapsed=round(perf_counter() - self._budget.started_at, 6),
            agent_seconds=round(data["agent_seconds"], 6),
            between_calls_seconds=round(data["between_calls_seconds"], 6),
            inflight_step_seconds=round(inflight, 6),
            actions=data["actions"], first_step_seconds=data["first_step_seconds"],
            plan_remaining=len(active._actions) - active._next if active else None,
            cutoff=round(self._budget.cutoff - self._budget.started_at, 6),
            observed_status=observation.get("info", {}).get("status"),
            predicted_status=[dict(score=h.state.score, win=h.state.win, lose=h.state.lose)
                for h in self._hypotheses],
            stats=self._stats, tail=data["tail"])
        raise Task2EndDiagnostic("TASK2_END_0b61 " + json.dumps(report, separators=(",", ":"), default=str))

    def step(self, observation):
        if isinstance(observation, GridState):
            return super().step(observation)
        data = self._end_probe
        before = perf_counter()
        gap = before - (data["last_end"] if data["last_end"] is not None else self._budget.started_at)
        data["between_calls_seconds"] += gap
        # Leave time to serialize before the evaluator's five-second stop.
        if before - self._budget.started_at >= 4.75:
            self._end_report(observation, "near_hard_limit_before_step")
        action = super().step(observation)
        after = perf_counter()
        step_seconds = after - before
        data["agent_seconds"] += step_seconds
        data["last_end"] = after
        data["actions"] += 1
        if data["first_step_seconds"] is None:
            data["first_step_seconds"] = round(step_seconds, 6)
        reason = self._stats.get("reason")
        data["tail"].append(dict(turn=observation.get("info", {}).get("status", {}).get("turn"),
            action=str(action), reason=reason, step_seconds=round(step_seconds, 6),
            gap_seconds=round(gap, 6)))
        data["tail"] = data["tail"][-3:]
        if reason != "conditional_plan":
            self._end_report(observation, "first_fallback")
        if self._hypotheses and all(h.state.win for h in self._hypotheses):
            self._end_report(observation, "predicted_terminal_action")
        if after - self._budget.started_at >= 4.75:
            self._end_report(observation, "near_hard_limit_after_step")
        return action
