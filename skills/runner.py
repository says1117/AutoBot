import random
from typing import Literal

from pydantic import BaseModel

from planner.runtime import log
from skills.contract import SkillSpec

OutcomeName = Literal["success", "failure", "timeout"]

# class storing results for each attempt
class AttemptResult(BaseModel):
    outcome: OutcomeName
    reason: str | None = None

class SkillResult(BaseModel):
    skill: str
    outcome: OutcomeName
    attempts: int
    reason: str | None = None

class StubAttempter:
    def __init__(self, seed: int = 0, success_rate: float = 0.6, timeout_rate: float = 0.1):
        self.rng = random.Random(seed)
        self.success_rate = success_rate
        self.timeout_rate = timeout_rate

    def attempt(self, spec: SkillSpec, args: dict) -> AttemptResult:
        roll = self.rng.random()
        if roll < self.success_rate:
            return AttemptResult(outcome="success")
        if roll < self.success_rate + self.timeout_rate:
            return AttemptResult(outcome="timeout", reason="did not finish in time")
        return AttemptResult(outcome="failure", reason="object dropped")

# retry policy, decides how many times to run and when to stop
def run_skill(spec: SkillSpec, args: dict, attempter: StubAttempter) -> SkillResult:
    for attempt_num in range(1, spec.max_attempts + 1):
        result = attempter.attempt(spec, args)
        log.info("%s attempt %d of %d: %s (%s)", spec.name, attempt_num, spec.max_attempts,
                 result.outcome, result.reason)
        action = getattr(spec.outcomes, result.outcome).next_action
        if action == "report":
            return SkillResult(skill=spec.name, outcome=result.outcome, attempts=attempt_num, reason=result.reason)
    return SkillResult(skill=spec.name, outcome=result.outcome, attempts=spec.max_attempts, reason=result.reason)