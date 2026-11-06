import pytest

from skills.registry import load_registry
from skills.runner import AttemptResult, StubAttempter, run_skill

ARGS = {"object_id": "green_cube"}


class ScriptedAttempter:
    """Plays back a fixed list of outcomes, so each retry path can be tested."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def attempt(self, spec, args):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        return AttemptResult(outcome=outcome, reason=None if outcome == "success" else outcome)


@pytest.fixture
def spec():
    return load_registry()["pick"]


def test_success_stops_right_away(spec):
    attempter = ScriptedAttempter("success")
    result = run_skill(spec, ARGS, attempter)
    assert (result.outcome, result.attempts, attempter.calls) == ("success", 1, 1)


@pytest.mark.parametrize("first", ["failure", "timeout"])
def test_retries_after_a_bad_attempt(spec, first):
    result = run_skill(spec, ARGS, ScriptedAttempter(first, "success"))
    assert (result.outcome, result.attempts) == ("success", 2)


def test_gives_up_at_max_attempts(spec):
    attempter = ScriptedAttempter(*["failure"] * spec.max_attempts)
    result = run_skill(spec, ARGS, attempter)
    assert result.outcome == "failure"
    assert result.attempts == attempter.calls == spec.max_attempts
    assert result.reason == "failure"


def test_reports_the_last_outcome_when_it_gives_up(spec):
    outcomes = ["failure"] * (spec.max_attempts - 1) + ["timeout"]
    assert run_skill(spec, ARGS, ScriptedAttempter(*outcomes)).outcome == "timeout"


def test_report_action_stops_retries(spec):
    spec = spec.model_copy(deep=True)
    spec.outcomes.failure.next_action = "report"
    attempter = ScriptedAttempter("failure", "success")
    result = run_skill(spec, ARGS, attempter)
    assert (result.outcome, result.attempts, attempter.calls) == ("failure", 1, 1)


def test_stub_gives_the_same_rolls_for_the_same_seed(spec):
    first = [StubAttempter(seed=42).attempt(spec, ARGS) for _ in range(5)]
    second = [StubAttempter(seed=42).attempt(spec, ARGS) for _ in range(5)]
    assert first == second


def test_stub_gives_a_reason_only_when_it_fails(spec):
    attempter = StubAttempter(seed=7)
    results = [attempter.attempt(spec, ARGS) for _ in range(200)]
    assert {result.outcome for result in results} == {"success", "failure", "timeout"}
    for result in results:
        assert (result.reason is None) == (result.outcome == "success")
