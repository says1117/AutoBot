import json
from types import SimpleNamespace

import pytest

from planner import planner
from skills.runner import AttemptResult

SEEN = {"green_cube": [0.5, 0.0, 0.03]}


def text_reply(content):
    return SimpleNamespace(content=content, tool_calls=None)


def tool_reply(name="pick", arguments='{"object_id": "green_cube"}'):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="call_1", function=function)])


class FakeLLM:
    """Stands in for the OpenAI client: no network, no key, and replies set by the test."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        # copy the list, because the planner keeps adding to the one it passed in
        self.requests.append({**request, "messages": list(request["messages"])})
        return SimpleNamespace(choices=[SimpleNamespace(message=self.replies.pop(0))])


class CountingAttempter:
    def __init__(self, outcome="success"):
        self.outcome = outcome
        self.calls = 0

    def attempt(self, spec, args):
        self.calls += 1
        return AttemptResult(outcome=self.outcome)


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")

    def run_with(replies, attempter, seen=SEEN, **kwargs):
        llm = FakeLLM(replies)
        monkeypatch.setattr(planner, "OpenAI", lambda **_: llm)
        answer = planner.run_instruction("Pick up the cube.", attempter, observe_fn=lambda: seen, **kwargs)
        return answer, llm

    return run_with


def test_plain_reply_runs_nothing(run):
    attempter = CountingAttempter()
    answer, _ = run([text_reply("Nothing to do.")], attempter)
    assert answer == "Nothing to do."
    assert attempter.calls == 0


def test_llm_is_given_the_camera_view_and_the_skills(run):
    _, llm = run([text_reply("ok")], CountingAttempter())
    request = llm.requests[0]
    assert str(SEEN) in request["messages"][1]["content"]
    assert "Pick up the cube." in request["messages"][1]["content"]
    assert [tool["function"]["name"] for tool in request["tools"]] == ["pick"]


def test_visible_object_runs_the_skill_and_reports_back(run):
    attempter = CountingAttempter()
    answer, llm = run([tool_reply(), text_reply("Done.")], attempter)
    assert answer == "Done."
    assert attempter.calls == 1
    tool_message = llm.requests[1]["messages"][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"])["outcome"] == "success"


def test_object_the_camera_cannot_see_never_reaches_the_arm(run):
    attempter = CountingAttempter()
    call = tool_reply(arguments='{"object_id": "red_cube"}')
    answer, llm = run([call, text_reply("I cannot see it.")], attempter)
    assert answer == "I cannot see it."
    assert attempter.calls == 0
    assert llm.requests[1]["messages"][-1]["content"] == "Error: object not visible."


def test_empty_camera_view_blocks_every_object(run):
    attempter = CountingAttempter()
    run([tool_reply(), text_reply("Nothing visible.")], attempter, seen={})
    assert attempter.calls == 0


def test_stops_when_the_llm_never_finishes(run):
    attempter = CountingAttempter()
    answer, llm = run([tool_reply(), tool_reply()], attempter, max_rounds=2)
    assert answer == "stopped: too many rounds"
    assert len(llm.requests) == 2


# These are real crashes in planner.py. Step 3 fixes them, and strict=True then
# fails these tests until the xfail marks are removed, so they can't be forgotten.
@pytest.mark.xfail(strict=True, raises=KeyError, reason="made-up skill name crashes the planner")
def test_made_up_skill_is_refused(run):
    attempter = CountingAttempter()
    run([tool_reply(name="fly"), text_reply("I cannot do that.")], attempter)
    assert attempter.calls == 0


@pytest.mark.xfail(strict=True, raises=KeyError, reason="a missing object_id crashes the planner")
def test_call_without_object_id_is_refused(run):
    attempter = CountingAttempter()
    run([tool_reply(arguments="{}"), text_reply("Which object?")], attempter)
    assert attempter.calls == 0


@pytest.mark.xfail(strict=True, raises=json.JSONDecodeError, reason="broken JSON crashes the planner")
def test_broken_json_is_refused(run):
    attempter = CountingAttempter()
    run([tool_reply(arguments="{oops"), text_reply("Sorry.")], attempter)
    assert attempter.calls == 0
