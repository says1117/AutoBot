import json
from types import SimpleNamespace

import pytest
from openai import APIError

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
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=reply)] if reply else [])


class CountingAttempter:
    def __init__(self, outcome="success"):
        self.outcome = outcome
        self.calls = 0

    def attempt(self, spec, args):
        self.calls += 1
        return AttemptResult(outcome=self.outcome)


@pytest.fixture
def run(monkeypatch):
    def run_with(replies, attempter, seen=SEEN, **kwargs):
        llm = FakeLLM(replies)
        monkeypatch.setattr(planner, "make_client", lambda: llm)
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


class NetworkError(APIError):
    def __init__(self):
        pass


@pytest.mark.parametrize("call, error", [
    (tool_reply(name="fly"), "Error: there is no skill named fly."),
    (tool_reply(arguments="{}"), "Error: missing arguments: object_id."),
    (tool_reply(arguments="{oops"), "Error: the arguments were not valid JSON."),
    (tool_reply(arguments="null"), "Error: the arguments must be a JSON object."),
    (tool_reply(arguments='{"object_id": ["green_cube"]}'), "Error: object not visible."),
])
def test_bad_tool_call_is_refused_and_explained_to_the_llm(run, call, error):
    attempter = CountingAttempter()
    answer, llm = run([call, text_reply("Sorry about that.")], attempter)
    assert answer == "Sorry about that."
    assert attempter.calls == 0
    assert llm.requests[1]["messages"][-1]["content"] == error


def test_llm_can_correct_itself_after_a_bad_call(run):
    attempter = CountingAttempter()
    answer, _ = run([tool_reply(arguments="{oops"), tool_reply(), text_reply("Done.")], attempter)
    assert answer == "Done."
    assert attempter.calls == 1


def test_failed_request_stops_with_a_clear_message(run):
    answer, _ = run([NetworkError()], CountingAttempter())
    assert answer == "stopped: the language model request failed (NetworkError)"


def test_request_failure_after_a_pick_still_returns(run):
    attempter = CountingAttempter()
    answer, _ = run([tool_reply(), NetworkError()], attempter)
    assert attempter.calls == 1
    assert answer.startswith("stopped:")


def test_empty_response_stops_with_a_clear_message(run):
    answer, _ = run([None], CountingAttempter())
    assert answer == "stopped: the language model sent an empty response"


def test_empty_reply_stops_with_a_clear_message(run):
    answer, _ = run([text_reply(None)], CountingAttempter())
    assert answer == "stopped: the language model sent an empty reply"
