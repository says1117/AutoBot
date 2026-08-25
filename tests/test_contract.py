"""The contract tests. These run on CPU, in CI, with no Isaac install."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from skills.contract import SkillSpec
from skills.registry import load_registry, tool_schemas
from skills.runner import StubSkillRunner


def test_registry_loads():
    registry = load_registry()
    assert "pick" in registry
    assert registry["pick"].timeout_s == 15.0


def test_tool_schema_is_openai_shaped():
    """The planner hands this straight to Grok, so the shape has to be exact."""
    schemas = tool_schemas(load_registry())
    pick = next(s for s in schemas if s["function"]["name"] == "pick")

    assert pick["type"] == "function"
    params = pick["function"]["parameters"]
    assert params["type"] == "object"
    assert params["required"] == ["object_id"]
    assert params["properties"]["object_id"]["enum"] == ["red_cube", "blue_cube", "green_cube"]


def test_required_must_reference_declared_args():
    """Catches the drift where an arg is renamed but 'required' is not."""
    with pytest.raises(ValidationError):
        SkillSpec.model_validate(
            {
                "name": "bad",
                "description": "x",
                "timeout_s": 1.0,
                "args": {"a": {"type": "string", "description": "d"}},
                "required": ["b"],
                "success": {"predicate": "p", "threshold": 0.0},
            }
        )


def test_stub_runner_is_reproducible():
    """Same seed must give the same outcomes, or the scorecard means nothing."""
    spec = load_registry()["pick"]
    a = [StubSkillRunner(seed=42).run(spec, {"object_id": "red_cube"}) for _ in range(1)]
    b = [StubSkillRunner(seed=42).run(spec, {"object_id": "red_cube"}) for _ in range(1)]
    assert [r.success for r in a] == [r.success for r in b]


def test_failure_reason_present_only_on_failure():
    spec = load_registry()["pick"]
    runner = StubSkillRunner(seed=7)
    for _ in range(50):
        result = runner.run(spec, {"object_id": "red_cube"})
        assert (result.failure_reason is None) == result.success
