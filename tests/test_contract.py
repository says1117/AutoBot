import copy

import pytest
import yaml
from pydantic import ValidationError

from perception.locate import COLOR_RANGES, CUBE_RGBA
from skills.contract import SkillSpec
from skills.registry import SKILLS_DIR, load_registry


@pytest.fixture
def pick_data():
    # the real file, so each test breaks exactly one thing about a valid skill
    return yaml.safe_load((SKILLS_DIR / "pick" / "skill.yaml").read_text())


def test_registry_finds_pick():
    registry = load_registry()
    assert "pick" in registry
    assert registry["pick"].max_attempts == 5


def test_tool_schema_has_the_shape_the_llm_needs():
    schema = load_registry()["pick"].to_tool_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "pick"
    parameters = schema["function"]["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["object_id"]
    assert "green_cube" in parameters["properties"]["object_id"]["enum"]


def test_every_color_is_defined_in_all_three_places(pick_data):
    allowed = set(pick_data["args"]["object_id"]["enum"])
    assert allowed == set(COLOR_RANGES) == set(CUBE_RGBA)


def test_optional_enum_is_left_out_of_the_schema(pick_data):
    del pick_data["args"]["object_id"]["enum"]
    schema = SkillSpec.model_validate(pick_data).to_tool_schema()
    assert "enum" not in schema["function"]["parameters"]["properties"]["object_id"]


def test_required_arg_must_exist(pick_data):
    pick_data["required"] = ["not_an_arg"]
    with pytest.raises(ValidationError, match="required names not found"):
        SkillSpec.model_validate(pick_data)


def test_unknown_next_action_is_rejected(pick_data):
    pick_data["outcomes"]["failure"]["next_action"] = "panic"
    with pytest.raises(ValidationError):
        SkillSpec.model_validate(pick_data)


@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_every_outcome_is_required(pick_data, outcome):
    data = copy.deepcopy(pick_data)
    del data["outcomes"][outcome]
    with pytest.raises(ValidationError):
        SkillSpec.model_validate(data)
