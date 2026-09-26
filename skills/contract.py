from typing import Literal

from pydantic import BaseModel, model_validator


class Arg(BaseModel):
    type: Literal["string", "number", "integer", "boolean"]
    description: str
    enum: list[str] | None = None


class Outcome(BaseModel):
    predicates: list[str] | None = None
    # reports are Literal inputs that I wrote. Only exact values allowed
    next_action: Literal["report", "retry_new_angle"]
    # = None makes field optional, and | None makes type honest
    # this is a float or nothing, so None counts as a valid value
    hold_s: float | None = None
    threshold: float | None = None
    description: str | None = None


class Outcomes(BaseModel):
    success: Outcome
    failure: Outcome
    timeout: Outcome


class SkillSpec(BaseModel):
    name: str
    description: str
    timeout_s: float
    max_attempts: int
    reach_dist_in_meters: float
    # object_id is the name (string) and Arg is what lives under it
    args: dict[str, Arg]
    required: list[str]
    preconditions: list[str]
    outcomes: Outcomes

    def to_tool_schema(self) -> dict:
        # creates dictionary of properties based on the arg name
        properties = {}
        # loops through names, all args, adds to entry
        for arg_name, arg in self.args.items():
            entry = {"type": arg.type, "description": arg.description}
            # enums optional, so if its there, add it
            if arg.enum is not None:
                entry["enum"] = arg.enum
            properties[arg_name] = entry

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": self.required
                },
            },
        }

    # tells Pydantic to run this function as a check everytime a SkillSpec is created
    # @ is the decorator that registers it
    @model_validator(mode="after")
    def check_required_args_exist(self):
        # finds that required names are in args. if there arent, the subtraction of these sets will result in false
        # and error raised
        missing = set(self.required) - set(self.args)
        if missing:
            raise ValueError(f"required names not found in args: {missing}")
        return self
