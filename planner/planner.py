import json

from openai import APIError

from perception.scene import observe
from planner.runtime import log, make_client
from skills.registry import load_registry
from skills.runner import StubAttempter, run_skill

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
SYSTEM_PROMPT = (
    "You control a robot arm. Only act on objects the camera can see. "
    "If the requested object is not visible, do not call a skill. Explain why instead. "
    "After a skill finishes, report what happened."
)


def check_call(call, registry: dict, objects: dict):
    """Returns (spec, args, None) for a call that can run, or (None, None, error) if it can't."""
    spec = registry.get(call.function.name)
    if spec is None:
        return None, None, f"Error: there is no skill named {call.function.name}."

    # converts JSON text into a dict
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return None, None, "Error: the arguments were not valid JSON."
    if not isinstance(args, dict):
        return None, None, "Error: the arguments must be a JSON object."

    missing = [name for name in spec.required if name not in args]
    if missing:
        return None, None, f"Error: missing arguments: {', '.join(missing)}."

    object_id = args.get("object_id")
    if not isinstance(object_id, str) or object_id not in objects:
        return None, None, "Error: object not visible."
    return spec, args, None


def run_instruction(instruction: str, attempter, max_rounds: int = 5, observe_fn=observe) -> str:
    client = make_client()
    registry = load_registry()
    tools = [spec.to_tool_schema() for spec in registry.values()]
    objects = observe_fn()
    log.info("instruction: %s | camera sees: %s", instruction, objects)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Camera sees (x, y, z in meters): {objects}\n\nInstruction: {instruction}"},
    ]

    # round of conversations with the model up to max_rounds times
    for round_num in range(max_rounds):
        # convo sent to NIM and wait for reply
        try:
            response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        except APIError as error:
            log.error("language model request failed: %s", type(error).__name__)
            return f"stopped: the language model request failed ({type(error).__name__})"
        if not response.choices:
            log.error("language model sent an empty response")
            return "stopped: the language model sent an empty response"
        message = response.choices[0].message

        # if theres nothing the model needs, returns message content
        if not message.tool_calls:
            return message.content or "stopped: the language model sent an empty reply"

        messages.append(message)
        for call in message.tool_calls:
            spec, args, error = check_call(call, registry, objects)
            # the model gets the error back, so it can correct itself or explain
            if error:
                log.warning("round %d: refused %s: %s", round_num, call.function.name, error)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": error})
                continue
            # actually runs the skill and returns a SkillResult plus the reason
            result = run_skill(spec, args, attempter)
            log.info("round %d: %s(%s) -> %s", round_num, call.function.name, args, result.outcome)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.model_dump_json()})

    log.warning("stopped after %d rounds", max_rounds)
    return "stopped: too many rounds"


if __name__ == "__main__":
    print(run_instruction("Pick up the red cube.", StubAttempter(seed=1)))
