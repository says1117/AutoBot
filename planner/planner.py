import json
import os
from perception.scene import observe

from openai import OpenAI

from skills.registry import load_registry
from skills.runner import StubAttempter, run_skill

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
SYSTEM_PROMPT = (
    "You control a robot arm. Only act on objects the camera can see. "
    "If the requested object is not visible, do not call a skill. Explain why instead. "
    "After a skill finishes, report what happened."
)


def run_instruction(instruction: str, attempter, max_rounds: int = 5) -> str:
    client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.environ["LLM_API_KEY"], timeout=60)
    registry = load_registry()
    tools = [spec.to_tool_schema() for spec in registry.values()]
    objects = observe()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Camera sees (x, y, z in meters): {objects}\n\nInstruction: {instruction}"},
    ]

    # round of conversations with the model up to max_rounds times
    for round_num in range(max_rounds):
        # convo sent to VIM and wait for reply
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        message = response.choices[0].message

        # if theres nothing the model needs, returns message content
        if not message.tool_calls:
            return message.content

        messages.append(message)
        for call in message.tool_calls:
            spec = registry[call.function.name]
            # converts JSON text into a dict 
            args = json.loads(call.function.arguments)
            # actually runs the skill and returns a SkillResult plus the reason
            result = run_skill(spec, args, attempter)
            print(f"round {round_num}: {call.function.name}({args}) -> {result.outcome}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result.model_dump_json()})

    return "stopped: too many rounds"


if __name__ == "__main__":
    print(run_instruction("Pick up the red cube.", StubAttempter(seed=1, success_rate=0.0)))