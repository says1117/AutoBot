import json
import os

from openai import OpenAI

from skills.registry import load_registry

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["LLM_API_KEY"],
    # had issues testing where client stuck
    timeout=60,
)

registry = load_registry()
# one tool schema for each registered skill, in the format the LLM API expects
tools = [spec.to_tool_schema() for spec in registry.values()]

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    messages=[{"role": "user", "content": "Pick up the red cube."}],
    tools=tools,
)

# model gives its 1st choice reply in the form of a message
message = response.choices[0].message
print("text:", message.content)
# tool calls is what the model asked to run
for call in message.tool_calls or []:
    # prints each skill the model wants. from name: in skill.yaml ("pick")
    # args is what to run it with ("object_id")
    print("wants:", call.function.name, json.loads(call.function.arguments))