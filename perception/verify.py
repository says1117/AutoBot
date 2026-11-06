import base64
import os
import re

import cv2
from openai import APIError, OpenAI

MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
PROMPT = (
    "This photo is a side view of a robot arm above a tiled floor. Is the {color} cube being held "
    "by the gripper, lifted clearly above the floor? Finish your reply with exactly one word: YES or NO."
)


class VLMVerifier:
    """Asks a vision-language model whether a photo shows the pick worked."""

    def __init__(self, client=None, tries: int = 2):
        self.client = client or OpenAI(base_url="https://integrate.api.nvidia.com/v1",
                                       api_key=os.environ["LLM_API_KEY"], timeout=60, max_retries=0)
        self.tries = tries

    def confirm(self, rgb, object_id: str) -> bool | None:
        """Returns True or False, or None if the model never gave a usable answer."""
        # JPEG keeps the request small, since NIM limits the size of inline images
        _, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode()
        color = object_id.removesuffix("_cube")
        content = [{"type": "text", "text": PROMPT.format(color=color)},
                   {"type": "image_url", "image_url": {"url": image}}]

        for _ in range(self.tries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL, temperature=0, max_tokens=2048,
                    messages=[{"role": "user", "content": content}])
            except APIError:
                continue
            # the model may reason first, so its last YES or NO is the answer
            words = re.findall(r"\b(YES|NO)\b", (response.choices[0].message.content or "").upper())
            if words:
                return words[-1] == "YES"
        return None
