import base64
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from openai import APIError

from perception.verify import VLMVerifier

PHOTO = np.zeros((48, 64, 3), np.uint8)


class FakeVLM:
    """Plays back replies, where an exception in the list is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])


class NetworkError(APIError):
    def __init__(self):
        pass


def network_error():
    return NetworkError()


@pytest.mark.parametrize("reply, expected", [
    ("YES", True),
    ("NO", False),
    ("yes", True),
    ("The cube is on the floor.\n\nNO", False),
    ("At first NO, but looking closer it is held. YES", True),
])
def test_reads_the_final_answer(reply, expected):
    assert VLMVerifier(FakeVLM(reply)).confirm(PHOTO, "green_cube") is expected


def test_words_that_contain_yes_or_no_do_not_count():
    fake = FakeVLM("NOTHING to report", "EYES and NOSE")
    assert VLMVerifier(fake).confirm(PHOTO, "green_cube") is None


def test_asks_again_after_an_empty_reply():
    fake = FakeVLM("", "YES")
    assert VLMVerifier(fake).confirm(PHOTO, "green_cube") is True
    assert len(fake.requests) == 2


def test_asks_again_after_a_network_error():
    fake = FakeVLM(network_error(), "NO")
    assert VLMVerifier(fake).confirm(PHOTO, "green_cube") is False


def test_gives_none_when_it_never_gets_an_answer():
    fake = FakeVLM(network_error(), None)
    assert VLMVerifier(fake).confirm(PHOTO, "green_cube") is None


def test_request_names_the_color_and_carries_the_photo():
    fake = FakeVLM("YES")
    photo = np.full((48, 64, 3), (255, 0, 0), np.uint8)
    VLMVerifier(fake).confirm(photo, "purple_cube")

    text, image = fake.requests[0]["messages"][0]["content"]
    assert "purple cube" in text["text"]

    encoded = image["image_url"]["url"].removeprefix("data:image/jpeg;base64,")
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == photo.shape
    # OpenCV reads blue, green, red, so a red photo must arrive with red in the last channel
    assert decoded[24, 32, 2] > 200 and decoded[24, 32, 0] < 50
