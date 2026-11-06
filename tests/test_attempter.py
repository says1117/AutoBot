from pathlib import Path
from types import SimpleNamespace

import pytest

# the attempter loads a Brax policy, which only the full install has
pytest.importorskip("brax")

from planner.runtime import SetupError
from skills import policy_attempter


def test_missing_policy_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(policy_attempter, "POLICY_PATH", Path("/not/a/real/policy"))
    with pytest.raises(SetupError, match="policy file not found"):
        policy_attempter.PolicyAttempter(SimpleNamespace())


def test_tracked_policy_file_exists():
    assert policy_attempter.POLICY_PATH.exists()
