import pytest

from perception.verify import VLMVerifier
from planner import runtime
from planner.runtime import SetupError, log, make_client, setup_logging


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SetupError, match="LLM_API_KEY"):
        make_client()


def test_empty_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(SetupError, match="LLM_API_KEY"):
        make_client()


def test_verifier_fails_as_soon_as_it_is_built_without_a_key(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(SetupError):
        VLMVerifier()


def test_client_uses_the_key_and_the_nim_address(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    client = make_client(timeout=5)
    assert client.api_key == "not-a-real-key"
    assert "integrate.api.nvidia.com" in str(client.base_url)


@pytest.fixture
def clean_logger():
    handlers, level, propagate = list(log.handlers), log.level, log.propagate
    yield
    for handler in log.handlers:
        handler.close()
    log.handlers[:] = handlers
    log.setLevel(level)
    log.propagate = propagate


def test_messages_reach_the_log_file(tmp_path, clean_logger):
    path = setup_logging(tmp_path)
    log.info("the arm moved")
    log.debug("too much detail")
    text = path.read_text()
    assert "the arm moved" in text
    assert "too much detail" not in text


def test_setting_up_twice_does_not_double_every_line(tmp_path, clean_logger):
    setup_logging(tmp_path)
    path = setup_logging(tmp_path)
    log.info("only once")
    assert path.read_text().count("only once") == 1


def test_default_log_folder_is_ignored_by_git():
    gitignore = (runtime.LOGS.parent / ".gitignore").read_text().splitlines()
    assert "logs/" in gitignore
