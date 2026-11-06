import logging
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

LOGS = Path(__file__).parent.parent / "logs"
log = logging.getLogger("autobot")


class SetupError(RuntimeError):
    """Something the person running AutoBot needs to fix before it can start."""


def make_client(timeout: float = 60, max_retries: int = 2) -> OpenAI:
    key = os.environ.get("LLM_API_KEY")
    if not key:
        raise SetupError("LLM_API_KEY is not set. Run: export LLM_API_KEY=your_key_here")
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=key,
                  timeout=timeout, max_retries=max_retries)


def setup_logging(folder: Path = LOGS) -> Path:
    """Sends AutoBot's own messages to the screen and to a new file for this run."""
    folder.mkdir(exist_ok=True)
    path = folder / f"autobot_{datetime.now():%Y%m%d_%H%M%S}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    log.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(path)):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    # keeps these lines separate from the libraries' own logging
    log.propagate = False
    return path
