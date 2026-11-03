import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAMPLES = ROOT / "sample_invoices"

# Tests must never call the real OpenAI API.
os.environ["OPENAI_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["EXTRACTION_MODE"] = "rule-based"


def sample_bytes(name: str) -> bytes:
    path = SAMPLES / name
    if not path.exists():
        pytest.skip(f"sample invoice {name} not present")
    return path.read_bytes()


@pytest.fixture(scope="session")
def ground_truth() -> dict:
    return json.loads((SAMPLES / "ground_truth.json").read_text())["invoices"]


class FakeOpenAIClient:
    """Mimics ``client.chat.completions.parse`` returning queued parsed objects or raising errors."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse, create=self._create))

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        msg = SimpleNamespace(parsed=resp, refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def _create(self, **kwargs):
        """Plain JSON mode: queued responses are returned as JSON text."""
        self.calls.append(kwargs)
        resp = self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        msg = SimpleNamespace(content=resp.model_dump_json(), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


@pytest.fixture(autouse=True)
def _fresh_key_state():
    """Keys blocked by one test (quota, rejected) must not leak into the next."""
    from services.llm_extractor import reset_key_state

    reset_key_state()
    yield
    reset_key_state()
