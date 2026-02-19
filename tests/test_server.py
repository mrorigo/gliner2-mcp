from __future__ import annotations

import asyncio
import logging
import sys

import pytest

from gliner2_mcp.config import Settings
from gliner2_mcp.server import Gliner2Service
from gliner2_mcp.server import _configure_logging


class FakeModel:
    def __init__(self) -> None:
        self.last_json_args = None

    def extract_entities(self, text, labels):
        assert text == "hello from Apple"
        assert labels == ["company", "person"]
        return {"entities": {"company": ["Apple"], "person": [], "ignored": ["X"]}}

    def classify_text(self, text, schema):
        assert text == "great camera"
        assert "sentiment" in schema
        return {"sentiment": "positive", "aspects": ["camera", "display"]}

    def extract_json(self, text, schema):
        self.last_json_args = (text, schema)
        return {"product": [{"name": "iPhone"}]}


@pytest.fixture
def service(monkeypatch) -> Gliner2Service:
    fake_model = FakeModel()
    monkeypatch.setattr("gliner2_mcp.server._load_model", lambda _settings: fake_model)
    return Gliner2Service(
        Settings(
            model_id="fastino/gliner2-base-v1",
            device=None,
            max_text_length=0,
            log_level="INFO",
        )
    )


def test_extract_entities_normalizes_output(service: Gliner2Service) -> None:
    result = asyncio.run(service.extract_entities("hello from Apple", ["company", "person"]))
    assert result == {"company": ["Apple"], "person": []}


def test_classify_text_supports_single_and_multi_label(service: Gliner2Service) -> None:
    result = asyncio.run(
        service.classify_text(
        "great camera",
        {
            "sentiment": ["positive", "negative", "neutral"],
            "aspects": {
                "labels": ["camera", "performance", "battery", "display"],
                "multi_label": True,
                "cls_threshold": 0.4,
            },
        },
        )
    )

    assert result == {"sentiment": "positive", "aspects": ["camera", "display"]}


def test_extract_json_passes_text_and_schema(service: Gliner2Service) -> None:
    schema = {"product": ["name::str::Product name"]}
    result = asyncio.run(service.extract_json("iPhone listed", schema))
    assert result == {"product": [{"name": "iPhone"}]}


def test_logging_is_configured_to_stderr() -> None:
    _configure_logging("INFO")
    root = logging.getLogger()
    assert root.handlers, "Expected at least one root logging handler"
    assert getattr(root.handlers[0], "stream", None) is sys.stderr
