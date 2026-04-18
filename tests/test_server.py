from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys

import pytest

from gliner2_mcp.config import Settings
from gliner2_mcp.server import Gliner2Service
from gliner2_mcp.server import _configure_logging


class FakeModel:
    def __init__(self) -> None:
        self.last_json_args = None
        self.last_entities_args = None
        self.last_classify_args = None

    def extract_entities(self, text, labels):
        self.last_entities_args = (text, labels)
        return {"entities": {"company": ["Apple"], "person": [], "ignored": ["X"]}}

    def classify_text(self, text, schema):
        self.last_classify_args = (text, schema)
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
    result = asyncio.run(
        service.extract_entities(
            labels=["company", "person"],
            text="hello from Apple",
        )
    )
    assert result == {"company": ["Apple"], "person": []}
    assert service.model.last_entities_args == ("hello from Apple", ["company", "person"])


def test_classify_text_supports_single_and_multi_label(service: Gliner2Service) -> None:
    result = asyncio.run(
        service.classify_text(
            {
                "sentiment": ["positive", "negative", "neutral"],
                "aspects": {
                    "labels": ["camera", "performance", "battery", "display"],
                    "multi_label": True,
                    "cls_threshold": 0.4,
                },
            },
            text="great camera",
        )
    )

    assert result == {"sentiment": "positive", "aspects": ["camera", "display"]}
    assert service.model.last_classify_args is not None
    assert service.model.last_classify_args[0] == "great camera"


def test_extract_json_passes_text_and_schema(service: Gliner2Service) -> None:
    schema = {"product": ["name::str::Product name"]}
    result = asyncio.run(service.extract_json(schema, text="iPhone listed"))
    assert result == {"product": [{"name": "iPhone"}]}
    assert service.model.last_json_args == ("iPhone listed", schema)


def test_extract_entities_reads_text_from_filename(
    service: Gliner2Service, tmp_path: Path, monkeypatch
) -> None:
    input_file = tmp_path / "sample.txt"
    input_file.write_text("hello from Apple", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        service.extract_entities(labels=["company", "person"], filename="sample.txt")
    )

    assert result == {"company": ["Apple"], "person": []}
    assert service.model.last_entities_args == ("hello from Apple", ["company", "person"])


def test_classify_text_reads_text_from_filename(
    service: Gliner2Service, tmp_path: Path, monkeypatch
) -> None:
    input_file = tmp_path / "review.txt"
    input_file.write_text("great camera", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = asyncio.run(
        service.classify_text(
            {
                "sentiment": ["positive", "negative", "neutral"],
                "aspects": {
                    "labels": ["camera", "performance", "battery", "display"],
                    "multi_label": True,
                    "cls_threshold": 0.4,
                },
            },
            filename="review.txt",
        )
    )

    assert result == {"sentiment": "positive", "aspects": ["camera", "display"]}
    assert service.model.last_classify_args is not None
    assert service.model.last_classify_args[0] == "great camera"


def test_extract_json_reads_text_from_filename(
    service: Gliner2Service, tmp_path: Path, monkeypatch
) -> None:
    input_file = tmp_path / "product.txt"
    input_file.write_text("iPhone listed", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    schema = {"product": ["name::str::Product name"]}
    result = asyncio.run(service.extract_json(schema, filename="product.txt"))

    assert result == {"product": [{"name": "iPhone"}]}
    assert service.model.last_json_args == ("iPhone listed", schema)


def test_rejects_both_text_and_filename(service: Gliner2Service, tmp_path: Path, monkeypatch) -> None:
    input_file = tmp_path / "sample.txt"
    input_file.write_text("hello from Apple", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="exactly one of 'text' or 'filename'"):
        asyncio.run(
            service.extract_entities(
                labels=["company", "person"],
                text="hello from Apple",
                filename="sample.txt",
            )
        )


def test_rejects_missing_text_and_filename(service: Gliner2Service) -> None:
    with pytest.raises(ValueError, match="exactly one of 'text' or 'filename'"):
        asyncio.run(service.extract_entities(labels=["company", "person"]))


def test_rejects_absolute_filename_path(service: Gliner2Service, tmp_path: Path) -> None:
    absolute = tmp_path / "sample.txt"
    absolute.write_text("hello from Apple", encoding="utf-8")
    with pytest.raises(ValueError, match="relative path"):
        asyncio.run(
            service.extract_entities(
                labels=["company", "person"],
                filename=str(absolute),
            )
        )


def test_rejects_filename_outside_cwd(
    service: Gliner2Service, tmp_path: Path, monkeypatch
) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "sample.txt"
    outside_file.write_text("hello from Apple", encoding="utf-8")

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    with pytest.raises(ValueError, match="resolve inside the current working directory"):
        asyncio.run(
            service.extract_entities(
                labels=["company", "person"],
                filename="../outside/sample.txt",
            )
        )


def test_rejects_non_utf8_filename_content(
    service: Gliner2Service, tmp_path: Path, monkeypatch
) -> None:
    input_file = tmp_path / "binary.bin"
    input_file.write_bytes(b"\xff\xfe\x00")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="UTF-8 text"):
        asyncio.run(
            service.extract_entities(
                labels=["company", "person"],
                filename="binary.bin",
            )
        )


def test_max_text_length_applies_to_filename_input(monkeypatch, tmp_path: Path) -> None:
    fake_model = FakeModel()
    monkeypatch.setattr("gliner2_mcp.server._load_model", lambda _settings: fake_model)
    limited_service = Gliner2Service(
        Settings(
            model_id="fastino/gliner2-base-v1",
            device=None,
            max_text_length=5,
            log_level="INFO",
        )
    )
    input_file = tmp_path / "long.txt"
    input_file.write_text("too long", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="exceeds limit 5"):
        asyncio.run(
            limited_service.extract_entities(
                labels=["company", "person"],
                filename="long.txt",
            )
        )


def test_logging_is_configured_to_stderr() -> None:
    _configure_logging("INFO")
    root = logging.getLogger()
    assert root.handlers, "Expected at least one root logging handler"
    assert getattr(root.handlers[0], "stream", None) is sys.stderr
