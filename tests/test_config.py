from __future__ import annotations

from pathlib import Path

from gliner2_mcp.config import load_settings


def test_load_settings_from_env_file(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / "test.env"
    env_path.write_text(
        "\n".join(
            [
                "GLINER2_MODEL_ID=fastino/gliner2-custom",
                "GLINER2_DEVICE=cpu",
                "GLINER2_MAX_TEXT_LENGTH=4096",
                "GLINER2_MCP_LOG_LEVEL=debug",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("GLINER2_MCP_ENV_FILE", str(env_path))
    monkeypatch.delenv("GLINER2_MODEL_ID", raising=False)
    monkeypatch.delenv("GLINER2_DEVICE", raising=False)
    monkeypatch.delenv("GLINER2_MAX_TEXT_LENGTH", raising=False)
    monkeypatch.delenv("GLINER2_MCP_LOG_LEVEL", raising=False)

    settings = load_settings()

    assert settings.model_id == "fastino/gliner2-custom"
    assert settings.device == "cpu"
    assert settings.max_text_length == 4096
    assert settings.log_level == "debug"
