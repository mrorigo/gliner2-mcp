from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Settings:
    model_id: str
    device: Optional[str]
    max_text_length: int
    log_level: str


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _load_env_file(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        if key:
            os.environ.setdefault(key, value)


def load_settings() -> Settings:
    env_file = os.environ.get("GLINER2_MCP_ENV_FILE")
    if env_file:
        _load_env_file(Path(env_file))
    else:
        default_env = Path.cwd() / ".env"
        _load_env_file(default_env)

    model_id = os.environ.get("GLINER2_MODEL_ID", "fastino/gliner2-base-v1")
    device = os.environ.get("GLINER2_DEVICE")
    max_text_length = _parse_int(os.environ.get("GLINER2_MAX_TEXT_LENGTH"), 0)
    log_level = os.environ.get("GLINER2_MCP_LOG_LEVEL", "INFO")

    return Settings(
        model_id=model_id,
        device=device,
        max_text_length=max_text_length,
        log_level=log_level,
    )
