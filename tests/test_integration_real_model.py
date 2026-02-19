from __future__ import annotations

import asyncio
import os

import pytest

from gliner2_mcp.config import Settings
from gliner2_mcp.server import Gliner2Service


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("GLINER2_MCP_RUN_INTEGRATION") != "1",
    reason="Set GLINER2_MCP_RUN_INTEGRATION=1 to run real-model integration tests.",
)
def test_real_model_inference_end_to_end() -> None:
    model_id = os.getenv("GLINER2_MODEL_ID", "fastino/gliner2-base-v1")
    service = Gliner2Service(
        Settings(
            model_id=model_id,
            device=os.getenv("GLINER2_DEVICE"),
            max_text_length=0,
            log_level="INFO",
        )
    )

    entities = asyncio.run(
        service.extract_entities(
            "Apple CEO Tim Cook unveiled the iPhone 15 Pro in Cupertino for $999.",
            ["company", "person", "product", "location"],
        )
    )
    assert "Apple" in entities["company"]
    assert "Tim Cook" in entities["person"]
    assert any("iPhone 15 Pro" in item for item in entities["product"])
    assert "Cupertino" in entities["location"]

    classification = asyncio.run(
        service.classify_text(
            "Great camera quality, but poor battery life.",
            {
                "sentiment": ["positive", "negative", "neutral"],
                "aspects": {
                    "labels": ["camera", "battery", "display"],
                    "multi_label": True,
                    "cls_threshold": 0.4,
                },
            },
        )
    )
    assert classification["sentiment"] in {"positive", "negative", "neutral"}
    assert isinstance(classification["aspects"], list)
    assert "camera" in classification["aspects"]
    assert "battery" in classification["aspects"]

    structured = asyncio.run(
        service.extract_json(
            "iPhone 15 Pro Max with 256GB storage priced at $1199.",
            {
                "product": [
                    "name::str::Full product name",
                    "storage::str::Storage capacity",
                    "price::str::Price with currency",
                ]
            },
        )
    )

    assert isinstance(structured, dict)
    assert "product" in structured
    assert isinstance(structured["product"], list)
    assert structured["product"]
    first = structured["product"][0]
    assert first.get("name")
    assert first.get("storage") == "256GB"
    assert "$" in first.get("price", "")
