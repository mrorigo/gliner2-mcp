from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List

from gliner2 import GLiNER2
from mcp.server.fastmcp import FastMCP

from gliner2_mcp.config import Settings


logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    # MCP stdio transport uses stdout for protocol frames, so logs must go to stderr.
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _load_model(settings: Settings) -> GLiNER2:
    if settings.device:
        try:
            return GLiNER2.from_pretrained(settings.model_id, device=settings.device)
        except TypeError:
            logger.warning(
                "GLiNER2.from_pretrained does not accept device=; falling back to default"
            )
    return GLiNER2.from_pretrained(settings.model_id)


ClassificationValue = str | List[str]
ClassificationSchema = Dict[str, List[str] | Dict[str, Any]]


class Gliner2Service:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = _load_model(settings)

    def _validate_text(self, text: str) -> None:
        max_len = self.settings.max_text_length
        if max_len > 0 and len(text) > max_len:
            raise ValueError(
                f"text length {len(text)} exceeds limit {max_len}. "
                "Use GLINER2_MAX_TEXT_LENGTH=0 to disable the limit."
            )

    async def extract_entities(self, text: str, labels: List[str]) -> Dict[str, List[str]]:
        self._validate_text(text)
        result = self.model.extract_entities(text, labels)
        if isinstance(result, dict):
            entities = result.get("entities")
            if isinstance(entities, dict):
                requested = set(labels)
                return {
                    str(label): [str(item) for item in values if isinstance(item, str)]
                    for label, values in entities.items()
                    if isinstance(values, list) and str(label) in requested
                }
        raise ValueError("Unexpected response shape from GLiNER2.extract_entities")

    async def classify_text(
        self, text: str, schema: ClassificationSchema
    ) -> Dict[str, ClassificationValue]:
        self._validate_text(text)
        outputs = self.model.classify_text(text, schema)
        if not isinstance(outputs, dict):
            raise ValueError("Unexpected response shape from GLiNER2.classify_text")

        out: Dict[str, ClassificationValue] = {}
        for field, result in outputs.items():
            if isinstance(result, list):
                out[field] = [str(item) for item in result]
            else:
                out[field] = str(result)
        return out

    async def extract_json(self, text: str, schema: Dict[str, Any]) -> Any:
        self._validate_text(text)
        return self.model.extract_json(text, schema)


def create_server(settings: Settings) -> FastMCP:
    _configure_logging(settings.log_level)
    service = Gliner2Service(settings)

    mcp = FastMCP("gliner2")

    @mcp.tool()
    async def extractEntities(text: str, labels: List[str]) -> Dict[str, List[str]]:
        """
        Extract named entities from free text using an explicit label list.

        Args:
            text: Raw text to analyze. Keep the full sentence/paragraph context for best results.
            labels: Entity types to extract, for example
                ["person", "company", "product", "location"].
                Labels are schema-driven prompts, not fixed model classes.

        Returns:
            Dict[str, List[str]] where each requested label maps to extracted surface forms.
            Only requested labels are returned.

        Example:
            text="Apple CEO Tim Cook unveiled iPhone 15 in Cupertino."
            labels=["company", "person", "product", "location"]
            -> {
                "company": ["Apple"],
                "person": ["Tim Cook"],
                "product": ["iPhone 15"],
                "location": ["Cupertino"]
            }

        Notes:
            - If GLINER2_MAX_TEXT_LENGTH is configured and exceeded, the tool raises ValueError.
            - Output values are plain strings (no offsets/confidence in this tool contract).
        """

        return await service.extract_entities(text, labels)

    @mcp.tool()
    async def classifyText(
        text: str, schema: ClassificationSchema
    ) -> Dict[str, ClassificationValue]:
        """
        Run schema-driven text classification tasks (single-label or multi-label).

        Args:
            text: Input text to classify.
            schema: Task definition mapping task name -> config.
                Supported forms:
                - Single-label task:
                    {"sentiment": ["positive", "negative", "neutral"]}
                - Multi-label / advanced task:
                    {
                        "aspects": {
                            "labels": ["camera", "battery", "display"],
                            "multi_label": True,
                            "cls_threshold": 0.4
                        }
                    }

        Returns:
            Dict[str, str | List[str]]:
            - Single-label tasks return a string label.
            - Multi-label tasks return a list of labels.

        Example:
            schema={
                "sentiment": ["positive", "negative", "neutral"],
                "aspects": {"labels": ["camera", "battery"], "multi_label": True}
            }
            -> {"sentiment": "positive", "aspects": ["camera"]}

        Notes:
            - Use distinct task names (dict keys) because they become output keys.
            - If GLINER2_MAX_TEXT_LENGTH is configured and exceeded, the tool raises ValueError.
        """

        return await service.classify_text(text, schema)

    @mcp.tool()
    async def extractJson(text: str, schema: Dict[str, Any]) -> Any:
        """
        Extract structured JSON from text using GLiNER2 structure schema syntax.

        Args:
            text: Source text containing structured facts.
            schema: Structure specification dictionary. Common pattern:
                {
                    "product": [
                        "name::str::Full product name",
                        "price::str::Price with currency",
                        "features::list::List of key features"
                    ]
                }

        Returns:
            JSON-serializable object, typically Dict[str, List[Dict[str, Any]]],
            shaped according to the schema.

        Example:
            text="iPhone 15 Pro Max 256GB priced at $1199."
            schema={"product": ["name::str", "storage::str", "price::str"]}
            -> {"product": [{"name": "iPhone 15 Pro Max", "storage": "256GB", "price": "$1199"}]}

        Notes:
            - Field specs use "field_name::dtype::description" format.
            - If GLINER2_MAX_TEXT_LENGTH is configured and exceeded, the tool raises ValueError.
        """

        return await service.extract_json(text, schema)

    return mcp
