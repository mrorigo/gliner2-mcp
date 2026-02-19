# gliner2-mcp

MCP server that exposes GLiNER2 entity extraction, classification, and JSON extraction as tools.

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Run the MCP server:

```bash
uv run gliner2-mcp
```

## Environment configuration

This server is configured via environment variables (optionally loaded from `.env`).

| Variable | Default | Description |
| --- | --- | --- |
| `GLINER2_MODEL_ID` | `fastino/gliner2-base-v1` | GLiNER2 model id or path. |
| `GLINER2_DEVICE` | empty | Device override passed to `from_pretrained` when supported. |
| `GLINER2_MAX_TEXT_LENGTH` | `0` | Maximum input length. `0` disables the limit. |
| `GLINER2_MCP_LOG_LEVEL` | `INFO` | Log level for server logging. |
| `GLINER2_MCP_ENV_FILE` | empty | Path to an env file to load before reading the variables above. |

If `GLINER2_MCP_ENV_FILE` is not set, the server will load `.env` from the working directory when it exists.

## MCP tools

- `extractEntities(text, labels)` returns a mapping of label to entity spans.
- `classifyText(text, schema)` returns a mapping of field name to chosen label or list of labels.
- `extractJson(text, schema)` returns a JSON object that matches the provided schema.

## Available Models

[GLiNER2 family models on Hugging Face](https://huggingface.co/collections/fastino/gliner2-family)

## Development

```bash
uv sync --dev
uv run ruff check .
```

## Testing

Run the default test suite (unit + contract tests):

```bash
uv sync --extra dev
uv run --extra dev pytest -q
```

Run real-model integration tests (downloads/loads actual GLiNER2 weights):

```bash
GLINER2_MCP_RUN_INTEGRATION=1 uv run --extra dev pytest -q -m integration
```

Recommended pre-release check:

```bash
uv run --extra dev pytest -q
GLINER2_MCP_RUN_INTEGRATION=1 uv run --extra dev pytest -q -m integration
```

## License

MIT. See `LICENSE`.
