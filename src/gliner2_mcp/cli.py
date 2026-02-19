from __future__ import annotations

from gliner2_mcp.config import load_settings
from gliner2_mcp.server import create_server


def main() -> None:
    settings = load_settings()
    server = create_server(settings)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
