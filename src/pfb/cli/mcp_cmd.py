"""`pfb mcp`: start the MCP server over stdio (the same server as `python -m pfb.mcp`)."""

from __future__ import annotations

import typer

mcp_app = typer.Typer()


@mcp_app.command("mcp")
def mcp_command() -> None:
    """Serve the pfb tools to an MCP client over stdin and stdout (protocol only on stdout)."""
    # Imported here so other pfb commands do not pay for loading the MCP SDK.
    from pfb.mcp.server import serve_stdio

    serve_stdio()
