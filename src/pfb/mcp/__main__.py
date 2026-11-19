"""`python -m pfb.mcp`: serve the pfb tools over stdio (the command .mcp.json gives Claude Code)."""

from __future__ import annotations

from pfb.mcp.server import serve_stdio

if __name__ == "__main__":
    serve_stdio()
