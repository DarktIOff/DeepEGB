"""
MCP server adapters — connect external Model Context Protocol servers
(e.g. an arXiv search MCP) as Agno tools.

We default to **blazickjp/arxiv-mcp-server** (Python, pip-installable),
which exposes search/download/read_paper/list_papers tools. Override the
launch command via the `DEEPEGB_ARXIV_MCP_CMD` env var if you have a
different MCP server installed.

Usage
-----
    from deepegb.agent.mcp_tools import build_arxiv_mcp_tools
    arxiv = await build_arxiv_mcp_tools()        # async — Agno requirement
    agent = Agent(..., tools=[arxiv])

References
----------
* MCP spec: https://modelcontextprotocol.io/
* Agno MCPTools: https://docs.agno.com/tools/mcp
* arxiv-mcp-server: https://github.com/blazickjp/arxiv-mcp-server
"""
from __future__ import annotations

import os
import shlex
from typing import Any

# Default launch command for blazickjp/arxiv-mcp-server.
# `uvx` is the cheap-and-cheerful way to run pip tools in an ephemeral venv;
# users with a permanent install can override via DEEPEGB_ARXIV_MCP_CMD.
DEFAULT_ARXIV_MCP_CMD = "uvx arxiv-mcp-server --storage-path ~/.deepegb/arxiv_papers"


def _arxiv_mcp_command() -> tuple[str, list[str]]:
    """Return (command, args) for launching the arXiv MCP server."""
    raw = os.environ.get("DEEPEGB_ARXIV_MCP_CMD", DEFAULT_ARXIV_MCP_CMD)
    parts = shlex.split(raw)
    if not parts:
        raise RuntimeError("DEEPEGB_ARXIV_MCP_CMD is empty")
    return parts[0], parts[1:]


async def build_arxiv_mcp_tools() -> Any:
    """Construct an Agno MCPTools wrapper that launches the arXiv MCP server.

    Must be called from within an async context (Agno requirement). The
    returned tools list is consumed by `Agent(tools=[...])`.

    Returns the MCPTools instance ready to be passed to the Agno Agent.
    """
    try:
        from agno.tools.mcp import MCPTools
    except ImportError as exc:
        raise RuntimeError(
            "agno.tools.mcp is not available. "
            "Update Agno: `pip install -U agno`."
        ) from exc
    cmd, args = _arxiv_mcp_command()
    # MCPTools accepts a single command string in current Agno; let it parse.
    full_cmd = " ".join([cmd, *args])
    tools = MCPTools(command=full_cmd, timeout_seconds=60)
    await tools.initialize()
    return tools


def has_arxiv_mcp_configured() -> bool:
    """Heuristic: check whether the configured MCP launch command points to
    something real on disk (or is `uvx ...` which we trust)."""
    try:
        cmd, _ = _arxiv_mcp_command()
    except Exception:
        return False
    if cmd in ("uvx", "uv", "npx", "python", "python3"):
        return True   # trust common launchers
    # Otherwise check $PATH
    import shutil
    return shutil.which(cmd) is not None
