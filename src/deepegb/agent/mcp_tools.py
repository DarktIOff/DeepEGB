"""
MCP server adapters — connect external Model Context Protocol servers
as Agno tools.

Two MCP servers are supported:

1. **arXiv MCP** (blazickjp/arxiv-mcp-server) — exposes search/download/
   read_paper/list_papers tools.  Override launch command via
   ``DEEPEGB_ARXIV_MCP_CMD``.

2. **CosmoRAG MCP** (cosmorag-mcp) — exposes cosmorag_search /
   cosmorag_list_papers / cosmorag_get_paper / cosmorag_add_arxiv / …
   Override launch command via ``DEEPEGB_COSMORAG_MCP_CMD``.
   NOTE: direct Python import (cosmorag_tools.py) is preferred over MCP
   because it avoids subprocess overhead.  Use MCP only when the direct
   import fails (e.g. CosmoRAG is installed in a separate environment).

Usage
-----
    from deepegb.agent.mcp_tools import build_arxiv_mcp_tools, build_cosmorag_mcp_tools
    arxiv = await build_arxiv_mcp_tools()
    cosmorag = await build_cosmorag_mcp_tools()
    agent = Agent(..., tools=[arxiv, cosmorag])

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
    parts = [os.path.expanduser(part) if part.startswith("~") else part
             for part in shlex.split(raw)]
    if not parts:
        raise RuntimeError("DEEPEGB_ARXIV_MCP_CMD is empty")
    return parts[0], parts[1:]


def _import_mcp_tools_class():
    """Try the known import paths for Agno's MCPTools across versions."""
    # `agno.tools.mcp` typically requires the separate `mcp` package
    # (the official MCP Python SDK). If that's missing, Agno's import
    # of its mcp submodule will itself ImportError. Detect that case
    # specifically so the user gets a useful hint.
    try:
        import mcp                        # noqa: F401
        mcp_pkg_present = True
    except ImportError:
        mcp_pkg_present = False

    candidates = (
        "agno.tools.mcp",
        "agno.tools.mcp_tools",
        "agno.mcp",
        "agno.tools.mcp.client",
    )
    last_exc: Exception | None = None
    for mod_path in candidates:
        try:
            mod = __import__(mod_path, fromlist=["MCPTools"])
            if hasattr(mod, "MCPTools"):
                return mod.MCPTools
        except Exception as exc:    # noqa: BLE001
            last_exc = exc

    if not mcp_pkg_present:
        hint = ("the official MCP SDK is missing. "
                "Run: pip install mcp")
    else:
        hint = ("your installed Agno version may not ship MCP support. "
                "Update with: pip install -U 'agno>=1.0'")
    raise ImportError(
        f"Could not locate Agno's MCPTools class — {hint}. "
        f"Tried: {', '.join(candidates)}. MCP integration disabled; "
        f"all other DeepEGB tools (search, analyze, plot, relic_gw, "
        f"local RAG) continue to work."
    ) from last_exc


async def build_arxiv_mcp_tools() -> Any:
    """Construct an Agno MCPTools wrapper that launches the arXiv MCP server.

    Must be called from within an async context (Agno requirement).
    """
    MCPTools = _import_mcp_tools_class()
    cmd, args = _arxiv_mcp_command()
    full_cmd = " ".join([cmd, *args])
    # Different Agno versions accept different kwargs; try the common ones.
    for kwargs in (
        dict(command=full_cmd, timeout_seconds=60, transport="stdio"),
        dict(command=full_cmd, timeout_seconds=60),
        dict(command=full_cmd),
        dict(server=full_cmd),
    ):
        try:
            tools = MCPTools(**kwargs)
            break
        except TypeError:
            continue
    else:
        # Fall back to positional argument
        tools = MCPTools(full_cmd)
    # Agno's `initialize()` expects an already-open session and logs an error
    # instead of raising when called too early. Prefer `connect()`, which
    # creates the session and discovers tools end-to-end.
    if hasattr(tools, "connect"):
        await tools.connect()
    elif hasattr(tools, "initialize"):
        await tools.initialize()

    if not getattr(tools, "initialized", False):
        raise RuntimeError(
            "MCPTools did not reach the initialized state after connect()."
        )
    if not getattr(tools, "functions", None):
        raise RuntimeError(
            "MCPTools connected but exposed no arXiv tools. Check the MCP server command."
        )
    return tools


def has_arxiv_mcp_configured() -> bool:  # kept for backward compat
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


# ---------------------------------------------------------------------------
# CosmoRAG MCP adapter
# ---------------------------------------------------------------------------

DEFAULT_COSMORAG_MCP_CMD = "cosmorag-mcp"


def _cosmorag_mcp_command() -> str:
    """Return the shell command used to launch the CosmoRAG MCP server."""
    return os.environ.get("DEEPEGB_COSMORAG_MCP_CMD", DEFAULT_COSMORAG_MCP_CMD)


async def build_cosmorag_mcp_tools() -> Any:
    """Construct an Agno MCPTools wrapper that launches the CosmoRAG MCP server.

    Prefer the direct Python integration in ``cosmorag_tools.py`` over this
    function — use MCP only when CosmoRAG lives in a separate virtualenv.
    Must be called from within an async context (Agno requirement).
    """
    MCPTools = _import_mcp_tools_class()
    full_cmd = _cosmorag_mcp_command()

    for kwargs in (
        dict(command=full_cmd, timeout_seconds=120, transport="stdio"),
        dict(command=full_cmd, timeout_seconds=120),
        dict(command=full_cmd),
        dict(server=full_cmd),
    ):
        try:
            tools = MCPTools(**kwargs)
            break
        except TypeError:
            continue
    else:
        tools = MCPTools(full_cmd)

    if hasattr(tools, "connect"):
        await tools.connect()
    elif hasattr(tools, "initialize"):
        await tools.initialize()

    if not getattr(tools, "initialized", False):
        raise RuntimeError(
            "CosmoRAG MCPTools did not reach the initialized state after connect()."
        )
    if not getattr(tools, "functions", None):
        raise RuntimeError(
            "CosmoRAG MCPTools connected but exposed no tools. "
            "Check that cosmorag-mcp is installed: pip install 'cosmorag[mcp]'"
        )
    return tools


def has_cosmorag_mcp_configured() -> bool:
    """Return True when the cosmorag-mcp binary is discoverable on PATH."""
    import shutil
    cmd = _cosmorag_mcp_command().split()[0]
    return bool(shutil.which(cmd))
