"""
MCP Server package for FHIR Agent Benchmark.

Provides tool execution (FHIR queries, code execution, medical code lookup)
via Model Context Protocol with task-scoped resource tracking.
"""

from typing import Optional

from .server import MCPServer
from .utils import verify_tool_access, execute_tool
from .tools import get_tool

_mcp_server: Optional[MCPServer] = None


def init_mcp_server(base_url: str = "") -> MCPServer:
    """
    Initialize the global MCP server instance.

    Should be called once at application startup.

    Args:
        base_url: Base URL where the server is accessible

    Returns:
        The initialized MCPServer instance
    """
    global _mcp_server
    _mcp_server = MCPServer(base_url=base_url)
    return _mcp_server


def get_mcp_server() -> MCPServer:
    """
    Get the global MCP server instance.

    Raises:
        RuntimeError: If MCP server is not initialized.
    """
    if _mcp_server is None:
        raise RuntimeError("MCP server not initialized.")
    return _mcp_server