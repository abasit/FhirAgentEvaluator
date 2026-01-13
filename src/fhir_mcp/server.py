"""
MCP server for FHIR agent evaluation.

This module provides an MCP (Model Context Protocol) server that exposes FHIR tools
to purple agents. It supports task-scoped logging to track which tools were called
and what results were returned for each evaluation task.

Key components:
- MCPServer: Main server class that registers tools and manages logging
- TaskScopedMCP: ASGI middleware that injects task_id into async context
- current_task_id: ContextVar for tracking the current task across async calls
"""

import contextlib
import functools
import json
import logging
from contextvars import ContextVar
from typing import Optional, Any

from mcp.server.fastmcp import FastMCP
from starlette.routing import Mount

from .tools import TOOL_REGISTRY


logger = logging.getLogger(__name__)

# Context var for task ID - isolated per async context, allowing concurrent tasks
current_task_id: ContextVar[Optional[str]] = ContextVar("task_id", default=None)


class MCPServer:
    """
    MCP server that exposes FHIR tools with task-scoped logging.

    The server wraps each registered tool to automatically log calls and results,
    keyed by task_id. This allows the green agent to retrieve tool usage after
    a purple agent completes a task.

    Attributes:
        base_url: Base URL where the server is accessible (e.g., "http://localhost:9009")
        app: The underlying FastMCP application
        tool_logs: Dict mapping task_id to list of tool call records
    """

    def __init__(self, base_url: str = ""):
        """
        Initialize the MCP server.

        Args:
            base_url: Base URL for constructing MCP endpoint URLs.
                      Used by green agent to tell purple agent where to connect.
        """
        self.base_url = base_url
        self.app = FastMCP(name="fhiragentbench-mcp", stateless_http=True, json_response=True)
        self.tool_logs: dict[str, list[dict[str, Any]]] = {}
        self.task_resources: dict[str, dict[str, list]] = {}

        self._register_tools()

        logger.info(f"MCP server initialized with {len(TOOL_REGISTRY)} tools")

    def _register_tools(self) -> None:
        """Register all tools from TOOL_REGISTRY with logging wrappers."""
        for name, fn in TOOL_REGISTRY.items():
            @functools.wraps(fn)
            def wrapped(*args, __fn=fn, __name=name, **kwargs):
                result = __fn(*args, **kwargs)
                self.log_tool_call(__name, kwargs, result)
                return result

            self.app.add_tool(wrapped, name=name)

    def log_tool_call(self, tool_name: str, args: dict, result: Any) -> None:
        """
        Log a tool call for the current task.

        Uses the current_task_id context var to determine which task this call
        belongs to. If no task_id is set (e.g., during testing), the call is not logged.

        Args:
            tool_name: Name of the tool that was called
            args: Arguments passed to the tool
            result: Return value from the tool
        """
        task_id = current_task_id.get()
        if task_id is None:
            return
        if task_id not in self.tool_logs:
            self.tool_logs[task_id] = []
        self.tool_logs[task_id].append({
            "tool": tool_name,
            "args": args,
            "result": result,
        })
        logger.debug(f"[{task_id}] {tool_name}({args}) ->\n{str(result)}")

    def merge_task_resources(self, resources: dict[str, list]) -> None:
        """
        Merge the given resources into the task-scoped storage for the current task.

        Args:
            resources: dict of resource_type -> list of resource objects
        """
        task_id = current_task_id.get()
        if not task_id:
            return

        task_dict = self.task_resources.setdefault(task_id, {})
        for rt, items in resources.items():
            task_dict.setdefault(rt, []).extend(items)

    def get_task_resources(self, task_id: str = None) -> dict:
        task_id = task_id or current_task_id.get()
        if not task_id:
            return {}
        return self.task_resources.get(task_id, {})

    def get_tool_logs(self, task_id: str) -> list[dict[str, Any]]:
        """
        Get tool logs for a task.

        Args:
            task_id: The task identifier

        Returns:
            List of tool call records, each containing:
            - tool: Name of the tool called
            - args: Arguments passed
            - result: Return value (may contain "error" key on failure)
        """
        return self.tool_logs.get(task_id, [])

    def clear_task(self, task_id: str) -> None:
        """
        Clear all stored data for a completed task.

        Args:
            task_id: The task identifier
        """
        self.tool_logs.pop(task_id, None)
        self.task_resources.pop(task_id, None)
        logger.debug(f"Cleared data for task {task_id}")

    def get_routes(self) -> Mount:
        """
        Create Starlette routes for the MCP server.

        Returns a Mount that handles requests to /tasks/{task_id}/mcp.
        The task_id is extracted and injected into the async context for logging.
        """
        mcp_asgi = self.app.streamable_http_app()
        scoped = TaskScopedMCP(mcp_asgi)
        return Mount("/tasks/{task_id}", app=scoped)

    @contextlib.asynccontextmanager
    async def lifespan(self, app):
        """
        Lifespan context manager for Starlette integration.

        Manages the MCP session manager lifecycle. Must be passed to
        Starlette app as the lifespan parameter.

        Args:
            app: The Starlette application (required by Starlette, not used)
        """
        async with self.app.session_manager.run():
            yield

    async def get_tool_definitions(self) -> str:
        """
        Get JSON schema definitions for all registered tools.

        Used by green agent to include tool descriptions in prompts
        for messaging mode (non-MCP) communication.

        Returns:
            JSON string containing list of tool schemas
        """
        tools = await self.app.list_tools()
        tools = [tool.model_dump(mode="json", exclude_none=True) for tool in tools]
        return json.dumps(tools, indent=2)

    def get_mcp_url(self, mcp_task_id: str) -> str:
        """
        Get the task specific MCP URL.

        Used by green agent to include mcp url in prompt
        for MCP agent.

        Returns:
            url string
        """
        return f"{self.base_url}/tasks/{mcp_task_id}/mcp"

class TaskScopedMCP:
    """
    ASGI middleware that injects task_id into the async context.

    Extracts task_id from the URL path parameters and sets it in
    the current_task_id context var before forwarding the request
    to the MCP ASGI app. This allows tool logging to be scoped per task.
    """

    def __init__(self, mcp_asgi):
        """
        Args:
            mcp_asgi: The MCP ASGI application to wrap
        """
        self.mcp_asgi = mcp_asgi

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            task_id = scope.get("path_params", {}).get("task_id")
            token = current_task_id.set(task_id)
            try:
                await self.mcp_asgi(scope, receive, send)
            finally:
                current_task_id.reset(token)
        else:
            await self.mcp_asgi(scope, receive, send)
