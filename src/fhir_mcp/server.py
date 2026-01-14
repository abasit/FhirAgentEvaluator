"""
MCP server for FHIR agent evaluation.

Provides an MCP (Model Context Protocol) server that exposes FHIR tools
to purple agents with task-scoped logging to track tool calls and results.
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

current_task_id: ContextVar[Optional[str]] = ContextVar("task_id", default=None)


class MCPServer:
    """
    MCP server that exposes FHIR tools with task-scoped logging.

    Wraps each registered tool to automatically log calls and results,
    keyed by task_id. This allows the green agent to retrieve tool usage
    after a purple agent completes a task.
    """

    def __init__(self, base_url: str = ""):
        """
        Initialize the MCP server.

        Args:
            base_url: Base URL for constructing MCP endpoint URLs
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
        """Log a tool call for the current task."""
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
        """Merge resources into task-scoped storage for the current task."""
        task_id = current_task_id.get()
        if not task_id:
            return

        task_dict = self.task_resources.setdefault(task_id, {})
        for rt, items in resources.items():
            task_dict.setdefault(rt, []).extend(items)

    def get_task_resources(self, task_id: str = None) -> dict:
        """Get stored resources for a task."""
        task_id = task_id or current_task_id.get()
        if not task_id:
            return {}
        return self.task_resources.get(task_id, {})

    def get_tool_logs(self, task_id: str) -> list[dict[str, Any]]:
        """Get tool call logs for a task."""
        return self.tool_logs.get(task_id, [])

    def clear_task(self, task_id: str) -> None:
        """Clear all stored data for a completed task."""
        self.tool_logs.pop(task_id, None)
        self.task_resources.pop(task_id, None)
        logger.debug(f"Cleared data for task {task_id}")

    def get_routes(self) -> Mount:
        """Create Starlette routes for the MCP server at /tasks/{task_id}/mcp."""
        mcp_asgi = self.app.streamable_http_app()
        scoped = TaskScopedMCP(mcp_asgi)
        return Mount("/tasks/{task_id}", app=scoped)

    @contextlib.asynccontextmanager
    async def lifespan(self, app):
        """Lifespan context manager for Starlette integration."""
        async with self.app.session_manager.run():
            yield

    async def get_tool_definitions(self) -> str:
        """Get JSON schema definitions for all registered tools."""
        tools = await self.app.list_tools()
        tools = [tool.model_dump(mode="json", exclude_none=True) for tool in tools]
        return json.dumps(tools, indent=2)

    def get_mcp_url(self, mcp_task_id: str) -> str:
        """Get the task-specific MCP URL."""
        return f"{self.base_url}/tasks/{mcp_task_id}/mcp"


class TaskScopedMCP:
    """ASGI middleware that injects task_id into the async context."""

    def __init__(self, mcp_asgi):
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