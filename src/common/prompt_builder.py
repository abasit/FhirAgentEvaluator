"""
Task prompt construction for FHIR Agent Benchmark.

Builds system prompts for MCP mode (tool access via URL) and messaging mode
(iterative tool calls with definitions).
"""

from fhir_mcp import get_mcp_server
from fhir_mcp.tools import FHIR_SCHEMA


RESPOND_ACTION_NAME = "response"

TASK_INSTRUCTIONS = """
Task instructions:
- Use the Patient FHIR ID provided in context, not the numeric ID in the question.
- Provide answers in the same format as retrieved data. If multiple answers, provide all in a list.
"""

RESPONSE_FORMAT = f"""
Response format:
- Respond in the JSON format.
- Wrap your response in <json>...</json> tags.
- For final answer, use: {{"name": "{RESPOND_ACTION_NAME}", "kwargs": {{"content": "The final answer is: <your answer>"}}}}
"""


def build_task_prompt_mcp(mcp_task_id: str) -> str:
    """Build system prompt for MCP mode with tool server URL."""
    mcp_server = get_mcp_server()
    mcp_url = mcp_server.get_mcp_url(mcp_task_id)

    tool_access = f"Tools are available via MCP server at: {mcp_url}"

    return f"""
{TASK_INSTRUCTIONS}

{RESPONSE_FORMAT}

{FHIR_SCHEMA}

{tool_access}
"""


async def build_task_prompt_messaging() -> str:
    """Build system prompt for messaging mode with inline tool definitions."""
    mcp_server = get_mcp_server()
    tool_definitions = await mcp_server.get_tool_definitions()

    tool_access = f"""
Available tools:
{tool_definitions}

You may call only one tool at a time.
Tool call format:
<json>{{"name": "<tool_name>", "kwargs": {{"arg": "value"}}}}</json>
"""

    return f"""
{TASK_INSTRUCTIONS}

{RESPONSE_FORMAT}

{FHIR_SCHEMA}

{tool_access}
"""