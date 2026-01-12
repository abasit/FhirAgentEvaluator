from fhir_mcp import get_mcp_server
from fhir_mcp.tools import SUPPORTED_TYPES

RESPOND_ACTION_NAME = "response"


def build_task_prompt_mcp(mcp_task_id: str) -> str:
    """
    Build the system prompt for the purple agent.
    """
    mcp_server = get_mcp_server()
    mcp_url = f"{mcp_server.base_url}/tasks/{mcp_task_id}/mcp"

    return f"""
Use tools available on the MCP server available at: {mcp_url}

Task instructions:
- Always use the Patient FHIR ID provided in context; ignore numeric patient IDs in the question text.
- Retrieve only what you need; avoid broad/all-observation pulls (no bulk fetches).
- Provide all answers found in the same format as the retrieved data. If you cannot find the answer, clearly say so.
- Do not guess attributes and do not repeat the same failing action.

Response format:
- Wrap your response in <json>...</json>.
- For final answer use:
{{"name": "{RESPOND_ACTION_NAME}", "kwargs": {{"content": "The final answer is: ..."}}}}

IMPORTANT: The content for your final message must start with 'The final answer is:' followed by your conclusion.
"""


async def build_task_prompt_messaging() -> str:
    """Build the system prompt with tool descriptions."""
    mcp_server = get_mcp_server()
    all_tools = await mcp_server.get_tool_definitions()

    return f"""
Here's a list of tools you can use (you can use one tool at a time):
{all_tools}

Available FHIR resource types: {', '.join(SUPPORTED_TYPES)}.
You can only call on these FHIR resources types for retrieval.

To answer questions about patient data:
1. Always use the Patient FHIR ID provided in context; do not use the numeric patient ID from the question text.
2. Use lookup_medical_code to find codes for medical items, labs, and procedures
3. Use fhir_request_get to get data from the FHIR server
4. Analyze the retrieved data to answer the question

If there are multiple answers, provide all of them.
When you provide answers, make sure to provide them in the same format as they are in the retrieved data. If multiple answers are provided, provide them all in a list.
If you cannot find the answer or relevant patient data, clearly state that you cannot find the information.
Do not guess attributes; instead, use the provided tool to retrieve the data.
Do not get stuck or repeat the same action.

Please respond in the JSON format. Please wrap the JSON part with <json>...</json> tags.
The JSON can be either a single action or a list of actions.

Tool Call:
{{"name": "tool_name", "kwargs": {{"arg": "value"}}}}

For final response:
{{"name": "{RESPOND_ACTION_NAME}", "kwargs": {{"content": "The final answer is: ..."}}}}

IMPORTANT: The content for your final message must start with 'The final answer is:' followed by your conclusion. This is required for proper processing.
"""
