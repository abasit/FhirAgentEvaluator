"""
Utility functions for FHIR MCP server.

Provides connection verification and health checking for the FHIR server.
"""

import logging
import time


logger = logging.getLogger(__name__)

def check_fhir_connection() -> None:
    """
    Check that the FHIR server is accessible.

    Performs a simple count query against the Patient resource.

    Raises:
        RuntimeError: If the FHIR server returns an error
    """
    from .tools.fhir_tools import fhir_request_get

    result = fhir_request_get(query_string="Patient?_summary=count")
    if "error" in result:
        raise RuntimeError(result["error"])


def verify_tool_access(max_retries: int = 10, delay_seconds: int = 30) -> None:
    """
    Wait for FHIR server to become available.

    Useful during startup when the FHIR server container may still be initializing.

    Args:
        max_retries: Maximum number of connection attempts
        delay_seconds: Seconds to wait between retries

    Raises:
        RuntimeError: If connection fails after all retries exhausted
    """
    for attempt in range(1, max_retries + 1):
        logger.info(f"Checking FHIR connection (attempt {attempt}/{max_retries})")
        try:
            check_fhir_connection()
            logger.info("FHIR connection verified")
            return
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Could not connect to FHIR database after {max_retries} attempts") from e
            logger.warning(f"FHIR connection failed: {e}, retrying in {delay_seconds}s...")
            time.sleep(delay_seconds)


def execute_tool(tool_name: str, tool_args: dict) -> dict:
    """
    Execute a tool by name and log the call.

    Looks up the tool function from the registry, executes it with the
    provided arguments, and logs the call to the MCP server's task-scoped
    storage (using the current_task_id context).

    Args:
        tool_name: Name of the tool to execute (e.g., "fhir_request_get")
        tool_args: Dictionary of arguments to pass to the tool

    Returns:
        dict: The tool's return value

    Raises:
        ValueError: If tool_name is not found in the registry
        Exception: Any exception raised by the tool function
    """
    from fhir_mcp import get_tool, get_mcp_server

    tool_function = get_tool(tool_name)
    result = tool_function(**tool_args)

    # Log the tool call (uses current_task_id context)
    get_mcp_server().log_tool_call(tool_name, tool_args, result)

    return result
