"""
Python code execution tool for processing FHIR resources.

Allows agents to write and execute Python code to analyze FHIR resources
retrieved via fhir_request_get. Resources are available in the
`retrieved_resources` variable, organized by resource type.
"""

import json
import logging
import traceback

logger = logging.getLogger(__name__)


def _normalize_retrieved_resources(resources: dict) -> dict:
    """Normalize retrieved resources to consistent list format."""
    normalized = {}

    for key, value in resources.items():
        try:
            if isinstance(value, str) and value.strip().startswith(('{', '[')):
                parsed = json.loads(value)
                normalized[key] = [parsed] if isinstance(parsed, dict) else parsed
            elif isinstance(value, dict):
                normalized[key] = [value]
            else:
                normalized[key] = value
        except (json.JSONDecodeError, TypeError, ValueError):
            normalized[key] = value

    return normalized


def execute_python_code(code: str) -> dict:
    """
    Execute Python code with access to retrieved FHIR resources.

    The variable `retrieved_resources` contains all FHIR data fetched via
    fhir_request_get in the current task. Set an `answer` variable with
    the final result.

    Args:
        code: Python code to execute.
            Example:
            '''
            obs = retrieved_resources.get("Observation", [])
            values = [o["valueQuantity"]["value"] for o in obs if "valueQuantity" in o]
            answer = min(values) if values else None
            '''

    Returns:
        dict: {"answer": <value>} on success, {"error": "<message>"} on failure.

    Available:
        - retrieved_resources: dict of resource_type -> list of FHIR resources
        - json, re, datetime, math, statistics modules
    """
    logger.debug(f"Executing code:\n{code}")

    try:
        exec_globals = {
            'json': json,
            're': __import__('re'),
            'datetime': __import__('datetime'),
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            'answer': None,
        }

        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        resources = mcp_server.get_task_resources()

        if isinstance(resources, dict):
            resources = _normalize_retrieved_resources(resources)

        exec_globals['retrieved_resources'] = resources

        logger.debug(f"Available resources: {list(resources.keys())}")

        exec(code, exec_globals)

        answer = exec_globals.get('answer', None)

        if answer is not None:
            logger.debug(f"Code result: {answer}")
            return {"answer": answer}
        else:
            logger.warning("Code executed but no answer variable set")
            return {"error": "Code executed successfully (no answer variable set)"}

    except Exception as e:
        error_info = traceback.format_exc()
        logger.warning(f"Code execution failed: {e}")
        return {"error": f"Error executing code: {e}\n\nFull traceback:\n{error_info}"}