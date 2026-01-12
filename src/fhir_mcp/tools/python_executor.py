"""
Python code execution tool for processing FHIR resources.

This tool allows agents to write and execute Python code to analyze
FHIR resources retrieved via fhir_request_get. Resources are available
in the `retrieved_resources` variable, organized by resource type.

Example usage by agent:
    1. Call fhir_request_get to fetch Observations
    2. Call execute_python_code with code that processes retrieved_resources
    3. Set the 'answer' variable with the final result

Available globals in executed code:
    - retrieved_resources: dict of resource_type -> list of FHIR resources
    - json, re, datetime, math, statistics: common Python modules
"""

import json
import logging
import traceback

logger = logging.getLogger(__name__)

def normalize_retrieved_resources(resources: dict) -> dict:
    """
    Normalize retrieved resources to consistent list format.

    Handles edge cases where resources might be JSON strings or
    single dicts instead of lists.

    Args:
        resources: Raw resources dict from task storage

    Returns:
        dict: Normalized resources with all values as lists
    """
    normalized = {}

    for key, value in resources.items():
        try:
            # Parse JSON strings
            if isinstance(value, str) and value.strip().startswith(('{', '[')):
                parsed = json.loads(value)
                normalized[key] = [parsed] if isinstance(parsed, dict) else parsed
            # Wrap single dicts in lists
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

    The code has access to all FHIR resources fetched via fhir_request_get
    in the current task, available as `retrieved_resources`. The code must
    set an `answer` variable with the final result.

    Args:
        code: Python code to execute.
            Example:
            '''
            obs = retrieved_resources.get("Observation", [])
            values = [o["valueQuantity"]["value"] for o in obs if "valueQuantity" in o]
            answer = min(values) if values else None
            '''

    Returns:
        dict: Execution result.
            On success:
            {
                "answer": <value of answer variable>
            }
            On failure or no answer set:
            {
                "error": "<error message>"
            }

    Available globals:
        - retrieved_resources: dict mapping resource types to lists of FHIR resources
        - json: JSON module for parsing/serialization
        - re: Regular expressions module
        - datetime: Date/time module
        - math: Math module
        - statistics: Statistics module

    Notes:
        - Always set the `answer` variable with your final result
        - Resources accumulate across multiple fhir_request_get calls
        - Use try/except in your code for robust error handling
    """
    logger.debug(f"Executing code:\n{code}")

    try:
        # Add common imports and the retrieved resources to global scope
        exec_globals = {
            'json': json,
            're': __import__('re'),
            'datetime': __import__('datetime'),
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            'answer': None,  # Initialize answer variable
        }

        # Get and normalize retrieved resources
        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        resources = mcp_server.get_task_resources()

        if isinstance(resources, dict):
            resources = normalize_retrieved_resources(resources)

        exec_globals['retrieved_resources'] = resources

        logger.debug(f"Available resources: {list(resources.keys())}")

        # Execute the code
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
