"""
Python code execution tool for processing FHIR resources.

Allows agents to write and execute Python code to analyze FHIR resources
retrieved via fhir_request_get. Resources are available in the
`retrieved_resources` variable, organized by resource type.
"""

import json
import traceback

_NOT_SET = object()

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



EXEC_TIMEOUT = 60  # seconds

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
        - Built-in functions: len, min, max, sum, sorted, list, dict, str, int, float, bool, range, enumerate, zip, any, all, abs, round
    """

    try:
        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        resources = mcp_server.get_task_resources()
    except Exception as e:
        return {"error": f"Failed to get resources: {e}"}

    if isinstance(resources, dict):
        resources = _normalize_retrieved_resources(resources)

    safe_builtins = {
        'len': len, 'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
        'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
        'str': str, 'int': int, 'float': float, 'bool': bool,
        'range': range, 'enumerate': enumerate, 'zip': zip,
        'any': any, 'all': all, 'abs': abs, 'round': round,
        'True': True, 'False': False, 'None': None,
    }

    exec_globals = {
        '__builtins__': safe_builtins,
        'json': json,
        're': __import__('re'),
        'datetime': __import__('datetime'),
        'math': __import__('math'),
        'statistics': __import__('statistics'),
        'retrieved_resources': resources,
        'answer': _NOT_SET,
    }

    try:
        exec(code, exec_globals)
    except Exception as e:
        error_info = traceback.format_exc()
        return {"error": f"Execution failed: {e}\n\nTraceback:\n{error_info}"}

    answer = exec_globals.get('answer', _NOT_SET)

    if answer is _NOT_SET:
        return {"error": "No answer variable set"}

    return {"answer": answer}