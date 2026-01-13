"""
MCP tool registry.

This module registers all tools available to purple agents via the MCP server.
Tools are automatically wrapped with logging when registered by MCPServer.

Available tools:
- fhir_request_get: Query FHIR resources, stores results for later processing
- fhir_request_post: Record FHIR write operations (no-op for evaluation)
- lookup_medical_code: Search local code tables (LOINC, SNOMED, etc.)
- analyze_drug_interactions: Check drug-drug interactions
- execute_python_code: Execute Python code with access to retrieved FHIR resources
"""

from .fhir_tools import FHIR_SCHEMA, fhir_request_get, fhir_request_post
from .medical_codes import lookup_medical_code
from .python_executor import execute_python_code
from .drug_labels import get_fda_drug_labels

TOOL_REGISTRY = {
    "fhir_request_get": fhir_request_get,
    "fhir_request_post": fhir_request_post,
    "lookup_medical_code": lookup_medical_code,
    "execute_python_code": execute_python_code,
    "get_fda_drug_labels": get_fda_drug_labels
}

def get_tool(name: str):
    """
    Get tool function by name for execution.

    Used by green agent in messaging mode to execute tools locally.

    Args:
        name: Tool name as registered in TOOL_REGISTRY

    Returns:
        The tool function

    Raises:
        ValueError: If tool name not found
    """
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name]
