"""
MCP tool registry.

Registers all tools available to purple agents via the MCP server.
Tools are automatically wrapped with logging when registered by MCPServer.
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
    "get_fda_drug_labels": get_fda_drug_labels,
}


def get_tool(name: str):
    """Get tool function by name. Raises ValueError if not found."""
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_REGISTRY[name]