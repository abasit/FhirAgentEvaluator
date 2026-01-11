"""
FHIR tools for querying and recording operations against a FHIR server.

These tools provide the primary interface for agents to interact with patient data.
Results from fhir_request_get are stored in task-scoped storage and can be accessed
via execute_python_code using the `retrieved_resources` variable.

Supported FHIR resource types:
- Patient, Encounter, Condition, MedicationRequest, Procedure
- Observation, MedicationAdministration, Location, Specimen, Medication
"""
import json
import logging

from common.fhir_client import get_fhir_client

logger = logging.getLogger(__name__)


SUPPORTED_TYPES = [
    "Patient",
    "Encounter",
    "Condition",
    "MedicationRequest",
    "Procedure",
    "Observation",
    "MedicationAdministration",
    "Location",
    "Specimen",
    "Medication",
]


def fhir_request_get(query_string: str) -> dict:
    """
    Perform a FHIR GET request and store results for later processing.

    Results are automatically stored in task-scoped storage, organized by
    resource type. Use execute_python_code with `retrieved_resources` to
    access the full data.

    Args:
        query_string: Relative FHIR path or search query.
            Examples:
                - "Patient/dd2bf984-33c3-5874-8f68-84113327877e"
                - "Observation?patient=<FHIR-ID>&code=220210"
                - "Observation?patient=<FHIR-ID>&code=220210&date=ge2133-12-31"

    Returns:
        dict: Summary of retrieved resources.
            On success:
            {
                "message": "Retrieved 47 resources across 2 types",
                "resource_counts": {"Observation": 45, "Patient": 2}
            }
            On failure:
            {
                "error": "<error message>"
            }

    Notes:
        - Use lookup_medical_code to find codes before querying
        - Use the Patient FHIR ID from context, not numeric IDs from questions
        - Results accumulate across multiple calls within the same task
    """
    logger.debug(f"FHIR GET: {query_string}")
    try:
        client = get_fhir_client()
        response = client.session.get(f"{client.fhir_store_url}/{query_string}")
        response.raise_for_status()


        all_resources = []
        if "entry" in response.json():
            for entry in response.json()["entry"]:
                all_resources.append(entry["resource"])

        resources_by_type = {}
        for resource in all_resources:
            rt = resource.get("resourceType", "Unknown")
            if rt not in resources_by_type:
                resources_by_type[rt] = []
            resources_by_type[rt].append(resource)

        # Merge into task-scoped storage
        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        mcp_server.merge_task_resources(resources_by_type)

        # Summarise results
        total_resources = sum(len(v) for v in resources_by_type.values())
        resource_counts = {rt: len(items) for rt, items in resources_by_type.items()}

        logger.debug(f"FHIR response: {total_resources} resources")

        return {
            "message": f"Retrieved {total_resources} resources across {len(resource_counts)} types",
            "resource_counts": resource_counts
        }
    except Exception as e:
        logger.warning(f"FHIR request failed: {e}")
        return {"error": str(e)}


def fhir_request_post(resource: dict) -> dict:
    """
    Record a FHIR POST operation without mutating the backend.

    Used for MedAgentBench action tasks where agents need to demonstrate
    they would create/update resources. The operation is logged but not
    actually executed against the FHIR server.

    Args:
        resource: FHIR resource the agent intends to create/update.
            Must include "resourceType" field.
            Example:
            {
                "resourceType": "MedicationRequest",
                "subject": {"reference": "Patient/123"},
                "medicationCodeableConcept": {...}
            }

    Returns:
        dict: Acknowledgement of the recorded operation.
            On success:
            {
                "message": "FHIR POST recorded",
                "resource_type": "MedicationRequest"
            }
            On failure:
            {
                "error": "<error message>"
            }
    """
    resource_type = resource.get("resourceType", "Unknown")

    logger.debug(f"FHIR POST ({resource_type}):\n{json.dumps(resource, indent=2)}")

    try:
        return {
            "message": "FHIR POST recorded",
            "resource_type": resource_type
        }

    except Exception as e:
        logger.warning(f"FHIR POST failed: {e}")
        return {"error": str(e)}