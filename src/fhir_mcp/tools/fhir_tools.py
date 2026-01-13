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


FHIR_SCHEMA = """
Available Resource Types and FHIR Resource Schemas:

Condition: {
  "resourceType": "Condition",
  "id": "<resource_id>",
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  "subject": {"reference": "Patient/<id>"},
  "onsetDateTime": "<datetime>"
}

Encounter: {
  "resourceType": "Encounter",
  "id": "<resource_id>",
  "class": {"code": "IMP | AMB"},
  "period": {"start": "<datetime>", "end": "<datetime>"},
  "subject": {"reference": "Patient/<id>"}
}

Location: {
  "resourceType": "Location",
  "id": "<resource_id>",
  "name": "<location_name>",
  "type": [{"coding": [{"code": "<type>"}]}],
  "physicalType": {"coding": [{"code": "<physical_type>"}]}
}

Medication: {
  "resourceType": "Medication",
  "id": "<resource_id>",
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]}
}

MedicationAdministration: {
  "resourceType": "MedicationAdministration",
  "id": "<resource_id>",
  "status": "completed | in-progress | stopped",
  "medicationCodeableConcept": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  // OR "medicationReference": {"reference": "Medication/<id>"},
  "subject": {"reference": "Patient/<id>"},
  "effectiveDateTime": "<datetime>",
  // OR "effectivePeriod": {"start": "<datetime>", "end": "<datetime>"},
  "dosage": {
    "route": {"coding": [{"code": "<route>"}]},
    "dose": {"value": <n>, "unit": "<unit>"}
  }
}

MedicationRequest: {
  "resourceType": "MedicationRequest",
  "id": "<resource_id>",
  "status": "active | completed | cancelled | stopped",
  "intent": "order | proposal | plan",
  "medicationCodeableConcept": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  // OR "medicationReference": {"reference": "Medication/<id>"},
  "subject": {"reference": "Patient/<id>"},
  "authoredOn": "<datetime>",
  "dosageInstruction": [{
    "route": "<route>",
    "doseAndRate": [{"doseQuantity": {"value": <n>, "unit": "<unit>"}, "rateQuantity": {"value": <n>, "unit": "<unit>"}}]
  }]
}

Observation: {
  "resourceType": "Observation",
  "id": "<resource_id>",
  "status": "final | preliminary",
  "category": [{"coding": [{"code": "laboratory | vital-signs"}]}],
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  "subject": {"reference": "Patient/<id>"},
  "effectiveDateTime": "<datetime>",
  "valueQuantity": {"value": <n>, "unit": "<unit>"}
  // OR "valueString": "<text>"
}

Patient: {
  "resourceType": "Patient",
  "id": "<resource_id>",
  "name": [{"given": ["<first>"], "family": "<last>"}],
  "birthDate": "<date>",
  "gender": "male | female"
}

Procedure: {
  "resourceType": "Procedure",
  "id": "<resource_id>",
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  "subject": {"reference": "Patient/<id>"},
  "performedDateTime": "<datetime>"
}

Specimen: {
  "resourceType": "Specimen",
  "id": "<resource_id>",
  "type": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  "subject": {"reference": "Patient/<id>"},
  "collection": {
    "collectedDateTime": "<datetime>",
    "bodySite": {"coding": [{"code": "<site>"}]}
  }
}
"""


def fhir_request_get(query_string: str) -> dict:
    """
    Perform a FHIR GET request.

    Results are stored in `retrieved_resources` (dict keyed by resource type),
    accessible via execute_python_code. Results accumulate across calls.

    Args:
        query_string: FHIR search query.
            Format: "<ResourceType>?patient=<fhir_id>&<param>=<value>&..."
            Examples:
                - "Patient/<fhir_id>"
                - "MedicationRequest?patient=<fhir_id>&status=active"
                - "Observation?patient=<fhir_id>&code=220210&date=ge2133-12-31"

    Returns:
        dict: Summary of retrieved resources.
            On success: {
                "message": "Retrieved 47 resources across 2 types",
                "resource_counts": {"Observation": 45, "Patient": 2}
            }
            On error: {"error": "<message>"}

    Note: Use lookup_medical_code to find codes before querying.
    """
    logger.debug(f"FHIR GET: {query_string}")
    try:
        client = get_fhir_client()

        # Fetch all resources via pagination
        all_resources = client.search_with_pagination(query_string)

        resources_by_type = {}
        for resource in all_resources:
            rt = resource.get("resourceType", "Unknown")
            resources_by_type.setdefault(rt, []).append(resource)

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


def fhir_request_post(resource_type: str, params: dict) -> dict:
    """
    Create a FHIR resource.

    Can be used to:
    - Record vital signs (Observation)
    - Order medications (MedicationRequest)
    - Order lab tests or referrals (ServiceRequest)

    Args:
        resource_type: One of "Observation", "MedicationRequest", "ServiceRequest"
        params: Flat dict of parameters (varies by resource_type)

    For Observation (vitals):
        - patient_id: Patient FHIR ID
        - code: Vital code (e.g., "220050")
        - value: Measurement with unit (e.g., "120 mmHg")
        - datetime: When measured (ISO format)

    For MedicationRequest:
        - patient_id: Patient FHIR ID
        - medication_code: NDC code
        - dose_value: Dose amount (e.g., 2)
        - dose_unit: Dose unit (e.g., "g")
        - rate_value: Infusion rate amount (e.g., 2)
        - rate_unit: Rate unit (e.g., "h")
        - route: Administration route (e.g., "IV")
        - datetime: Order datetime (ISO format)

    For ServiceRequest (lab orders, referrals):
        - patient_id: Patient FHIR ID
        - code: Procedure code (SNOMED)
        - datetime: Order datetime (ISO format)
        - note: Optional free text comment

    Returns:
        {"message": "<resource_type> recorded", "params": <params>} or {"error": "<message>"}
    """
    valid_types = ["Observation", "MedicationRequest", "ServiceRequest"]
    if resource_type not in valid_types:
        return {"error": f"Invalid resource_type. Must be one of: {valid_types}"}

    if not params:
        return {"error": "params is required"}

    if not params.get("patient_id"):
        return {"error": "patient_id is required in params"}

    logger.debug(f"FHIR POST {resource_type}: {params}")

    return {"message": f"{resource_type} recorded", "params": params}