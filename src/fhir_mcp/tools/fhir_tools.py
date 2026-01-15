"""
FHIR tools for querying and creating resources.

Provides the primary interface for agents to interact with patient data.
Results from fhir_request_get are stored in task-scoped storage and can be
accessed via execute_python_code using the `retrieved_resources` variable.
"""

import logging

from common.fhir_client import get_fhir_client

logger = logging.getLogger(__name__)


FHIR_SCHEMA = """
Available Resource Types and FHIR Resource Schemas:

Condition: {
  "resourceType": "Condition",
  "id": "<resource_id>",
  "subject": {"reference": "Patient/<id>"},
  "encounter": {"reference": "Encounter/<id>"},
  "category": [{"coding": [{"system": "<system>", "code": "<category_code>", "display": "<category_name>"}]}],
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<diagnosis_name>"}]}
}

Encounter: {
  "resourceType": "Encounter",
  "id": "<resource_id>",
  "identifier": [{"system": "<system>", "value": "<encounter_number>"}],
  // identifier.system:
  //   "http://mimic.mit.edu/fhir/mimic/identifier/encounter-hosp" = hospital visit
  //   "http://mimic.mit.edu/fhir/mimic/identifier/encounter-icu" = ICU stay
  //   "http://mimic.mit.edu/fhir/mimic/identifier/encounter-ed" = emergency dept
  "status": "finished",
  "class": {"system": "<system>", "code": "IMP | AMB | EMER | OBSENC | ACUTE", "display": "<class_name>"},
  "subject": {"reference": "Patient/<id>"},
  "period": {"start": "<datetime>", "end": "<datetime>"},
  "location": [{"location": {"reference": "Location/<id>"}, "period": {"start": "<datetime>", "end": "<datetime>"}}],
  // location array tracks care unit transfers - use Location reference to get unit name
  "partOf": {"reference": "Encounter/<parent_id>"}  // ICU encounters reference parent hospital encounter
}

Location: {
  "resourceType": "Location",
  "id": "<resource_id>",
  "status": "active",
  "name": "<careunit_name>",
  "physicalType": {"coding": [{"system": "<system>", "code": "<code>", "display": "<type>"}]}
}

Medication: {
  "resourceType": "Medication",
  "id": "<resource_id>",
  "identifier": [
    {"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-medication-ndc", "value": "<ndc_code>"},
    {"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-medication-formulary-drug-cd", "value": "<formulary_code>"},
    {"system": "http://mimic.mit.edu/fhir/mimic/CodeSystem/mimic-medication-name", "value": "<medication_name>"}
  ],
  // To get medication name: find identifier where system contains "medication-name"
  "code": {"coding": [{"system": "<system>", "code": "<ndc_code>"}]},
  "status": "active"
}

MedicationRequest: {
  "resourceType": "MedicationRequest",
  "id": "<resource_id>",
  "status": "completed | stopped | unknown",
  "authoredOn": "<datetime>",
  "subject": {"reference": "Patient/<id>"},
  // Medication can be inline OR referenced:
  "medicationCodeableConcept": {"coding": [{"system": "<system>", "code": "<medication_name>"}]},
  // -OR-
  "medicationReference": {"reference": "Medication/<medication_id>"},  // Must be retrieved to get medication info
  "dosageInstruction": [{
    "route": {"coding": [{"system": "<system>", "code": "<route_code>"}]},
    "doseAndRate": [{"doseQuantity": {"value": <n>, "unit": "<unit>"}, "rateQuantity": {"value": <n>, "unit": "<unit>"}}]
  }],
  "dispenseRequest": {
    "validityPeriod": {
      "start": "<datetime>",  // When patient starts medication
      "end": "<datetime>"     // When patient ends medication
    }
  }
}

Observation: {
  "resourceType": "Observation",
  "id": "<resource_id>",
  "subject": {"reference": "Patient/<id>"},
  "status": "final | preliminary",
  "category": [{"coding": [{"code": "laboratory | vital-signs | Output"}]}],
  // category: "laboratory" = labs/micro, "vital-signs" = vitals, "Output" = output events (case-sensitive)
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  "effectiveDateTime": "<datetime>",
  "valueQuantity": {"value": <n>, "unit": "<unit>"},
  // OR "valueString": "<text>"  // For microbiology test results
  "specimen": {"reference": "Specimen/<id>"}  // Present for lab and microbiology tests
}

Patient: {
  "resourceType": "Patient",
  "id": "<resource_id>",
  "identifier": [{"system": "<system>", "value": "<patient_id>"}],
  "name": [{"family": "<name>"}],
  "gender": "male | female | other | unknown",
  "birthDate": "<date>"
}

Procedure: {
  "resourceType": "Procedure",
  "id": "<resource_id>",
  "code": {"coding": [{"system": "<system>", "code": "<code>", "display": "<name>"}]},
  "subject": {"reference": "Patient/<id>"},
  "performedDateTime": "<datetime>"
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
        all_resources = client.search_with_pagination(query_string)

        resources_by_type = {}
        for resource in all_resources:
            rt = resource.get("resourceType", "Unknown")
            resources_by_type.setdefault(rt, []).append(resource)

        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        mcp_server.merge_task_resources(resources_by_type)

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
        - medication_code: Medication code
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

    return {"message": f"{resource_type} recorded"}