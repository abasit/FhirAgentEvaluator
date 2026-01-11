import json
import re
import time
from ast import literal_eval
from typing import Any, Dict, List

import requests

_SIMPLE_DRUG_KEYS = ("drug", "drug_name", "medication", "medication_name")
FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"


def _unique_preserve_order(items: List[str]) -> List[str]:
    """Deduplicate while preserving order and normalizing case/whitespace."""
    seen = set()
    ordered = []
    for item in items:
        if not item:
            continue
        normalized = str(item).strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            ordered.append(normalized)
    return ordered


def _extract_from_codeable_concept(concept: Dict[str, Any]) -> List[str]:
    names = []
    if not isinstance(concept, dict):
        return names
    text_val = concept.get("text")
    if text_val:
        names.append(text_val.strip())
    coding_list = concept.get("coding") or []
    for coding in coding_list:
        display_val = None
        if isinstance(coding, dict):
            display_val = coding.get("display") or coding.get("code")
        if display_val:
            names.append(str(display_val).strip())
    return names


def _extract_from_med_resource(resource: Dict[str, Any]) -> List[str]:
    names = []
    if not isinstance(resource, dict):
        return names

    if "medicationCodeableConcept" in resource:
        names.extend(_extract_from_codeable_concept(resource.get("medicationCodeableConcept")))
    if "medicationReference" in resource and isinstance(resource["medicationReference"], dict):
        display = resource["medicationReference"].get("display")
        if display:
            names.append(display.strip())

    if "code" in resource:
        names.extend(_extract_from_codeable_concept(resource.get("code")))

    identifiers = resource.get("identifier") or []
    if isinstance(identifiers, list):
        for ident in identifiers:
            if isinstance(ident, dict):
                val = ident.get("value")
                if isinstance(val, str):
                    parts = re.split(r"--", val)
                    for part in parts:
                        cleaned = part.replace("_", " ").strip()
                        if cleaned and re.search(r"[A-Za-z]", cleaned):
                            names.append(cleaned)

    ingredients = resource.get("ingredient") or []
    if isinstance(ingredients, list):
        for ing in ingredients:
            if isinstance(ing, dict):
                ref = ing.get("itemReference") or ing.get("item")
                if isinstance(ref, dict):
                    display = ref.get("display")
                    if display:
                        names.append(display.strip())

    if "drug" in resource:
        val = resource.get("drug")
        if isinstance(val, str):
            names.append(val.strip())

    return names


def _walk_resources(obj: Any) -> List[str]:
    names: List[str] = []
    if isinstance(obj, list):
        if all(isinstance(entry, str) for entry in obj):
            names.extend([entry.strip() for entry in obj if entry and str(entry).strip()])
        else:
            for entry in obj:
                names.extend(_walk_resources(entry))
    elif isinstance(obj, dict):
        for key in _SIMPLE_DRUG_KEYS:
            val = obj.get(key)
            if isinstance(val, str):
                names.append(val.strip())
            elif isinstance(val, list) and all(isinstance(v, str) for v in val):
                names.extend([v.strip() for v in val if v and str(v).strip()])

        for key in ["medications", "drugs", "prescriptions"]:
            if key in obj:
                names.extend(_walk_resources(obj[key]))

        if "resourceType" in obj or any(k in obj for k in ["MedicationRequest", "Medication", "MedicationAdministration"]):
            if obj.get("resourceType") in {"MedicationRequest", "MedicationAdministration", "Medication"}:
                names.extend(_extract_from_med_resource(obj))
            else:
                for key, val in obj.items():
                    if key in {"MedicationRequest", "MedicationAdministration", "Medication"}:
                        names.extend(_walk_resources(val))
        else:
            for val in obj.values():
                names.extend(_walk_resources(val))
    return names


def extract_drug_names(data: Any) -> List[str]:
    """Extract drug names from free text, DB rows, or FHIR Medication* resources."""
    if data is None:
        return []

    parsed: Any = data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            try:
                parsed = literal_eval(data)
            except Exception:
                parsed = data

    if isinstance(parsed, (dict, list)):
        names = _walk_resources(parsed)
        return _unique_preserve_order([n for n in names if n])

    text = str(data)
    chunks = re.split(r",|;|\band\b", text, flags=re.IGNORECASE)
    names = [c.strip() for c in chunks if c and c.strip()]
    return _unique_preserve_order(names)


def get_fda_label_section(drug_name: str) -> str:
    """
    Fetch drug interaction / warnings text from OpenFDA for a single drug.
    Returns a condensed string of relevant sections (drug_interactions, boxed_warning, warnings, contraindications).
    """
    if not drug_name:
        return ""

    base_url = FDA_LABEL_URL
    query = f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"'
    params = {"search": query, "limit": 1}

    try:
        time.sleep(0.5)
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code != 200:
            return ""

        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            result = data["results"][0]
            fields_to_check = [
                "drug_interactions",
                "boxed_warning",
                "warnings",
                "contraindications",
            ]

            extracted_text = ""
            for field in fields_to_check:
                if field in result and result[field]:
                    content = " ".join(result[field])
                    extracted_text += f"{field.upper()} --- {content[:1500]}..."

            return extracted_text if extracted_text else "Label found, but no interaction data."

        return ""

    except Exception:
        return ""


def analyze_drug_interactions(drug_names: Any) -> Dict[str, Any]:
    """
    Normalize medication names, fetch FDA label interaction text, and return context for agent-side reasoning.
    - Input: drug_names (list of strings preferred; can also be FHIR/JSON structures or free text)
    - Output: dict with medications_found, fda_label_sections, and optional drug_pairs for pairwise checks
    Note: Requires network access to api.fda.gov.
    """
    if isinstance(drug_names, list):
        meds = _unique_preserve_order([str(d).strip() for d in drug_names if d and str(d).strip()])
    else:
        meds = extract_drug_names(drug_names)

    label_context = {drug: get_fda_label_section(drug) for drug in meds}

    result: Dict[str, Any] = {
        "medications_found": meds,
        "checked_drugs": meds,
        "fda_label_sections": label_context,
    }

    if len(meds) >= 2:
        pairs = []
        for i in range(len(meds)):
            for j in range(i + 1, len(meds)):
                pairs.append((meds[i], meds[j]))
        result["drug_pairs"] = pairs
    else:
        result["note"] = "At least two medications are needed to assess interactions."

    return result
