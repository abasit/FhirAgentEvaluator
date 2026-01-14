"""
FDA drug label fetching tool.

Fetches drug interaction/warning text from the FDA OpenFDA API for a list of drugs.
The agent should use this data to reason about potential drug-drug interactions.
"""

import logging
import time

import requests

logger = logging.getLogger(__name__)

FDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
INFO_FIELDS = ["drug_interactions", "boxed_warning", "warnings", "contraindications", "precautions",
               "general_precautions"]


def get_fda_drug_labels(drug_list: list[str]) -> dict[str, str]:
    """
    Fetch FDA label interaction/warning text for a list of drugs.

    Args:
        drug_list: List of drug names to check (e.g., ["nifedipine", "heparin", "lisinopril"])

    Returns:
        dict mapping each drug name to its FDA label text (interactions, warnings,
        boxed warnings, contraindications), or an error message if fetching fails.
    """
    if not drug_list:
        return {"error": "No drug names provided"}

    logger.info(f"Fetching FDA labels for {len(drug_list)} drugs: {drug_list}")

    # Fetch FDA label for each drug
    labels = {}
    for drug in drug_list:
        try:
            labels[drug] = _fetch_fda_label(drug)
        except Exception as e:
            labels[drug] = f"Error: {e}"
            logger.warning(f"Failed to fetch FDA label for {drug}: {e}")

    return labels


def _fetch_fda_label(drug_name: str) -> str:
    """Fetch drug interaction/warning text from OpenFDA for a single drug."""
    query = f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"'
    params = {"search": query, "limit": 1}

    #rate limiting
    time.sleep(0.5)

    response = requests.get(FDA_LABEL_URL, params=params, timeout=10)
    if response.status_code == 404:
        return "No FDA label found"

    response.raise_for_status()

    results = response.json().get("results", [])
    if not results:
        return "No FDA label found"

    result = results[0]

    sections = []
    for field in INFO_FIELDS:
        content = result.get(field, [])
        if content:
            sections.append("\n".join(content))

    return "\n\n".join(sections) if sections else "No interaction data found"
