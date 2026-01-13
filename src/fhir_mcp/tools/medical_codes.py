import os
import pandas as pd
import logging


logger = logging.getLogger(__name__)

# Get the directory where this file is located
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.join(TOOLS_DIR, "codes_dataset")

# Load code tables at module import
CODE_TABLES = {
    'items': pd.read_csv(os.path.join(CODES_DIR, "d_items.csv")),
    'diagnoses': pd.read_csv(os.path.join(CODES_DIR, "d_icd_diagnoses.csv")),
    'procedures': pd.read_csv(os.path.join(CODES_DIR, "d_icd_procedures.csv")),
    'labitems': pd.read_csv(os.path.join(CODES_DIR, "d_labitems.csv"))
}


def lookup_medical_code(search_term: str, code_type: str) -> dict:
    """
    Search local code tables by term.

    Args:
        search_term: Free text search term (e.g., "magnesium", "heart rate")
        code_type: Table to search:
            - "labitems": Laboratory tests (e.g., potassium, glucose, hemoglobin)
            - "items": Chart events (vitals, measurements, inputs)
            - "diagnoses": ICD diagnosis codes
            - "procedures": ICD procedure codes

    Returns:
        dict with "codes": list of {"code": "<code>", "display": "<name>"}
    """
    try:
        logger.debug(f"Looking up medical code {search_term} of type {code_type}")
        if code_type not in CODE_TABLES:
            return {"error": f"Invalid code_type. Must be one of: {list(CODE_TABLES.keys())}"}

        df = CODE_TABLES[code_type]

        # Search in label column (or long_title for ICD codes)
        search_col = 'label' if code_type in ['items', 'labitems'] else 'long_title'
        matches = df[df[search_col].str.contains(search_term, case=False, na=False)]

        # Standardize output format
        code_col = 'itemid' if code_type in ['items', 'labitems'] else 'icd_code'
        codes = [
            {"code": str(row[code_col]), "display": row[search_col]}
            for _, row in matches.iterrows()
        ]

        results = {"codes": codes}

        # Merge into task-scoped storage
        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        mcp_server.merge_task_resources(results)

        logger.debug(f"Code lookup result: {results}")
        return results
    except Exception as e:
        logger.warning(f"Code lookup failed: {e}")
        return {"error": "Code lookup failed"}
