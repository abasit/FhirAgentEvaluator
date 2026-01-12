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

def lookup_medical_code(search_term: str, code_type: str = "items") -> dict:
    """    Search local code tables (items, labitems, diagnoses, procedures) by term.
    - Input: search_term (free text), code_type in {items, labitems, diagnoses, procedures}
    - Output: dict with "Codes": list of matching rows from the table
    """
    try:
        logger.debug(f"Looking up medical code {search_term} of type {code_type}")
        if code_type not in CODE_TABLES:
            return {"error": f"Invalid code_type. Must be one of: {list(CODE_TABLES.keys())}"}

        df = CODE_TABLES[code_type]

        # Search in label column (or long_title for ICD codes)
        search_col = 'label' if code_type in ['items', 'labitems'] else 'long_title'
        results = df[df[search_col].str.contains(search_term, case=False, na=False)]

        results = {
            "Codes": results.to_dict('records'),
        }

        # Merge into task-scoped storage
        from fhir_mcp import get_mcp_server
        mcp_server = get_mcp_server()
        mcp_server.merge_task_resources(results)

        logger.debug(f"Code lookup result: {results}")
        # Convert to dict format
        return results
    except Exception as e:
        logger.warning(f"Code lookup failed: {e}")
        return {"error": str(e)}
