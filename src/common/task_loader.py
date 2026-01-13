import json
import logging

import pandas as pd

from common.models import ConversationState, TaskResult
from fhir_mcp import get_mcp_server

logger = logging.getLogger("fhir_green_agent.evaluation")


def load_tasks(tasks_file: str, num_tasks: int = 0) -> pd.DataFrame:
    """Load tasks from file."""
    tasks_df = pd.read_csv(tasks_file)[
        # For FHIR-Agent-Bench
        # ["question_id", "question", "true_answer", "assumption", "patient_fhir_id", "true_fhir_ids"]
        # For drug interaction
        # ["question_id", "question", "true_answer", "assumption", "patient_fhir_id", "true_fhir_ids", "current_medications"]
        # For MedAgentBench
        ["question_id", "question", "true_answer", "assumption", "patient_fhir_id", "true_fhir_ids", "task_type", "expected_actions"]
    ]

    if num_tasks:
        tasks_df = tasks_df[:num_tasks].copy()

    # Add question_with_context
    def create_input_str(row):
        input_str = f"Question: {row['question']}\nContext:"
        input_str += f"\nPatient FHIR ID is {row['patient_fhir_id']}."
        if pd.notnull(row['assumption']):
            input_str += f"\n{row['assumption']}"
        # input_str += f"Current medications include: {row['current_medications']}"
        return input_str

    tasks_df["question_with_context"] = tasks_df.apply(create_input_str, axis=1)

    return tasks_df


def parse_agent_response(response_text: str) -> list:
    import re

    json_str = None
    match = re.search(r'<json>\s*(.*?)\s*</json>', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
            if match:
                json_str = match.group(1)

    if json_str:
        parsed = json.loads(json_str)
    else:
        parsed = json.loads(response_text)

    if isinstance(parsed, list):
        return parsed
    else:
        return [parsed]


def is_final_answer(content: str) -> bool:
    if not content:
        return False
    return "the final answer is:" in content.lower().strip()


def make_result(
        state: ConversationState,
        mcp_task_id: str,
        final_answer: str = None,
        error: str = None,
) -> TaskResult:
    mcp_server = get_mcp_server()

    retrieved_resources = mcp_server.get_task_resources(mcp_task_id)

    # Collect resource IDs by type
    retrieved_resource_ids = {}
    for resource_type, resources in retrieved_resources.items():
        ids = []
        for resource in resources:
            if isinstance(resource, dict) and "id" in resource:
                ids.append(resource["id"])
        if ids:
            retrieved_resource_ids[resource_type] = ids

    result = TaskResult(
        final_answer=final_answer,
        tools_used=mcp_server.get_tool_logs(mcp_task_id),
        retrieved_fhir_resources=retrieved_resource_ids,
        trace=state.trace,
        iterations=state.iterations,
        error=error,
    )

    mcp_server.clear_task(mcp_task_id)

    return result

