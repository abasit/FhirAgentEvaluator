"""
Task loading and response parsing for FHIR Agent Benchmark.

Handles CSV loading, agent response extraction, and result construction.
"""

import ast
import json
import re

import pandas as pd

from common.models import ConversationState, Task, TaskResult
from fhir_mcp import get_mcp_server


def load_tasks(tasks_file: str, num_tasks: int = 0) -> list[Task]:
    """Load tasks from CSV file."""
    tasks_df = pd.read_csv(tasks_file)[
        ["question_id", "question", "assumption", "patient_fhir_id", "true_answer", "true_fhir_ids", "task_type",
         "expected_actions"]
    ]

    if num_tasks:
        tasks_df = tasks_df[:num_tasks]

    def create_input_str(row):
        input_str = f"Question: {row['question']}\nContext:"
        input_str += f"\nPatient FHIR ID is {row['patient_fhir_id']}."
        if pd.notnull(row['assumption']):
            input_str += f"\n{row['assumption']}"
        return input_str

    def parse_json_field(val, default):
        if pd.isnull(val) or val == "":
            return default
        return ast.literal_eval(val) if isinstance(val, str) else val

    return [
        Task(
            question_id=row["question_id"],
            question_with_context=create_input_str(row),
            true_answer=row["true_answer"],
            true_fhir_ids=parse_json_field(row["true_fhir_ids"], {}),
            task_type=row["task_type"],
            expected_actions=parse_json_field(row["expected_actions"], []),
        )
        for _, row in tasks_df.iterrows()
    ]


def parse_agent_response(response_text: str) -> list:
    """Extract JSON from agent response (supports <json>, ```json```, or raw JSON)."""
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

    return parsed if isinstance(parsed, list) else [parsed]


def make_result(
        state: ConversationState,
        mcp_task_id: str,
        final_answer: str = None,
        error: str = None,
) -> TaskResult:
    """Build TaskResult with retrieved resources and tool logs from MCP server."""
    mcp_server = get_mcp_server()
    retrieved_resources = mcp_server.get_task_resources(mcp_task_id)

    retrieved_resource_ids = {}
    for resource_type, resources in retrieved_resources.items():
        ids = [r["id"] for r in resources if isinstance(r, dict) and "id" in r]
        if ids:
            retrieved_resource_ids[resource_type] = ids

    result = TaskResult(
        final_answer=final_answer,
        trace=state.trace,
        tools_used=mcp_server.get_tool_logs(mcp_task_id),
        retrieved_fhir_ids=retrieved_resource_ids,
        iterations=state.iterations,
        error=error,
    )

    mcp_server.clear_task(mcp_task_id)

    return result


def format_time(seconds: int) -> str:
    """
    Format time in seconds into a human-readable string.
    """
    if seconds < 60:
        return f"{seconds}s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"

    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"
