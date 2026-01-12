import ast
import asyncio
import logging

import pandas as pd

from common.eval_metrics import retrieval_recall, retrieval_precision, check_answer_correctness
from common.models import FHIRAgentBenchResult, TaskResult

logger = logging.getLogger("fhir_green_agent.evaluation")


async def evaluate_results(
        tasks_df: pd.DataFrame,
        time_used: float,
        eval_model: str,
        max_concurrent: int,
) -> FHIRAgentBenchResult:
    eval_df = tasks_df.copy()

    # Parse string columns to dicts/lists
    eval_df["true_fhir_ids"] = eval_df["true_fhir_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {}
    )
    # eval_df["expected_actions"] = eval_df["expected_actions"].apply(
    #     lambda x: ast.literal_eval(x) if isinstance(x, str) and x else []
    # )

    logger.info("Calculating retrieval metrics")
    eval_df = _calculate_retrieval_metrics(eval_df)

    logger.info("Calculating answer metrics")
    eval_df = await _calculate_answer_metrics(eval_df, eval_model, max_concurrent)

    # Update task results with evaluation metrics
    for idx, row in eval_df.iterrows():
        result: TaskResult = row["result"]
        result.true_answer = row["true_answer"]
        result.correct = int(row.get("answer_correctness", 0))
        result.precision = row.get("precision")
        result.recall = row.get("recall")

    # Calculate summary metrics
    total = len(eval_df)
    correct = int(eval_df["answer_correctness"].sum())
    avg_precision = eval_df["precision"].mean()
    avg_recall = eval_df["recall"].mean()
    f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0
    )

    logger.info(f"Evaluation complete: {correct}/{total} correct ({correct / total * 100:.1f}%)")
    logger.info(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {f1:.4f}")

    return FHIRAgentBenchResult(
        total_tasks=total,
        correct_answers=correct,
        accuracy=correct / total if total > 0 else 0,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        f1_score=f1,
        time_used=time_used,
        task_results=[row["result"] for _, row in eval_df.iterrows()],
    )


def _calculate_retrieval_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate retrieval precision and recall metrics."""

    # Extract agent's retrieved resource IDs from TaskResult
    def extract_agent_resource_ids(row) -> list[str]:
        """Extract FHIR resource IDs from retrieved resources, filtered by true resource types."""
        result: TaskResult = row["result"]
        true_fhir_ids: dict = row["true_fhir_ids"]

        if not result or not result.retrieved_fhir_resources:
            return []

        if not isinstance(true_fhir_ids, dict):
            return []

        resource_ids = []
        for resource_type in true_fhir_ids.keys():
            ids = result.retrieved_fhir_resources.get(resource_type, [])
            resource_ids.extend(ids)

        return resource_ids

    eval_df["agent_resource_ids"] = eval_df.apply(extract_agent_resource_ids, axis=1)

    # Flatten true_fhir_ids to list
    eval_df["true_fhir_ids_list"] = eval_df["true_fhir_ids"].apply(
        lambda d: sum(d.values(), []) if isinstance(d, dict) else []
    )

    # Calculate metrics
    eval_df["recall"] = eval_df.apply(
        lambda row: retrieval_recall(row["agent_resource_ids"], row["true_fhir_ids_list"]),
        axis=1
    )
    eval_df["precision"] = eval_df.apply(
        lambda row: retrieval_precision(row["agent_resource_ids"], row["true_fhir_ids_list"]),
        axis=1
    )

    logger.info(f"Retrieval Precision: {eval_df['precision'].mean():.4f}")
    logger.info(f"Retrieval Recall: {eval_df['recall'].mean():.4f}")

    return eval_df


async def _calculate_answer_metrics(eval_df: pd.DataFrame, model: str, max_concurrent: int) -> pd.DataFrame:
    """Calculate answer correctness using LLM evaluation or action validation."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(eval_df)
    completed = 0

    async def check_single_answer(idx: int, row) -> tuple[int, int]:
        nonlocal completed
        async with semaphore:
            result: TaskResult = row["result"]

            # Action tasks - evaluate POST payloads
            if row.get("task_type") == "action":
                expected_actions = row.get("expected_actions", [])
                correctness = _evaluate_action_task(expected_actions, result)
            # Regular tasks - LLM evaluation
            elif result.error or not result.final_answer:
                correctness = 0
            else:
                correctness = await check_answer_correctness(
                    answer=result.final_answer,
                    ref_answer=row["true_answer"],
                    question=row["question"],
                    model=model,
                )

            completed += 1
            if completed % 10 == 0 or completed == total:
                logger.info(f"Answer evaluation progress: {completed}/{total}")

            return idx, correctness

    tasks = [check_single_answer(idx, row) for idx, row in eval_df.iterrows()]
    results = await asyncio.gather(*tasks)

    for idx, correctness in results:
        eval_df.at[idx, "answer_correctness"] = correctness

    logger.info(f"Answer accuracy: {eval_df['answer_correctness'].mean():.4f}")

    return eval_df


def _evaluate_action_task(expected_actions: list, task_result: TaskResult) -> int:
    """Evaluate action tasks by checking POST payloads. Returns 1 if correct, 0 if not."""
    if not expected_actions:
        return 1  # No expected actions, nothing to check

    # Get actual POST calls from tools_used
    actual_posts = [
        t["args"].get("resource", {})
        for t in task_result.tools_used
        if t["tool"] == "fhir_request_post"
    ]

    if len(actual_posts) != len(expected_actions):
        logger.debug(f"Expected {len(expected_actions)} POST(s), got {len(actual_posts)}")
        return 0

    # Check each expected action has a matching actual POST
    for expected in expected_actions:
        required = expected["required_fields"]

        matched = any(_payloads_match(actual, required) for actual in actual_posts)
        if not matched:
            logger.debug(f"No matching POST for expected: {required.get('code')}")
            return 0

    return 1


def _payloads_match(actual: dict, expected: dict) -> bool:
    """Check if actual payload matches expected, with some leniency."""
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if not _values_match(actual_value, expected_value):
            return False
    return True


def _values_match(actual, expected) -> bool:
    """Compare values with leniency for formats."""
    if actual == expected:
        return True

    if actual is None or expected is None:
        return False

    # Numeric comparison (82 == 82.0)
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(actual) == float(expected)

    # String comparison - try datetime first, then strip whitespace
    if isinstance(expected, str) and isinstance(actual, str):
        # Try datetime comparison
        try:
            from dateutil import parser
            return parser.isoparse(actual) == parser.isoparse(expected)
        except (ValueError, TypeError):
            pass
        # Fall back to stripped string comparison
        return actual.strip() == expected.strip()

    # Recursive for dicts
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _payloads_match(actual, expected)

    # Recursive for lists
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return False
        return all(_values_match(a, e) for a, e in zip(actual, expected))

    return False
