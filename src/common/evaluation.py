"""
Evaluation pipeline for FHIR Agent Benchmark.

Handles retrieval metrics (precision/recall), answer correctness (LLM-based),
and action validation for MedAgentBench tasks.
"""

import asyncio
import logging

from dateutil import parser as date_parser

from common.eval_metrics import retrieval_recall, retrieval_precision, check_answer_correctness
from common.models import FHIRAgentBenchResult, Task, TaskOutput
from common.utils import format_time

logger = logging.getLogger("fhir_green_agent.evaluation")


def _round(value: float | None, decimals: int = 4) -> float | None:
    return round(value, decimals) if value is not None else None


async def evaluate_results(
        tasks: list[Task],
        time_used: float,
        eval_model: str,
        max_concurrent: int,
) -> FHIRAgentBenchResult:
    """Run full evaluation pipeline and return aggregated results."""

    logger.info("Calculating retrieval metrics")
    _calculate_retrieval_metrics(tasks)

    logger.info("Calculating answer metrics")
    await _calculate_answer_metrics(tasks, eval_model, max_concurrent)

    # Calculate aggregates
    total = len(tasks)
    correct = sum(1 for t in tasks if t.result and t.result.correct == 1)

    precision_values = [t.result.precision for t in tasks if t.result and t.result.precision is not None]
    recall_values = [t.result.recall for t in tasks if t.result and t.result.recall is not None]

    avg_precision = sum(precision_values) / len(precision_values) if precision_values else 0.0
    avg_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0

    f1 = (
        2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )

    logger.info(f"Evaluation complete")

    # Build output
    task_outputs = [
        TaskOutput(
            question_id=task.question_id,
            question=task.question_with_context,
            true_answer=task.true_answer,
            final_answer=task.result.final_answer,
            action_correctness=task.result.action_correctness,
            retrieval_correctness=task.result.retrieval_correctness,
            correct=task.result.correct,
            precision=_round(task.result.precision),
            recall=_round(task.result.recall),
            error=task.result.error,
        )
        for task in tasks
    ]

    return FHIRAgentBenchResult(
        total_tasks=total,
        correct_answers=correct,
        accuracy=_round(correct / total if total > 0 else 0),
        avg_precision=_round(avg_precision),
        avg_recall=_round(avg_recall),
        f1_score=_round(f1),
        time_used=int(time_used),
        task_results=task_outputs,
    )


def _calculate_retrieval_metrics(tasks: list[Task]) -> None:
    """Calculate retrieval precision and recall, excluding action-only tasks."""

    for task in tasks:
        result = task.result
        if not result:
            continue

        # Skip action-only tasks and errored tasks
        if task.task_type == "medagentbench_action" or result.error:
            result.precision = None
            result.recall = None
            continue

        # Extract agent's retrieved IDs for relevant resource types
        agent_ids = []
        for resource_type in task.true_fhir_ids.keys():
            agent_ids.extend(result.retrieved_fhir_ids.get(resource_type, []))

        # Flatten true IDs
        true_ids = sum(task.true_fhir_ids.values(), [])

        result.recall = retrieval_recall(agent_ids, true_ids)
        result.precision = retrieval_precision(agent_ids, true_ids)


async def _calculate_answer_metrics(
        tasks: list[Task],
        model: str,
        max_concurrent: int,
) -> None:
    """Calculate answer correctness using LLM evaluation or action validation."""
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(tasks)
    completed = 0

    async def check_single_answer(task: Task) -> None:
        nonlocal completed
        async with semaphore:
            result = task.result
            if not result:
                completed += 1
                return

            action_correctness = None
            retrieval_correctness = None

            # Action correctness (for action tasks)
            if task.task_type in ("medagentbench_action", "medagentbench_retrieval_action"):
                action_correct, reason = _evaluate_action_task(task.expected_actions, result)
                action_correctness = 1 if action_correct else 0
                if not action_correct:
                    logger.debug(f"[{task.question_id}] Action mismatch: {reason}")

            # Retrieval correctness (for non-action-only tasks)
            if task.task_type != "medagentbench_action":
                if result.error or not result.final_answer:
                    retrieval_correctness = 0
                else:
                    retrieval_correctness = await check_answer_correctness(
                        answer=result.final_answer,
                        ref_answer=str(task.true_answer),
                        question=task.question_with_context,
                        model=model,
                    )

            # Combined correctness
            non_null = [v for v in [action_correctness, retrieval_correctness] if v is not None]
            result.action_correctness = action_correctness
            result.retrieval_correctness = retrieval_correctness
            result.correct = 1 if non_null and all(v == 1 for v in non_null) else 0

            completed += 1
            if completed % 100 == 0 or completed == total:
                logger.info(f"Answer evaluation progress: {completed}/{total}")

    await asyncio.gather(*[check_single_answer(task) for task in tasks])


def _evaluate_action_task(expected_actions: list, result) -> tuple[int, str]:
    """Check if POST requests match expected actions (1=match, 0=mismatch)."""
    post_requests = [
        t["args"] for t in result.tools_used
        if t.get("tool") == "fhir_request_post"
    ]

    if not expected_actions:
        if not post_requests:
            return 1, "ok"
        return 0, f"Expected no POSTs but got {len(post_requests)}"

    if len(post_requests) != len(expected_actions):
        return 0, f"POST count mismatch: got {len(post_requests)}, expected {len(expected_actions)}"

    for expected in expected_actions:
        found = any(
            actual.get("resource_type") == expected.get("resource_type")
            and _dict_match(actual.get("params", {}), expected.get("params", {}))
            for actual in post_requests
        )
        if not found:
            return 0, f"No matching POST for expected: {expected}"

    return 1, "ok"


def _dict_match(actual: dict, expected: dict) -> bool:
    """Check if actual dict contains all expected keys with matching values."""
    for key, expected_val in expected.items():
        actual_key = "note" if key == "note_contains" else key
        if not _values_match(actual.get(actual_key), expected_val, field_name=key):
            return False
    return True


def _values_match(actual, expected, field_name: str = None) -> bool:
    """Compare values with leniency for common format variations."""
    if actual == expected:
        return True
    if actual is None or expected is None:
        return False

    if field_name == "note_contains" and isinstance(expected, list) and isinstance(actual, str):
        return all(substring in actual for substring in expected)

    if field_name == "note" and isinstance(expected, str) and isinstance(actual, str):
        return expected in actual

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(actual) == float(expected)

    if isinstance(expected, str) and isinstance(actual, str):
        try:
            actual_dt = date_parser.parse(actual)
            expected_dt = date_parser.parse(expected)
            return actual_dt.replace(tzinfo=None) == expected_dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            pass
        return actual.strip() == expected.strip()

    if isinstance(expected, dict) and isinstance(actual, dict):
        return _dict_match(actual, expected)

    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return False
        return all(_values_match(a, e) for a, e in zip(actual, expected))

    return False