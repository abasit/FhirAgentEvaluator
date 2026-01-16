"""
Evaluation pipeline for FHIR Agent Benchmark.

Handles retrieval metrics (precision/recall), answer correctness (LLM-based),
and action validation for MedAgentBench tasks.
"""

import ast
import asyncio
import logging

import numpy as np
import pandas as pd
from dateutil import parser as date_parser

from common.eval_metrics import retrieval_recall, retrieval_precision, check_answer_correctness
from common.models import FHIRAgentBenchResult, TaskResult

logger = logging.getLogger("fhir_green_agent.evaluation")


async def evaluate_results(
        tasks_df: pd.DataFrame,
        time_used: float,
        eval_model: str,
        max_concurrent: int,
) -> FHIRAgentBenchResult:
    """Run full evaluation pipeline and return aggregated results."""
    eval_df = tasks_df.copy()

    eval_df["true_fhir_ids"] = eval_df["true_fhir_ids"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x else {}
    )
    eval_df["expected_actions"] = eval_df["expected_actions"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) and x else []
    )

    logger.info("Calculating retrieval metrics")
    eval_df = _calculate_retrieval_metrics(eval_df)

    logger.info("Calculating answer metrics")
    eval_df = await _calculate_answer_metrics(eval_df, eval_model, max_concurrent)

    for idx, row in eval_df.iterrows():
        result: TaskResult = row["result"]
        result.true_answer = row["true_answer"]
        result.correct = int(row.get("answer_correctness", 0))
        result.true_fhir_ids = row["true_fhir_ids"]
        result.precision = row.get("precision")
        result.recall = row.get("recall")

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
    logger.debug(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1: {f1:.4f}")

    # Slim down task results for output (keep only essential fields)
    keep_fields = {'question_id', 'question', 'final_answer', 'true_answer', 'correct', 'precision', 'recall', 'error'}
    slim_results = [
        TaskResult(**{k: v for k, v in row["result"].model_dump().items() if k in keep_fields})
        for _, row in eval_df.iterrows()
    ]

    return FHIRAgentBenchResult(
        total_tasks=total,
        correct_answers=correct,
        accuracy=correct / total if total > 0 else 0,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        f1_score=f1,
        time_used=time_used,
        task_results=slim_results,
    )


def _calculate_retrieval_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate retrieval precision and recall, excluding action-only tasks."""

    def extract_agent_resource_ids(row) -> list[str]:
        result: TaskResult = row["result"]
        true_fhir_ids: dict = row["true_fhir_ids"]

        if not result or not result.retrieved_fhir_ids:
            return []
        if not isinstance(true_fhir_ids, dict):
            return []

        resource_ids = []
        for resource_type in true_fhir_ids.keys():
            resource_ids.extend(result.retrieved_fhir_ids.get(resource_type, []))
        return resource_ids

    eval_df["agent_resource_ids"] = eval_df.apply(extract_agent_resource_ids, axis=1)
    eval_df["true_fhir_ids_list"] = eval_df["true_fhir_ids"].apply(
        lambda d: sum(d.values(), []) if isinstance(d, dict) else []
    )

    def calc_recall(row):
        if row.get("task_type") == "medagentbench_action":
            return np.nan # Exclude from averages
        result: TaskResult = row["result"]
        if result and result.error:  # ← Add this check
            return np.nan  # Exclude from averages
        return retrieval_recall(row["agent_resource_ids"], row["true_fhir_ids_list"])

    def calc_precision(row):
        if row.get("task_type") == "medagentbench_action":
            return np.nan # Exclude from averages
        result: TaskResult = row["result"]
        if result and result.error:  # ← Add this check
            return np.nan  # Exclude from averages
        return retrieval_precision(row["agent_resource_ids"], row["true_fhir_ids_list"])

    eval_df["recall"] = eval_df.apply(calc_recall, axis=1)
    eval_df["precision"] = eval_df.apply(calc_precision, axis=1)

    logger.debug(f"Retrieval Precision: {eval_df['precision'].mean():.4f}")
    logger.debug(f"Retrieval Recall: {eval_df['recall'].mean():.4f}")

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
            task_type = row.get("task_type", "")
            expected_actions = row.get("expected_actions", [])

            if task_type in ("medagentbench_action", "medagentbench_retrieval_action"):
                action_correct = _evaluate_action_task(expected_actions, result)
                if not action_correct:
                    correctness = 0
                elif task_type == "medagentbench_action":
                    correctness = 0 if result.error else 1
                elif result.error or not result.final_answer:
                    correctness = 0
                else:
                    correctness = await check_answer_correctness(
                        answer=result.final_answer,
                        ref_answer=str(row["true_answer"]),
                        question=row["question"],
                        model=model,
                    )
            else:
                if result.error or not result.final_answer:
                    correctness = 0
                else:
                    correctness = await check_answer_correctness(
                        answer=result.final_answer,
                        ref_answer=row["true_answer"],
                        question=row["question"],
                        model=model,
                    )

            completed += 1
            if completed % 100 == 0 or completed == total:
                logger.info(f"Answer evaluation progress: {completed}/{total}")

            return idx, correctness

    tasks = [check_single_answer(idx, row) for idx, row in eval_df.iterrows()]
    results = await asyncio.gather(*tasks)

    for idx, correctness in results:
        eval_df.at[idx, "answer_correctness"] = correctness

    logger.debug(f"Answer accuracy: {eval_df['answer_correctness'].mean():.4f}")

    return eval_df


def _evaluate_action_task(expected_actions: list, task_result: TaskResult) -> int:
    """Check if POST requests match expected actions (1=match, 0=mismatch)."""
    post_requests = [
        t["args"] for t in task_result.tools_used
        if t.get("tool") == "fhir_request_post"
    ]

    if not expected_actions:
        return 1 if not post_requests else 0

    if len(post_requests) != len(expected_actions):
        logger.debug(f"POST count mismatch: got {len(post_requests)}, expected {len(expected_actions)}")
        return 0

    for expected in expected_actions:
        found = any(
            actual.get("resource_type") == expected.get("resource_type")
            and _dict_match(actual.get("params", {}), expected.get("params", {}))
            for actual in post_requests
        )
        if not found:
            logger.debug(f"No matching POST for expected: {expected}")
            return 0

    return 1


def _dict_match(actual: dict, expected: dict) -> bool:
    """Check if actual dict contains all expected keys with matching values."""
    for key, expected_val in expected.items():
        actual_key = "note" if key == "note_contains" else key
        if not _values_match(actual.get(actual_key), expected_val, field_name=key):
            logger.debug(f"Mismatch for {key}: got {actual.get(actual_key)}, expected {expected_val}")
            return False
    return True


def _values_match(actual, expected, field_name: str = None) -> bool:
    """Compare values with leniency for common format variations."""
    if actual == expected:
        return True
    if actual is None or expected is None:
        return False

    # note_contains: check all substrings present
    if field_name == "note_contains" and isinstance(expected, list) and isinstance(actual, str):
        return all(substring in actual for substring in expected)

    # note: substring match
    if field_name == "note" and isinstance(expected, str) and isinstance(actual, str):
        return expected in actual

    # Numeric comparison
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(actual) == float(expected)

    # String comparison - try datetime parsing
    if isinstance(expected, str) and isinstance(actual, str):
        try:
            actual_dt = date_parser.parse(actual)
            expected_dt = date_parser.parse(expected)
            return actual_dt.replace(tzinfo=None) == expected_dt.replace(tzinfo=None)
        except (ValueError, TypeError):
            pass
        return actual.strip() == expected.strip()

    # Recursive for dicts
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _dict_match(actual, expected)

    # Recursive for lists
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return False
        return all(_values_match(a, e) for a, e in zip(actual, expected))

    return False