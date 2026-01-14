"""
Evaluation metrics for FHIR Agent Benchmark.

Provides retrieval metrics (precision/recall) and LLM-based answer correctness checking.
"""

import logging

import litellm
import numpy as np

logger = logging.getLogger("fhir_common")


def retrieval_recall(pred: list, true: list) -> float:
    """
    Calculate retrieval recall.

    Returns NaN when no ground truth exists (to exclude from averages).
    """
    pred_set = set(pred)

    if len(true) == 0 and len(pred) == 0:
        return 1.0
    if len(true) == 0 and len(pred) > 0:
        return np.nan
    if len(true) > 0 and len(pred) == 0:
        return 0.0

    return np.mean([t in pred_set for t in true])


def retrieval_precision(pred: list, true: list) -> float:
    """
    Calculate retrieval precision.

    Returns NaN when no predictions made (to exclude from averages).
    """
    true_set = set(true)

    if len(true) == 0 and len(pred) == 0:
        return 1.0
    if len(true) == 0 and len(pred) > 0:
        return 0.0
    if len(true) > 0 and len(pred) == 0:
        return np.nan

    return np.mean([p in true_set for p in pred])


ANSWER_CORRECTNESS_PROMPT = """You are evaluating whether a model's answer to a question is correct by comparing it to the true answer.

Return 1 if correct, 0 if incorrect. Return only 0 or 1, nothing else.

Rules:
- Focus on semantic correctness, not formatting
- Null/empty cases: [] or [[0]] or [[None]] means "no answer" - if model also says no answer, return 1
- Yes/No: [[0]] = No, [[1]] = Yes - match on meaning
- Numbers: Round to nearest integer, ignore units and decimal formatting
- Dates: Ignore time/timezone unless specifically asked
- Lists: Model must include all values, ignore extra context
- Be lenient with brackets, quotes, spacing

Question: {question}
True answer: {ref_answer}
Model answer: {answer}

Return only 0 or 1:"""


async def check_answer_correctness(answer: str, ref_answer: str, question: str, model: str) -> int:
    """Check if agent answer matches reference using LLM evaluation."""
    prompt = ANSWER_CORRECTNESS_PROMPT.format(
        question=question,
        ref_answer=ref_answer,
        answer=answer,
    )

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    result = response.choices[0].message.content.strip()

    if result in ["0", "1"]:
        return int(result)

    logger.warning(f"Unexpected LLM response: {result}, defaulting to 0")
    return 0